"""Run native Codex with text compaction on an OpenAI-compatible chat server."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time

import httpx

from .install import ensure_codex
from .provider import Provider, atomic_json, create_app
from .sessions import attach_existing, choose_session, resolve_session, sessions_in
from .terminal import run_terminal
from .tools import INSTRUCTIONS, require_tools, research_servers, tool_status

DEFAULT_URL = 'http://127.0.0.1:8000/v1'
DEFAULT_CONTEXT_WINDOW = 131072
MODEL_SLUG = 'text-model'
PROVIDER_NAME = 'r_codex_local'


def data_root():
    return Path(os.environ.get('R_CODEX_HOME',
        str(Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) /
            'r-codex-text'))).expanduser()


def session_path(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name):
        raise ValueError('Invalid session name')
    path = root / 'sessions' / name
    if path.is_symlink():
        raise ValueError('A session directory cannot be a symlink')
    return path


def sessions(root):
    return sessions_in(root)


def select_session(root):
    return choose_session(root)


def last_session(root):
    available = sessions(root)
    if not available:
        raise ValueError('No saved sessions. Run r-codex to start one.')
    try:
        name = json.loads((root / 'last.json').read_text())['session']
    except (OSError, ValueError, KeyError, TypeError):
        name = None
    return next((item['session'] for item in available if item['session'] == name),
                available[0]['session'])


def toml(value):
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml(v) for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(toml(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False)


def environment(directory, token):
    env = dict(os.environ)
    # Isolate the native client's login and configuration. Upstream API credentials
    # are read by this launcher, never used for the local Responses connection.
    for key in list(env):
        if key.startswith(('CODEX_', 'OPENAI_')):
            env.pop(key)
    env.update(CODEX_HOME=str(directory / '.codex'), R_CODEX_SESSION_TOKEN=token)
    # Keep web/document proxies, but never proxy the local Responses service.
    bypass = set(filter(None, (env.get('NO_PROXY', '') + ',' + env.get('no_proxy', '')).split(',')))
    bypass.update({'localhost', '127.0.0.1', '::1'})
    env['NO_PROXY'] = env['no_proxy'] = ','.join(sorted(bypass))
    return env


def api_url(value):
    """Accept a server root or its /v1 base without silently storing credentials."""
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL:
        raise ValueError('--url must be a valid HTTP(S) server URL') from None
    if url.scheme not in ('http', 'https') or not url.host:
        raise ValueError('--url must be an HTTP(S) server URL')
    if url.username or url.password or url.query or url.fragment:
        raise ValueError('--url cannot contain credentials, query parameters or fragments; use --api-key-env')
    base = str(url).rstrip('/')
    return base if url.path.rstrip('/').endswith('/v1') else base + '/v1'


def positive_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def discover_context_window(url, model, api_key=None, *, transport=None):
    """Read optional model metadata; /models is not required by every chat server."""
    headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
    try:
        with httpx.Client(timeout=10, trust_env=False, transport=transport) as http:
            response = http.get(url + '/models', headers=headers)
            response.raise_for_status()
            models = response.json().get('data', [])
        for item in models:
            if not isinstance(item, dict) or item.get('id') != model:
                continue
            for field in ('max_model_len', 'context_window', 'max_context_length'):
                value = item.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        pass
    return None


def connection_settings(args, manifest):
    """Saved connections precede environment defaults; resume keeps backend identity."""
    url = api_url(args.url or manifest.get('url') or os.environ.get('R_CODEX_URL', DEFAULT_URL))
    model = args.model or manifest.get('model') or os.environ.get('R_CODEX_MODEL')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('Choose a chat model with --model or R_CODEX_MODEL')
    if args.action == 'resume':
        if manifest.get('url') and url != api_url(manifest['url']):
            raise ValueError('A resumed session must use its saved server URL; start a new session to change --url')
        if manifest.get('model') and model != manifest['model']:
            raise ValueError('A resumed session must use its saved model; start a new session to change --model')
    key_env = args.api_key_env or manifest.get('api_key_env') or 'R_CODEX_API_KEY'
    if not isinstance(key_env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env):
        raise ValueError('--api-key-env must name an environment variable')
    api_key = os.environ.get(key_env) or None
    limit = args.context_window or manifest.get('context_window')
    if limit is None:
        limit = discover_context_window(url, model, api_key)
        if limit is None:
            limit = DEFAULT_CONTEXT_WINDOW
            print(f'[r-codex] Context window metadata unavailable; using {limit} tokens '
                  '(override with --context-window).', file=sys.stderr, flush=True)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 4:
        raise ValueError('Context window must be an integer of at least 4 tokens')
    output = args.max_output_tokens or manifest.get('max_output_tokens') or min(16384, limit // 4)
    if not isinstance(output, int) or isinstance(output, bool) or not 0 < output < limit:
        raise ValueError('--max-output-tokens must be positive and less than the context window')
    reasoning = args.reasoning_effort if args.reasoning_effort is not None else manifest.get('reasoning_effort')
    image_input = (args.image_input if args.image_input is not None
                   else manifest.get('image_input', False))
    return dict(url=url, model=model, context_window=limit, max_output_tokens=output,
                api_key_env=key_env, reasoning_effort=reasoning, image_input=image_input), api_key


def build_catalog(binary, directory, env, limit, display_name, reasoning_effort=None,
                  image_input=False):
    result = subprocess.run([str(binary), 'debug', 'models', '--bundled'],
        env=env, capture_output=True, text=True, timeout=30, check=True)
    catalog = json.loads(result.stdout)
    candidates = [m for m in catalog['models'] if m.get('base_instructions')]
    if not candidates:
        raise RuntimeError('The native Codex catalog contains no base instructions')
    base = next((m for m in candidates if m.get('slug') == 'gpt-5.4'), candidates[0])
    model = {'base_instructions': base['base_instructions']}
    levels = ([{'effort': reasoning_effort, 'description': 'Configured upstream reasoning effort'}]
              if reasoning_effort else [])
    model.update(slug=MODEL_SLUG, display_name=display_name,
        description='OpenAI-compatible text model with native text compaction',
        supported_reasoning_levels=levels, default_reasoning_level=reasoning_effort or 'medium',
        visibility='list', supported_in_api=True, priority=0, context_window=limit,
        max_context_window=limit, effective_context_window_percent=100,
        multi_agent_version='v1', use_responses_lite=False,
        supports_reasoning_summaries=False, support_verbosity=False, shell_type='default',
        apply_patch_tool_type='freeform', input_modalities=['text', 'image'] if image_input else ['text'],
        supports_image_detail_original=False,
        experimental_supported_tools=[], include_apps_usage_instructions=False,
        truncation_policy={'mode': 'bytes', 'limit': 10000})
    # The bundled prompt remains upstream-owned and is read from the installed binary.
    path = directory / '.codex/model-catalog.json'
    atomic_json(path, {'models': [model]})
    return path


@contextmanager
def serve(provider, token):
    import uvicorn
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(provider, token=token),
        host='127.0.0.1', port=port, log_level='error', lifespan='off'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('Could not start the local Responses provider')
            time.sleep(.05)
        yield f'http://127.0.0.1:{port}/v1'
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        provider.close()
        listener.close()


def runtime_config(base_url, catalog, workspace, limit, output, reasoning_effort=None):
    runtime = dict(model=MODEL_SLUG, model_provider=PROVIDER_NAME,
        model_catalog_json=str(catalog), model_context_window=limit,
        model_auto_compact_token_limit=min(int(limit * .7), limit - output),
        check_for_update_on_startup=False,
        approval_policy='on-request', sandbox_mode='workspace-write',
        sandbox_workspace_write={'network_access': True},
        features={'unified_exec': True, 'goals': True, 'multi_agent': True},
        agents={'max_threads': 4, 'max_depth': 1}, web_search='disabled',
        projects={str(workspace): {'trust_level': 'trusted'}},
        model_providers={PROVIDER_NAME: {
            # Native 0.149 selects remote text compaction by this capability name.
            'name': 'OpenAI', 'base_url': base_url, 'env_key': 'R_CODEX_SESSION_TOKEN',
            'wire_api': 'responses', 'requires_openai_auth': False,
            'request_max_retries': 0, 'stream_max_retries': 0,
            'stream_idle_timeout_ms': 1800000}})
    if reasoning_effort is not None:
        runtime['model_reasoning_effort'] = reasoning_effort
    return runtime


def run(args):
    root = data_root().resolve()
    resuming = args.action == 'resume'
    name = args.session
    if resuming and args.last:
        name = last_session(root)
    if resuming and name is None:
        name = select_session(root)
        if name is None:
            return 0
    if resuming:
        name = resolve_session(root, name)
    name = name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(3)
    directory = session_path(root, name)
    saved = None
    if resuming:
        manifest = json.loads((directory / 'session.json').read_text())
        workspace = Path(manifest['workspace'])
        saved = next((item for item in sessions(root) if item['session'] == name), None)
        if saved is None:
            raise ValueError('This session has no saved Codex conversation')
        attached = attach_existing(directory)
        if attached is not None:
            return attached
    else:
        workspace = Path(args.directory or Path.cwd()).expanduser().resolve()
        manifest = {'session': name, 'workspace': str(workspace), 'created_at': time.time(),
                    'title': args.prompt or '', 'session_id': secrets.token_hex(24)}
    # Older native session manifests may not have a provider-side identifier.
    manifest.setdefault('session_id', secrets.token_hex(24))
    if not workspace.is_dir():
        raise ValueError('Working directory does not exist: ' + str(workspace))
    research_tools = args.research_tools or manifest.get('research_tools', False)
    if resuming and research_tools != manifest.get('research_tools', False):
        raise ValueError('Start a new session with --research-tools; existing sessions keep their tool profile')
    tools_venv = args.tools_venv or manifest.get('tools_venv') or os.environ.get('R_CODEX_TOOLS_VENV')
    if tools_venv:
        tools_venv = str(Path(tools_venv).expanduser().resolve())
    if research_tools:
        require_tools(tools_venv)
    settings, api_key = connection_settings(args, manifest)
    if not resuming:
        directory.mkdir(parents=True, mode=0o700)  # Never overwrite an existing session.
    with (directory / 'session.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This session is already open in another terminal') from None
        (directory / '.codex').mkdir(exist_ok=True, mode=0o700)
        print(f'[r-codex] Session: {name}\n[r-codex] Server: {settings["url"]}\n'
              f'[r-codex] Model: {settings["model"]}', flush=True)
        binary = ensure_codex(root / 'cache', args.codex or manifest.get('codex'))
        provider = Provider(settings['url'], settings['model'], directory / 'provider',
                            session_id=manifest['session_id'],
                            max_new_tokens=settings['max_output_tokens'], api_key=api_key,
                            reasoning_effort=settings['reasoning_effort'])
        token = secrets.token_hex(32)
        env = environment(directory, token)
        tool_servers = research_servers(directory, tools_venv, env=env) if research_tools else {}
        limit = settings['context_window']
        catalog = build_catalog(binary, directory, env, limit, settings['model'],
                                settings['reasoning_effort'], settings['image_input'])
        with serve(provider, token) as base_url:
            runtime = runtime_config(base_url, catalog, workspace, limit,
                                     settings['max_output_tokens'], settings['reasoning_effort'])
            mcp_config = args.mcp_config or manifest.get('mcp_config')
            if mcp_config:
                import tomllib
                mcp_path = Path(mcp_config).expanduser().resolve()
                custom_servers = tomllib.loads(mcp_path.read_text()).get('mcp_servers', {})
                duplicate = set(custom_servers) & set(tool_servers)
                if duplicate:
                    raise ValueError('MCP config duplicates bundled research tools: ' + ', '.join(sorted(duplicate)))
                tool_servers.update(custom_servers)
                manifest['mcp_config'] = str(mcp_path)
            if tool_servers:
                runtime['mcp_servers'] = tool_servers
            if research_tools:
                runtime['developer_instructions'] = INSTRUCTIONS
            (directory / '.codex/config.toml').write_text('\n'.join(
                f'{k} = {toml(v)}' for k, v in runtime.items()) + '\n')
            (directory / '.codex/config.toml').chmod(0o600)
            manifest.update(settings, updated_at=time.time(), status='running', pid=os.getpid(),
                            research_tools=research_tools, tools_venv=tools_venv, codex=str(binary))
            atomic_json(directory / 'session.json', manifest)
            atomic_json(root / 'last.json', {'session': name})
            print(f'[r-codex] Resume later: r-codex resume {name}', flush=True)
            try:
                from .host import host
                with host(directory, binary, workspace, env,
                          saved['thread_id'] if resuming else None, MODEL_SLUG, args.unsafe) as native:
                    manifest.update(thread_id=native['thread_id'], child_pid=native['child_pid'])
                    atomic_json(directory / 'session.json', manifest)
                    command = [str(binary), 'resume', '--remote', native['endpoint'], native['thread_id'],
                               '-C', str(workspace), '--no-alt-screen']
                    if args.prompt:
                        command.append(args.prompt)
                    return run_terminal(command, workspace, env, session=name)
            finally:
                manifest.update(status='closed', updated_at=time.time())
                thread = getattr(provider, 'meta', {}).get('root_thread_id')
                if thread:
                    manifest['thread_id'] = thread
                atomic_json(directory / 'session.json', manifest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'tools':
        parser = argparse.ArgumentParser(prog='r-codex tools',
            description='Check research-tool configuration without making API requests.')
        parser.add_argument('--tools-venv', default=os.environ.get('R_CODEX_TOOLS_VENV'))
        args = parser.parse_args(argv[1:])
        try:
            status = tool_status(args.tools_venv)
        except ValueError as exc:
            print(f'r-codex: {exc}', file=sys.stderr)
            return 1
        for name, item in status.items():
            print(f"{name}: {'configured' if item['ready'] else 'missing ' + item['needs']}")
        print('This checks local configuration only; API validity and quota are checked when a tool is used.')
        return 0 if all(item['ready'] for item in status.values()) else 1
    action = argv.pop(0) if argv and argv[0] in {'resume', 'list'} else 'new'
    if action == 'list':
        for item in sessions(data_root()):
            print(f"{item['session']}\t{item['workspace']}\t{item.get('title', '')}")
        return 0
    parser = argparse.ArgumentParser(prog='r-codex' + (' resume' if action == 'resume' else ''),
        description=__doc__)
    parser.add_argument('--url', help='OpenAI-compatible server root or /v1 URL (R_CODEX_URL or ' + DEFAULT_URL + ')')
    parser.add_argument('--model', help='Upstream chat model ID (or R_CODEX_MODEL; saved on resume)')
    parser.add_argument('--context-window', type=positive_integer,
                        help='Context token limit (saved value, /models metadata, or 131072)')
    parser.add_argument('--max-output-tokens', type=positive_integer,
                        help='Maximum generated tokens (default: up to 16384, bounded by context)')
    parser.add_argument('--api-key-env', help='Environment variable holding the upstream API key (default: R_CODEX_API_KEY)')
    parser.add_argument('--reasoning-effort', choices=('none', 'minimal', 'low', 'medium', 'high', 'xhigh'),
                        help='Optional upstream reasoning effort')
    parser.add_argument('--image-input', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable native image tools/input for a vision-capable backend (default: off)')
    parser.add_argument('--codex', help='Use an existing Codex 0.149.0 executable')
    parser.add_argument('--mcp-config', help='TOML file containing optional mcp_servers')
    parser.add_argument('--research-tools', action='store_true',
                        help='Enable Serper search, Docling, and MinerU (requires tool packages and API keys)')
    parser.add_argument('--tools-venv', help='Separate document-tool venv (or R_CODEX_TOOLS_VENV)')
    parser.add_argument('--unsafe', action='store_true', help='Disable native approvals and sandbox')
    parser.add_argument('-C', '--directory', help='Working directory for a new session')
    if action == 'resume':
        parser.add_argument('session', nargs='?', help='Saved r-codex session ID or native thread UUID')
        parser.add_argument('--last', action='store_true')
        parser.add_argument('--prompt')
    else:
        parser.add_argument('--session', help='Name a new session')
        parser.add_argument('prompt', nargs='?')
        parser.set_defaults(last=False)
    args = parser.parse_args(argv)
    args.action = action
    if args.last and args.session:
        parser.error('Choose a session name or --last, not both')
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error('Run r-codex in an interactive terminal')
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        print(f'r-codex: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
