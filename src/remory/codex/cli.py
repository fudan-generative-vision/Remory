"""Run native Codex with residual compaction on a Remory SGLang server."""
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


def data_root():
    return Path(os.environ.get('REMORY_HOME',
        str(Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'remory'))).expanduser()


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
    for key in list(env):
        if key.startswith(('CODEX_', 'OPENAI_')):
            env.pop(key)
    env.update(CODEX_HOME=str(directory / '.codex'), REMORY_CODEX_TOKEN=token)
    # Keep web/document proxies, but never proxy the local Responses service.
    bypass = set(filter(None, (env.get('NO_PROXY', '') + ',' + env.get('no_proxy', '')).split(',')))
    bypass.update({'localhost', '127.0.0.1', '::1'})
    env['NO_PROXY'] = env['no_proxy'] = ','.join(sorted(bypass))
    return env


def build_catalog(binary, directory, env, limit, display_name):
    result = subprocess.run([str(binary), 'debug', 'models', '--bundled'],
        env=env, capture_output=True, text=True, timeout=30, check=True)
    catalog = json.loads(result.stdout)
    candidates = [m for m in catalog['models'] if m.get('base_instructions')]
    base = next((m for m in candidates if m.get('slug') == 'gpt-5.4'), candidates[0])
    model = {'base_instructions': base['base_instructions']}
    model.update(slug='remory', display_name=display_name, description='Local model with Remory',
        supported_reasoning_levels=[], default_reasoning_level='high', visibility='list',
        supported_in_api=True, priority=0, context_window=limit, max_context_window=limit,
        effective_context_window_percent=100, multi_agent_version='v1', use_responses_lite=False,
        supports_reasoning_summaries=False, support_verbosity=False, shell_type='default',
        apply_patch_tool_type='freeform', input_modalities=['text', 'image'],
        supports_image_detail_original=True,
        experimental_supported_tools=[], include_apps_usage_instructions=False,
        truncation_policy={'mode': 'bytes', 'limit': 10000})
    # The bundled prompt remains upstream-owned and is read from the installed binary.
    model.pop('model_messages', None)
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
        provider.close()
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


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
    if not workspace.is_dir():
        raise ValueError('Working directory does not exist: ' + str(workspace))
    research_tools = args.research_tools or manifest.get('research_tools', False)
    if resuming and research_tools != manifest.get('research_tools', False):
        raise ValueError('Start a new session with --research-tools; existing sessions keep their tool profile')
    tools_venv = args.tools_venv or manifest.get('tools_venv') or os.environ.get('REMORY_TOOLS_VENV')
    if tools_venv:
        tools_venv = str(Path(tools_venv).expanduser().resolve())
    if research_tools:
        require_tools(tools_venv)
    if not resuming:
        directory.mkdir(parents=True, mode=0o700)  # Never overwrite an existing session.
    with (directory / 'session.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This session is already open in another terminal') from None
        (directory / '.codex').mkdir(exist_ok=True, mode=0o700)
        url = args.url or manifest.get('url') or os.environ.get('REMORY_URL', 'http://127.0.0.1:8421')
        print(f'[r-codex] Session: {name}\n[r-codex] Server: {url}', flush=True)
        with httpx.Client(timeout=20, trust_env=False) as http:
            response = http.get(url.rstrip('/') + '/v1/remory/info')
            response.raise_for_status()
            info = response.json()
        if info.get('schema') != 'remory.server.v1':
            raise ValueError('Start the server with python -m remory.launch_server')
        if resuming and manifest.get('identity') != info['identity']:
            raise ValueError('Server model/checkpoint changed; start a new session')
        config = info['config']
        codex = args.codex or manifest.get('codex')
        model_path = args.model_path or manifest.get('model_path')
        binary = ensure_codex(root / 'cache', codex)
        from transformers import AutoTokenizer
        from ..metadata import resolve_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer(config, model_path))
        limit = info['context_limit']
        online = None
        if (getattr(args, 'training_gpus', None) or getattr(args, 'train_python', None)) and not (
                getattr(args, 'online_training', False) or manifest.get('online_training')):
            raise ValueError('Enable --online-training before configuring its GPUs or interpreter')
        if getattr(args, 'online_training', False) or manifest.get('online_training'):
            from ..online.control import Control
            online = Control(directory, info, url)
            online.configure(gpus=getattr(args, 'training_gpus', None),
                             python=getattr(args, 'train_python', None))
        provider = Provider(url, info, tokenizer, directory / 'provider',
                            session_id=manifest['session_id'],
                            max_new_tokens=min(16384, limit // 4), online_control=online)
        token = secrets.token_hex(32)
        env = environment(directory, token)
        tool_servers = research_servers(directory, tools_venv, env=env) if research_tools else {}
        catalog = build_catalog(binary, directory, env, limit, config['target_model'] + ' + Remory')
        with serve(provider, token) as base_url:
            runtime = dict(model='remory', model_provider='remory_local',
                model_catalog_json=str(catalog), model_context_window=limit,
                model_auto_compact_token_limit=min(int(limit * .7), limit - min(32768, limit // 4)),
                model_reasoning_effort='high', check_for_update_on_startup=False,
                approval_policy='on-request', sandbox_mode='workspace-write',
                sandbox_workspace_write={'network_access': True},
                features={'unified_exec': True, 'goals': True, 'multi_agent': True},
                agents={'max_threads': 4, 'max_depth': 1}, web_search='disabled',
                projects={str(workspace): {'trust_level': 'trusted'}},
                model_providers={'remory_local': {
                    # Pinned Codex selects Remote Compaction V2 by this capability name.
                    'name': 'OpenAI', 'base_url': base_url, 'env_key': 'REMORY_CODEX_TOKEN',
                    'wire_api': 'responses', 'requires_openai_auth': False,
                    'request_max_retries': 0, 'stream_max_retries': 0,
                    'stream_idle_timeout_ms': 1800000}})
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
            manifest.update(identity=info['identity'], model=config['target_model'], url=url,
                            updated_at=time.time(), status='running', pid=os.getpid(),
                            research_tools=research_tools, tools_venv=tools_venv,
                            online_training=online is not None,
                            codex=str(binary), model_path=model_path)
            atomic_json(directory / 'session.json', manifest)
            atomic_json(root / 'last.json', {'session': name})
            command = [str(binary)]
            if resuming:
                thread = saved['thread_id']
                if thread:
                    command += ['resume', thread]
                else:
                    raise ValueError('This session has no saved Codex conversation')
            command += ['-C', str(workspace), '--no-alt-screen']
            if args.unsafe:
                command += ['--dangerously-bypass-approvals-and-sandbox']
            if args.prompt:
                command.append(args.prompt)
            print(f'[r-codex] Resume later: r-codex resume {name}', flush=True)
            try:
                if online is not None:
                    from .host import host
                    from .frontend import endpoint
                    with host(directory, binary, workspace, env,
                              saved['thread_id'] if resuming else None, 'remory', args.unsafe) as native:
                        manifest.update(thread_id=native['thread_id'], child_pid=native['child_pid'])
                        atomic_json(directory / 'session.json', manifest)
                        with endpoint(directory, native['endpoint'], control=online) as address:
                            command = [str(binary), 'resume', '--remote', address, native['thread_id'],
                                       '-C', str(workspace), '--no-alt-screen']
                            if args.prompt:
                                command.append(args.prompt)
                            print('[r-codex] REMORY: /remory train · /remory load · /remory status · /remory stop', flush=True)
                            return run_terminal(command, workspace, env, session=name, local_commands=True)
                return run_terminal(command, workspace, env, session=name)
            finally:
                manifest.update(status='closed', updated_at=time.time(),
                                thread_id=provider.meta['root_thread_id'] or manifest.get('thread_id'))
                atomic_json(directory / 'session.json', manifest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'tools':
        parser = argparse.ArgumentParser(prog='r-codex tools',
            description='Check research-tool configuration without making API requests.')
        parser.add_argument('--tools-venv', default=os.environ.get('REMORY_TOOLS_VENV'))
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
    parser.add_argument('--url', help='Remory server URL (default: REMORY_URL or http://127.0.0.1:8421)')
    parser.add_argument('--model-path', help='Local actor/tokenizer directory; no weights are loaded by this client')
    parser.add_argument('--codex', help='Use an existing Codex 0.149.0 executable')
    parser.add_argument('--mcp-config', help='TOML file containing optional mcp_servers')
    parser.add_argument('--research-tools', action='store_true',
                        help='Enable Serper search, Docling, and MinerU (requires tool packages and API keys)')
    parser.add_argument('--tools-venv', help='Separate document-tool venv (or REMORY_TOOLS_VENV)')
    parser.add_argument('--unsafe', action='store_true', help='Disable native approvals and sandbox')
    parser.add_argument('--online-training', action='store_true',
                        help='Collect accepted Qwen compactions and enable local REMORY training commands')
    parser.add_argument('--training-gpus', help='Reserved physical GPU indices/UUIDs (for example 2,3)')
    parser.add_argument('--train-python', help='Python in the separate training environment')
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
    if action == 'new' and (args.training_gpus or args.train_python) and not args.online_training:
        parser.error('--training-gpus and --train-python require --online-training')
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
