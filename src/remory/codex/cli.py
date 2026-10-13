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
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx

from .install import ensure_codex
from .provider import Provider, atomic_json, create_app


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
    found = []
    for path in (root / 'sessions').glob('*/session.json'):
        if path.parent.is_symlink():
            continue
        try:
            item = json.loads(path.read_text())
            found.append(item)
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda x: x.get('updated_at', 0), reverse=True)


def select_session(root):
    import curses
    rows = sessions(root)
    if not rows:
        raise ValueError('No saved sessions. Run r-codex to start one.')
    def screen(window):
        curses.curs_set(0)
        window.keypad(True)
        selected = 0
        while True:
            window.erase()
            height, width = window.getmaxyx()
            window.addnstr(0, 0, 'Remory sessions  ·  ↑/↓ select  ·  Enter resume  ·  Esc cancel',
                           max(1, width - 1), curses.A_BOLD)
            offset = max(0, selected - max(1, height - 4))
            for i, item in enumerate(rows[offset:offset + max(1, height - 2)], offset):
                line = f"{item['session']}  {item.get('title') or item['workspace']}"
                line = ' '.join(''.join(c if c.isprintable() else ' ' for c in line).split())
                window.addnstr(i - offset + 2, 0, line, max(1, width - 1),
                               curses.A_REVERSE if i == selected else 0)
            window.refresh()
            key = window.getch()
            if key in (27, 3, ord('q')):
                return None
            if key in (10, 13, curses.KEY_ENTER):
                return rows[selected]['session']
            if key in (curses.KEY_UP, ord('k')):
                selected = max(0, selected - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                selected = min(len(rows) - 1, selected + 1)
    return curses.wrapper(screen)


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


def run_terminal(command, workspace, env):
    child = None
    previous = {}
    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise KeyboardInterrupt
    try:
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, lambda *_: None)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, terminate)
        child = subprocess.Popen(command, cwd=workspace, env=env)
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(args):
    root = data_root().resolve()
    resuming = args.action == 'resume'
    name = args.session
    if resuming and args.last:
        name = json.loads((root / 'last.json').read_text())['session']
    if resuming and name is None:
        name = select_session(root)
        if name is None:
            return 0
    name = name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(3)
    directory = session_path(root, name)
    if resuming:
        manifest = json.loads((directory / 'session.json').read_text())
        workspace = Path(manifest['workspace'])
    else:
        workspace = Path(args.directory or Path.cwd()).expanduser().resolve()
        manifest = {'session': name, 'workspace': str(workspace), 'created_at': time.time(),
                    'title': args.prompt or '', 'session_id': secrets.token_hex(24)}
    if not workspace.is_dir():
        raise ValueError('Working directory does not exist: ' + str(workspace))
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
        binary = ensure_codex(root / 'cache', args.codex)
        from transformers import AutoTokenizer
        from ..metadata import resolve_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer(config, args.model_path))
        limit = info['context_limit']
        provider = Provider(url, info, tokenizer, directory / 'provider',
                            session_id=manifest['session_id'],
                            max_new_tokens=min(16384, limit // 4))
        token = secrets.token_hex(32)
        env = environment(directory, token)
        catalog = build_catalog(binary, directory, env, limit, config['target_model'] + ' + Remory')
        with serve(provider, token) as base_url:
            runtime = dict(model='remory', model_provider='remory_local',
                model_catalog_json=str(catalog), model_context_window=limit,
                model_auto_compact_token_limit=min(int(limit * .7), limit - min(32768, limit // 4)),
                model_reasoning_effort='high', check_for_update_on_startup=False,
                approval_policy='on-request', sandbox_mode='workspace-write',
                sandbox_workspace_write={'network_access': True},
                features={'unified_exec': True, 'goals': True, 'multi_agent': True,
                          'api_key_model_discovery': False},
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
                runtime['mcp_servers'] = tomllib.loads(mcp_path.read_text()).get('mcp_servers', {})
                manifest['mcp_config'] = str(mcp_path)
            (directory / '.codex/config.toml').write_text('\n'.join(
                f'{k} = {toml(v)}' for k, v in runtime.items()) + '\n')
            (directory / '.codex/config.toml').chmod(0o600)
            manifest.update(identity=info['identity'], model=config['target_model'], url=url,
                            updated_at=time.time(), status='running')
            atomic_json(directory / 'session.json', manifest)
            atomic_json(root / 'last.json', {'session': name})
            command = [str(binary)]
            if resuming:
                thread = provider.meta['root_thread_id']
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
                return run_terminal(command, workspace, env)
            finally:
                manifest.update(status='closed', updated_at=time.time(),
                                thread_id=provider.meta['root_thread_id'])
                atomic_json(directory / 'session.json', manifest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument('--unsafe', action='store_true', help='Disable native approvals and sandbox')
    parser.add_argument('-C', '--directory', help='Working directory for a new session')
    if action == 'resume':
        parser.add_argument('session', nargs='?')
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
