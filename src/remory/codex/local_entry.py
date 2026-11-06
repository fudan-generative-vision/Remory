"""One local launcher; legacy sessions retain their original runtime and state.

The optional deployment file contains paths and endpoints, never credentials.
An ordinary installation still uses remory.codex.cli directly.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shlex
import sys

from . import cli
from .sessions import choose_session, resolve_session, sessions_in


def load_environment(config):
    """Reuse tool credentials as literal values without sourcing executable code."""
    if config.get('tools_venv'):
        os.environ.setdefault('REMORY_TOOLS_VENV', config['tools_venv'])
    path = config.get('tools_env_file')
    if path:
        for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip().removeprefix('export ')
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if key.strip() not in {'SERPER_API_KEY', 'MINERU_API_KEY', 'MINERU_API_TOKEN'}:
                continue
            try:
                values = shlex.split(value, comments=True, posix=True)
                if not separator or len(values) != 1:
                    raise ValueError()
            except ValueError:
                raise ValueError(f'Invalid tools environment file at line {number}') from None
            os.environ.setdefault(key.strip(), values[0])


def legacy_modules(config):
    path = str(Path(config['legacy_harness']).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module('session_registry'), importlib.import_module('session_picker')


def entries(config, root):
    result = []
    _, picker = legacy_modules(config)
    for kind, values in (('remory', sessions_in(root)),
                         ('legacy', picker.sessions_in(Path(config['legacy_sessions'])))):
        for value in values:
            name = value.get('session') or value['name']
            result.append({**value, 'name': kind + ':' + name, 'session': name, 'kind': kind})
    return sorted(result, key=lambda item: item['updated'], reverse=True)


def resolve(config, root, name):
    selected = None
    if ':' in name:
        selected, name = name.split(':', 1)
        if selected not in {'remory', 'legacy'}:
            raise ValueError('Use remory:SESSION or legacy:SESSION')
    candidates = []
    if selected != 'legacy':
        try:
            candidates.append(('remory', resolve_session(root, name)))
        except ValueError as exc:
            if not str(exc).startswith('No saved r-codex session matches'):
                raise
    if selected != 'remory':
        registry, _ = legacy_modules(config)
        try:
            _, actual, _ = registry.resolve_session(Path(config['legacy_sessions']), name)
            candidates.append(('legacy', actual))
        except ValueError as exc:
            if not str(exc).startswith('No saved r-codex session matches'):
                raise
    if len(candidates) != 1:
        raise ValueError('Session is missing or ambiguous; use r-codex list and its prefixed session ID')
    return candidates[0]


def last(config, root):
    import hashlib
    import socket
    key = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]
    old = Path(config['legacy_sessions'])
    pointers = [('remory', root / 'last.json'), ('legacy', old / 'last.json'),
                ('legacy', old / ('last-' + key + '.json'))]
    valid = {item['name']: item for item in entries(config, root)}
    found = []
    for kind, path in pointers:
        try:
            name = kind + ':' + json.loads(path.read_text())['session']
            if name in valid:
                found.append((path.stat().st_mtime, name))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if found:
        return max(found)[1]
    if valid:
        return next(iter(valid))
    raise ValueError('No saved sessions. Run r-codex to start one.')


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(os.environ.get('REMORY_LAUNCHER_CONFIG',
                Path(__file__).resolve().parents[3] / '.remory/launcher.json'))
    if not path.exists():
        return cli.main(args)
    config = json.loads(path.read_text())
    root = Path(os.environ.get('REMORY_HOME', config['remory_home'])).expanduser().resolve()
    os.environ['REMORY_HOME'] = str(root)
    if args[:1] == ['--legacy']:
        command = [*config['legacy_command'], *args[1:]]
        return os.execv(command[0], command)
    if args[:1] == ['list']:
        for item in entries(config, root):
            print(f"{item['name']}\t{item['workspace']}\t{item['title']}")
        return 0
    if args[:1] == ['resume'] and not any(x in args for x in ('--help', '-h')):
        parser = argparse.ArgumentParser(prog='r-codex resume')
        parser.add_argument('session', nargs='?')
        parser.add_argument('--last', action='store_true')
        parser.add_argument('--unsafe', action='store_true')
        parser.add_argument('--prompt')
        options, extra = parser.parse_known_args(args[1:])
        if options.session and options.last:
            parser.error('Choose a session name or --last, not both')
        name = last(config, root) if options.last else options.session
        if not name:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                parser.error('Choose a session from r-codex list or use --last')
            name = choose_session(root, loader=lambda: entries(config, root))
            if name is None:
                return 0
        kind, name = resolve(config, root, name)
        args = ['resume', name, *extra]
        if options.unsafe:
            args.append('--unsafe')
        if options.prompt is not None:
            args.extend(['--prompt', options.prompt])
        if kind == 'legacy':
            command = [*config['legacy_command'], *args]
            return os.execv(command[0], command)
        load_environment(config)
        return cli.main(args)  # Resume uses its saved URL, never a new-session default.
    load_environment(config)
    research_tools = config.get('research_tools', False)
    if '--no-research-tools' in args:
        args.remove('--no-research-tools')
        research_tools = False
    if not any(x in args for x in ('--help', '-h')) and args[:1] != ['tools']:
        if research_tools and '--research-tools' not in args:
            args.append('--research-tools')
        use_default = ('--url' not in args and not any(x.startswith('--url=') for x in args)
                       and not os.environ.get('REMORY_URL'))
        if use_default:
            args = ['--url', config['default_url'], *args]
        for flag, key in (('--codex', 'codex'), ('--model-path', 'model_path')):
            if key == 'model_path' and not use_default:
                continue  # A different server may use another tokenizer/model family.
            if config.get(key) and flag not in args and not any(x.startswith(flag + '=') for x in args):
                args += [flag, config[key]]
    return cli.main(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print('r-codex: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
