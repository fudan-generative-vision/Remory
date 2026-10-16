"""Choose an isolated r-codex home before starting its local model gateway."""
import curses
from contextlib import closing
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
import unicodedata


def clean(value):
    return ' '.join(''.join(c if c.isprintable() else ' ' for c in str(value or '')).split())


def locked(session):
    try:
        with (session / 'session.lock').open('rb') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    except OSError:
        return True
    return False


def running_tmux(session):
    """Locate the pane that owns this session lock, without touching its UI."""
    try:
        if not locked(session):
            return None
        owner = int(json.loads((session / 'session.json').read_text())['pid'])
        proc = Path('/proc') / str(owner)
        lock_path = str((session / 'session.lock').resolve())
        # A stale/reused PID must never redirect the user to an unrelated pane.
        if not any(os.path.realpath(fd) == lock_path for fd in (proc / 'fd').iterdir()):
            return None
        tmux = next((item[5:].decode() for item in (proc / 'environ').read_bytes().split(b'\0')
                     if item.startswith(b'TMUX=')), '')
        if not tmux:
            return None
        socket = tmux.rsplit(',', 2)[0]
        ancestors = set()
        while owner > 1 and owner not in ancestors:
            ancestors.add(owner)
            fields = (Path('/proc') / str(owner) / 'stat').read_text().rsplit(')', 1)[1].split()
            owner = int(fields[1])
        result = subprocess.run(['tmux', '-S', socket, 'list-panes', '-a', '-F',
            '#{session_id}\t#{session_name}\t#{window_id}\t#{pane_id}\t#{pane_pid}'],
            capture_output=True, text=True, timeout=2)
        if result.returncode:
            return None
        for line in result.stdout.splitlines():
            ident, name, window, pane, pid = line.split('\t')
            if int(pid) in ancestors:
                return dict(socket=socket, session=ident, name=name, window=window, pane=pane)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        pass
    return None


def attach_existing(session):
    target = running_tmux(session)
    if target is None:
        return None
    command = ['tmux', '-S', target['socket']]
    print(f"[r-codex] Opening the running session in tmux {target['name']}...", flush=True)
    # Select the existing TUI, never send keystrokes or detach its other clients.
    for operation, ident in (('select-window', target['window']), ('select-pane', target['pane'])):
        result = subprocess.run([*command, operation, '-t', ident], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('The original tmux pane is no longer available')
    from .provider import atomic_json
    atomic_json(session.parent.parent / 'last.json', {'session': session.name})
    current_socket = os.environ.get('TMUX', '').rsplit(',', 2)[0]
    if current_socket == target['socket']:
        result = subprocess.run([*command, 'switch-client', '-E', '-t', target['session']],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Could not switch to the running tmux session: ' + result.stderr.strip())
        return 0
    env = dict(os.environ)
    env.pop('TMUX', None)
    env.pop('TMUX_PANE', None)
    os.execvpe('tmux', [*command, 'attach-session', '-E', '-t', target['session']], env)


def goal_title(session, thread):
    database = session / '.codex/goals_1.sqlite'
    if not thread or not database.exists():
        return ''
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=.5)) as conn:
            row = conn.execute('SELECT objective FROM thread_goals WHERE thread_id = ?', (thread,)).fetchone()
        return clean(row[0])[:180] if row else ''
    except (OSError, sqlite3.Error):
        return ''


def first_message(rollout):
    try:
        with rollout.open('rb') as stream:
            # Only inspect the beginning for a title, not the full conversation.
            for raw in stream.read(256000).splitlines():
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                item = row.get('payload') or {}
                if not isinstance(item, dict):
                    continue
                if row.get('type') != 'response_item' or item.get('type') != 'message' or item.get('role') != 'user':
                    continue
                value = item.get('content', [])
                text = value if isinstance(value, str) else '\n'.join(
                    str(part.get('text', '')) for part in value if isinstance(part, dict))
                if text.startswith(('<environment_context>', '# AGENTS.md', '<permissions instructions>',
                                    '<codex_internal_context', '<turn_aborted>', '<subagent_notification>')):
                    continue
                if clean(text):
                    return clean(text)[:180]
    except OSError:
        pass
    return ''


def sessions_in(root):
    """Only resumable root conversations; tolerate stale/partial manifests."""
    result = []
    for path in Path(root).glob('sessions/*/session.json'):
        folder = path.parent
        if folder.is_symlink() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', folder.name):
            continue
        try:
            manifest = json.loads(path.read_text())
            if not isinstance(manifest, dict):
                continue
            provider_path = folder / 'provider/provider.json'
            provider = json.loads(provider_path.read_text()) if provider_path.exists() else {}
            if not isinstance(provider, dict):
                provider = {}
            thread = manifest.get('thread_id') or provider.get('root_thread_id')
            if not thread or not re.fullmatch(r'[0-9a-fA-F-]{36}', thread):
                continue
            history = list((folder / '.codex/sessions').glob(f'*/*/*/rollout-*{thread}.jsonl'))
            rollout = max(history, key=lambda p: p.stat().st_mtime) if history else None
            if rollout is None:
                continue
            title = first_message(rollout) or goal_title(folder, thread)
            if not title:
                continue  # Persisted empty native threads are not conversations.
            busy = locked(folder)
            target = running_tmux(folder) if busy else None
            state = 'Running' if target else 'In use' if busy else 'Ready'
            updated = max(float(manifest.get('updated_at') or manifest.get('created_at') or 0),
                          rollout.stat().st_mtime)
            result.append(dict(name=folder.name, session=folder.name, thread_id=thread,
                title=title, workspace=clean(manifest['workspace']), updated=updated,
                updated_at=updated, state=state, tmux=target))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(result, key=lambda item: item['updated'], reverse=True)


def fit(text, width):
    output, used = [], 0
    for char in clean(text):
        size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used + size > width:
            break
        output.append(char)
        used += size
    return ''.join(output)


def choose_session(root):
    curses.use_env(False)  # Use the real terminal size, not stale LINES/COLUMNS.
    return curses.wrapper(_screen, Path(root))


def _screen(screen, root):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    curses.noecho()
    curses.raw()  # Handle Ctrl-C locally and restore the terminal on exit.
    screen.keypad(True)
    screen.timeout(500)
    normal, accent, muted, selected = 0, curses.A_BOLD, curses.A_DIM, curses.A_REVERSE
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        accent = curses.color_pair(1) | curses.A_BOLD
    query, cursor, message = '', 0, ''
    items, refreshed = sessions_in(root), time.monotonic()
    visible = []

    def put(row, value, attr=0):
        height, width = screen.getmaxyx()
        if 0 <= row < height and width > 3:
            try:
                screen.addstr(row, 2, fit(value, width - 4), attr)
            except curses.error:
                pass

    while True:
        if time.monotonic() - refreshed > 2:
            previous = visible[cursor]['name'] if 0 <= cursor < len(visible) else None
            items, refreshed = sessions_in(root), time.monotonic()
        else:
            previous = None
        visible = [item for item in items if query.casefold() in
                   (item['name'] + ' ' + item['title'] + ' ' + item['workspace']).casefold()]
        if previous:
            cursor = next((i for i, item in enumerate(visible) if item['name'] == previous), cursor)
        cursor = max(0, min(cursor, len(visible) - 1))
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 12 or width < 32:
            put(0, 'Enlarge the terminal · Esc to cancel', accent)
        else:
            put(1, 'Resume a REMORY session', accent)
            put(3, 'Search: ' + (query or 'type to filter…'), normal if query else muted)
            if message:
                put(4, message, accent)
            capacity = max(1, (height - 9) // 3)
            start = (cursor // capacity) * capacity
            if not visible:
                put(6, 'No matching sessions.' if items else 'No saved sessions. Run r-codex to start one.', muted)
            for n, item in enumerate(visible[start:start + capacity]):
                row, chosen = 5 + 3 * n, start + n == cursor
                attr = selected if chosen else normal if item['state'] in ('Ready', 'Running') else muted
                updated = datetime.fromtimestamp(item['updated']).strftime('%m-%d %H:%M')
                put(row, ('› ' if chosen else '  ') + item['title'], attr)
                status = f"Running in tmux {item['tmux']['name']} · Enter to open" if item['tmux'] else item['state']
                put(row + 1, f"  {item['name']}  ·  {updated}  ·  {status}", muted)
                put(row + 2, '  ' + item['workspace'], muted)
            put(height - 3, f'{len(visible)} session(s)', muted)
            put(height - 2, '↑/↓ select · Enter open · Esc cancel', muted)
        screen.refresh()
        try:
            key = screen.get_wch()
        except curses.error:
            continue
        if key in ('\x1b', '\x03', '\x04'):
            return None
        if key in (curses.KEY_UP, '\x10'):
            cursor = max(0, cursor - 1)
        elif key in (curses.KEY_DOWN, '\x0e'):
            cursor = min(max(0, len(visible) - 1), cursor + 1)
        elif key == curses.KEY_NPAGE:
            cursor += max(1, (height - 9) // 3)
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - max(1, (height - 9) // 3))
        elif key in ('\n', '\r', curses.KEY_ENTER) and visible:
            item = visible[cursor]
            fresh = next((row for row in sessions_in(root) if row['name'] == item['name']), None)
            if fresh is None:
                message = 'This session is no longer available.'
            elif fresh['state'] == 'In use':
                message = 'Running outside tmux. Return to its original terminal.'
            else:
                return item['name']
        elif key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
            query, cursor, message = query[:-1], 0, ''
        elif key == '\x15':
            query, cursor, message = '', 0, ''
        elif isinstance(key, str) and key.isprintable():
            query, cursor, message = query + key, 0, ''
