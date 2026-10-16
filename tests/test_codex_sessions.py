import curses
import fcntl
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from remory.codex import cli, sessions


def saved_session(root, name='saved', *, text='检查这个项目', provider_thread=False):
    folder = root / 'sessions' / name
    history = folder / '.codex/sessions/2026/10/16'
    history.mkdir(parents=True)
    thread = '11111111-1111-1111-1111-111111111111'
    manifest = {'workspace': str(root), 'updated_at': 1, 'pid': 999999999}
    if provider_thread:
        (folder / 'provider').mkdir()
        (folder / 'provider/provider.json').write_text(json.dumps({'root_thread_id': thread}))
    else:
        manifest['thread_id'] = thread
    (folder / 'session.json').write_text(json.dumps(manifest))
    rollout = history / f'rollout-test-{thread}.jsonl'
    rows = [None, {'type': 'response_item', 'payload': 'partial'},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                'content': '<environment_context>ignore</environment_context>'}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': text}]}}]
    rollout.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return folder, rollout


def test_discovery_filters_empty_deleted_and_stale_sessions(tmp_path):
    folder, rollout = saved_session(tmp_path, provider_thread=True)
    saved_session(tmp_path, 'empty', text='')
    saved_session(tmp_path, 'deleted')[1].unlink()
    (tmp_path / 'sessions/link').symlink_to(folder, target_is_directory=True)
    corrupt = tmp_path / 'sessions/corrupt'
    corrupt.mkdir()
    (corrupt / 'session.json').write_text('null')
    before = (folder / 'session.json').read_bytes()
    with (folder / 'session.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        items = sessions.sessions_in(tmp_path)
        assert len(items) == 1 and items[0]['state'] == 'In use'
        assert sessions.running_tmux(folder) is None  # Stale PID is not an attachment target.
    item, = sessions.sessions_in(tmp_path)
    assert item['title'] == '检查这个项目' and item['state'] == 'Ready'
    assert (folder / 'session.json').read_bytes() == before
    rollout.unlink()
    assert sessions.sessions_in(tmp_path) == []


@pytest.mark.parametrize('enter', ['\r', '\n', curses.KEY_ENTER])
def test_picker_unicode_search_resize_and_enter(tmp_path, monkeypatch, enter):
    saved_session(tmp_path, 'a', text='English task')
    saved_session(tmp_path, 'b', text='检查这个项目')
    keys = iter([curses.KEY_NPAGE, curses.KEY_RESIZE, '检', '查', enter])
    ticks = itertools.count(step=3)
    monkeypatch.setattr(sessions.time, 'monotonic', lambda: next(ticks))
    for name in ('curs_set', 'noecho', 'raw'):
        monkeypatch.setattr(curses, name, lambda *args: None)
    monkeypatch.setattr(curses, 'has_colors', lambda: False)
    screen = SimpleNamespace(keypad=lambda *_: None, timeout=lambda *_: None,
        getmaxyx=lambda: (24, 90), erase=lambda: None, addstr=lambda *_: None,
        refresh=lambda: None, get_wch=lambda: next(keys))
    assert sessions._screen(screen, tmp_path) == 'b'
    assert sessions.fit('中英文abc', 5) == '中英'


def test_last_skips_deleted_pointer_and_empty_thread(tmp_path):
    saved_session(tmp_path, 'valid')
    saved_session(tmp_path, 'empty', text='')
    (tmp_path / 'last.json').write_text(json.dumps({'session': 'empty'}))
    assert cli.last_session(tmp_path) == 'valid'
    (tmp_path / 'last.json').write_text('broken')
    assert cli.last_session(tmp_path) == 'valid'


def test_resume_cancel_and_tmux_attach_do_not_start_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'data_root', lambda: tmp_path)
    monkeypatch.setattr(cli, 'select_session', lambda _: None)
    args = SimpleNamespace(action='resume', session=None, last=False)
    assert cli.run(args) == 0
    saved_session(tmp_path)
    args.session = 'saved'
    monkeypatch.setattr(cli, 'attach_existing', lambda _: 0)
    assert cli.run(args) == 0  # Other startup arguments are intentionally absent.


def test_tool_profile_cannot_change_on_resume(tmp_path, monkeypatch):
    saved_session(tmp_path)
    monkeypatch.setattr(cli, 'data_root', lambda: tmp_path)
    monkeypatch.setattr(cli, 'attach_existing', lambda _: None)
    args = SimpleNamespace(action='resume', session='saved', last=False, research_tools=True)
    with pytest.raises(ValueError, match='Start a new session'):
        cli.run(args)


def test_proxy_environment_preserves_web_access_bypasses_local_provider(tmp_path, monkeypatch):
    monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.example:8080')
    monkeypatch.setenv('NO_PROXY', 'private.example')
    monkeypatch.setenv('no_proxy', 'other.example')
    monkeypatch.setenv('OPENAI_API_KEY', 'unrelated-account')
    env = cli.environment(tmp_path, 'local-token')
    assert env['HTTPS_PROXY'] == 'http://proxy.example:8080'
    assert {'localhost', '127.0.0.1', '::1', 'private.example', 'other.example'} <= set(env['NO_PROXY'].split(','))
    assert env['NO_PROXY'] == env['no_proxy']
    assert 'OPENAI_API_KEY' not in env
    assert Path(env['CODEX_HOME']) == tmp_path / '.codex'
