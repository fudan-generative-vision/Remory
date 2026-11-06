import json
from types import SimpleNamespace

import pytest

from remory.codex import local_entry
from test_codex_sessions import saved_session


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'new'
    config = dict(remory_home=str(root), default_url='http://new-default',
        model_path='/default-tokenizer', codex='/codex', legacy_sessions=str(tmp_path / 'old'),
        legacy_command=['/old-python', '/old-entry'])
    path = tmp_path / 'launcher.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv('REMORY_LAUNCHER_CONFIG', str(path))
    monkeypatch.delenv('REMORY_HOME', raising=False)
    monkeypatch.delenv('REMORY_URL', raising=False)
    old = {'old-only', 'collision'}
    def resolve(_root, name):
        if name not in old:
            raise ValueError('No saved r-codex session matches ' + name)
        return 'interactive', name, tmp_path / 'old' / name
    monkeypatch.setattr(local_entry, 'legacy_modules', lambda cfg: (
        SimpleNamespace(resolve_session=resolve), SimpleNamespace(sessions_in=lambda _: [])))
    invoked = []
    monkeypatch.setattr(local_entry.cli, 'main', lambda args: invoked.append(('remory', args)))
    monkeypatch.setattr(local_entry.os, 'execv', lambda executable, args: invoked.append(('legacy', args)))
    return config, root, invoked


def test_resume_selects_saved_owner_and_never_new_defaults(setup):
    config, root, invoked = setup
    saved_session(root, 'new-only')
    local_entry.main(['resume', 'new-only'])
    assert invoked == [('remory', ['resume', 'new-only'])]
    invoked.clear()
    # execv does not return in production; verify the selected command directly.
    kind, name = local_entry.resolve(config, root, 'old-only')
    assert (kind, name) == ('legacy', 'old-only')
    local_entry.main(['resume', 'old-only', '--prompt', 'Continue'])
    assert invoked == [('legacy', ['/old-python', '/old-entry', 'resume', 'old-only', '--prompt', 'Continue'])]


def test_collisions_require_explicit_runtime(setup):
    config, root, _ = setup
    saved_session(root, 'collision')
    with pytest.raises(ValueError, match='ambiguous'):
        local_entry.resolve(config, root, 'collision')
    assert local_entry.resolve(config, root, 'remory:collision') == ('remory', 'collision')
    assert local_entry.resolve(config, root, 'legacy:collision') == ('legacy', 'collision')


def test_new_sessions_use_default_but_explicit_server_keeps_its_tokenizer(setup):
    _, _, invoked = setup
    local_entry.main(['-C', '/work'])
    args = invoked[-1][1]
    assert args[:2] == ['--url', 'http://new-default']
    assert '--model-path' in args and '/default-tokenizer' in args
    local_entry.main(['--url', 'http://another-model'])
    assert '--model-path' not in invoked[-1][1]


def test_unified_list_preserves_runtime_namespaces(setup, monkeypatch):
    config, root, _ = setup
    saved_session(root, 'collision')
    registry, _ = local_entry.legacy_modules(config)
    old = dict(name='collision', title='old', workspace='/old', updated=1, state='Ready', tmux=None)
    monkeypatch.setattr(local_entry, 'legacy_modules', lambda _: (
        registry, SimpleNamespace(sessions_in=lambda _: [old])))
    names = {entry['name'] for entry in local_entry.entries(config, root)}
    assert names == {'remory:collision', 'legacy:collision'}


def test_credentials_are_literal_whitelisted_and_do_not_override_environment(tmp_path, monkeypatch):
    path = tmp_path / 'keys'
    path.write_text('SERPER_API_KEY="$(do-not-execute)"\nMINERU_API_KEY=from-file\nPYTHONPATH=/bad\n')
    monkeypatch.delenv('SERPER_API_KEY', raising=False)
    monkeypatch.setenv('MINERU_API_KEY', 'existing')
    before = local_entry.os.environ.get('PYTHONPATH')
    local_entry.load_environment({'tools_env_file': str(path)})
    assert local_entry.os.environ['SERPER_API_KEY'] == '$(do-not-execute)'
    assert local_entry.os.environ['MINERU_API_KEY'] == 'existing'
    assert local_entry.os.environ.get('PYTHONPATH') == before
