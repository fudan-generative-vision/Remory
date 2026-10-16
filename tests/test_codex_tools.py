from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import urllib.error

import pytest

from remory.codex import cli, search, tools


def test_research_profile_passes_keys_only_through_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'executable', lambda name, _: '/tools/bin/' + name)
    env = {'SERPER_API_KEY': 'test-serper-secret', 'MINERU_API_KEY': 'test-mineru-secret'}
    servers = tools.research_servers(tmp_path, '/tools', env=env)
    encoded = 'mcp_servers = ' + cli.toml(servers)
    assert tomllib.loads(encoded)['mcp_servers'] == servers
    assert 'test-serper-secret' not in encoded and 'test-mineru-secret' not in encoded
    assert env['MINERU_API_TOKEN'] == env['MINERU_API_KEY']
    assert servers['mineru']['env_vars'][0] == 'MINERU_API_TOKEN'
    assert 'HTTPS_PROXY' in servers['serper']['env_vars']
    assert servers['docling']['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert servers['docling']['env']['DOCLING_DEVICE'] == 'cpu'


def test_missing_tools_and_conflicting_keys_are_actionable_and_redacted(monkeypatch, capsys):
    for name in ('SERPER_API_KEY', 'MINERU_API_KEY', 'MINERU_API_TOKEN', 'REMORY_TOOLS_VENV'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tools, 'executable', lambda *args: None)
    with pytest.raises(ValueError, match='SERPER_API_KEY.*Docling.*MINERU_API_KEY'):
        tools.require_tools()
    assert cli.main(['tools']) == 1  # No terminal, model, or HTTP server needed.
    assert 'Docling: missing' in capsys.readouterr().out
    with pytest.raises(ValueError, match='differ') as exc:
        tools.mineru_token({'MINERU_API_KEY': 'private-one', 'MINERU_API_TOKEN': 'private-two'})
    assert 'private-one' not in str(exc.value)
    assert tools.mineru_token({'MINERU_API_TOKEN': 'native-token'}) == 'native-token'


def test_search_request_and_bounded_results(monkeypatch):
    monkeypatch.setenv('SERPER_API_KEY', 'test-search-secret')
    seen = []
    def urlopen(request, **kwargs):
        seen.append((request, kwargs))
        return BytesIO(json.dumps({'organic': [
            {'title': 'Paper ' + str(i), 'link': 'https://example.com/' + str(i),
             'snippet': 's' * 10000} for i in range(8)]}).encode())
    monkeypatch.setattr(search.urllib.request, 'urlopen', urlopen)
    result = search.search({'query': 'flow drafter', 'num_results': 2})
    assert 'Paper 1' in result and 'Paper 2' not in result
    assert len(result) < 9000 and '[begin untrusted web search results]' in result
    assert 'test-search-secret' not in result
    request, kwargs = seen[0]
    assert request.get_header('X-api-key') == 'test-search-secret'
    assert json.loads(request.data) == {'q': 'flow drafter', 'num': 2}
    assert kwargs['timeout'] == 60


def test_search_remote_errors_do_not_echo_credentials(monkeypatch):
    monkeypatch.setenv('SERPER_API_KEY', 'test-search-secret')
    def urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(search.ENDPOINT, 403, 'test-search-secret', {},
                                     BytesIO(b'test-search-secret'))
    monkeypatch.setattr(search.urllib.request, 'urlopen', urlopen)
    response = search.dispatch({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
        'params': {'name': 'search', 'arguments': {'query': 'test'}}})
    assert response['result']['isError']
    assert 'HTTP 403' in json.dumps(response) and 'test-search-secret' not in json.dumps(response)


def test_serper_stdio_handshake_and_validation():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    env.pop('SERPER_API_KEY', None)
    messages = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-03-26'}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
            'name': 'search', 'arguments': {'query': 'test'}}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {
            'name': 'search', 'arguments': {'query': 'test', 'num_results': 100}}},
        {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': None}]
    result = subprocess.run([sys.executable, '-m', 'remory.codex.search'], env=env,
        input='\n'.join(map(json.dumps, messages)) + '\n', capture_output=True, text=True,
        timeout=20, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row['id'] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]['result']['serverInfo']['name'] == 'remory-serper'
    assert rows[1]['result']['tools'][0]['name'] == 'search'
    assert 'SERPER_API_KEY' in rows[2]['result']['content'][0]['text']
    assert rows[3]['result']['content'][0]['text'] == 'Invalid search arguments'
    assert rows[4]['error']['code'] == -32602
