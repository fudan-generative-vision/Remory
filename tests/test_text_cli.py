import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from r_codex import cli


def arguments(**changes):
    values = dict(action='new', session='sample', last=False, directory=None, prompt=None,
                  url=None, model=None, context_window=None, max_output_tokens=None,
                  api_key_env=None, reasoning_effort=None, research_tools=False,
                  tools_venv=None, codex=None, mcp_config=None, unsafe=False, image_input=None)
    values.update(changes)
    return SimpleNamespace(**values)


class TextConnectionTest(unittest.TestCase):
    def test_home_is_independent_of_remory_and_native_codex(self):
        with patch.dict(os.environ, {'XDG_DATA_HOME': '/tmp/text-data',
                                    'REMORY_HOME': '/private/remory',
                                    'CODEX_HOME': '/private/codex'}, clear=True):
            self.assertEqual(cli.data_root(), Path('/tmp/text-data/r-codex-text'))
        with patch.dict(os.environ, {'R_CODEX_HOME': '/tmp/explicit-home'}, clear=True):
            self.assertEqual(cli.data_root(), Path('/tmp/explicit-home'))

    def test_model_required_and_legacy_environment_is_not_used(self):
        with patch.dict(os.environ, {'REMORY_URL': 'http://legacy:8421'}, clear=True):
            with self.assertRaisesRegex(ValueError, '--model'):
                cli.connection_settings(arguments(context_window=32000), {})

    def test_saved_connection_wins_over_environment_without_secrets_on_disk(self):
        saved = dict(url='https://saved.example/v1', model='saved-model',
                     context_window=65536, max_output_tokens=8192,
                     api_key_env='SAVED_KEY', reasoning_effort='high', image_input=False)
        with patch.dict(os.environ, {'R_CODEX_URL': 'http://other:8000',
                                    'R_CODEX_MODEL': 'different-model',
                                    'SAVED_KEY': 'private-key'}, clear=True):
            actual, api_key = cli.connection_settings(arguments(), saved)
        self.assertEqual(actual, saved)
        self.assertEqual(api_key, 'private-key')
        self.assertNotIn('private-key', json.dumps(actual))

    def test_explicit_options_override_saved_connection(self):
        saved = dict(url='https://old.example/v1', model='old-model', context_window=65536,
                     max_output_tokens=8192, api_key_env='OLD_KEY', reasoning_effort='high')
        args = arguments(url='https://new.example/', model='new-model', context_window=32768,
                         max_output_tokens=4096, api_key_env='NEW_KEY', reasoning_effort='none')
        with patch.dict(os.environ, {'NEW_KEY': 'new-secret'}, clear=True):
            actual, api_key = cli.connection_settings(args, saved)
        self.assertEqual(actual, dict(url='https://new.example/v1', model='new-model',
            context_window=32768, max_output_tokens=4096, api_key_env='NEW_KEY',
            reasoning_effort='none', image_input=False))
        self.assertEqual(api_key, 'new-secret')

    def test_resume_requires_same_backend_but_accepts_normalized_url(self):
        saved = dict(url='https://saved.example/v1', model='saved-model', context_window=65536)
        for override in [dict(model='different-model'), dict(url='https://other.example')]:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, 'start a new session'):
                cli.connection_settings(arguments(action='resume', **override), saved)
        settings, _ = cli.connection_settings(arguments(action='resume', url='https://saved.example/',
            max_output_tokens=4096, reasoning_effort='low'), saved)
        self.assertEqual(settings['url'], saved['url'])
        self.assertEqual(settings['max_output_tokens'], 4096)

    def test_image_capability_is_opt_in_and_saved_for_resume(self):
        saved = dict(url='https://saved.example/v1', model='vlm', context_window=65536, image_input=True)
        settings, _ = cli.connection_settings(arguments(action='resume'), saved)
        self.assertTrue(settings['image_input'])
        settings, _ = cli.connection_settings(arguments(action='resume', image_input=False), saved)
        self.assertFalse(settings['image_input'])

    def test_context_discovery_matches_requested_model_and_authenticates(self):
        def respond(request):
            self.assertEqual(str(request.url), 'https://example.test/v1/models')
            self.assertEqual(request.headers['authorization'], 'Bearer discovery-secret')
            return httpx.Response(200, json={'data': [
                {'id': 'other-model', 'max_model_len': 4096},
                {'id': 'chosen-model', 'max_model_len': 65536}]})
        limit = cli.discover_context_window('https://example.test/v1', 'chosen-model',
            'discovery-secret', transport=httpx.MockTransport(respond))
        self.assertEqual(limit, 65536)

    def test_missing_model_metadata_is_optional(self):
        for status, payload in [(404, {}), (200, {'data': [{'id': 'different', 'max_model_len': 4096}]}),
                                (200, {'data': [{'id': 'model', 'max_model_len': True}]}),
                                (200, ['not-a-model-list'])]:
            with self.subTest(payload=payload):
                transport = httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
                self.assertIsNone(cli.discover_context_window('http://server/v1', 'model', transport=transport))

    def test_context_discovery_is_not_repeated_for_saved_or_explicit_limit(self):
        with patch.object(cli, 'discover_context_window') as discover:
            settings, _ = cli.connection_settings(arguments(model='model', context_window=4096), {})
            self.assertEqual(settings['max_output_tokens'], 1024)
            cli.connection_settings(arguments(), dict(model='model', context_window=8192))
            discover.assert_not_called()

    def test_default_context_is_explained_when_metadata_is_unavailable(self):
        with patch.object(cli, 'discover_context_window', return_value=None), patch('sys.stderr') as stderr:
            settings, _ = cli.connection_settings(arguments(model='model'), {})
        self.assertEqual(settings['context_window'], 131072)
        self.assertIn('metadata unavailable', ''.join(call.args[0] for call in stderr.write.call_args_list))

    def test_url_credentials_are_rejected_before_session_persistence(self):
        for url in ['file:///tmp/model', 'http://user:password@host/v1',
                    'https://host/v1?api_key=secret', 'https://host/v1#secret']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                cli.api_url(url)
        self.assertEqual(cli.api_url('https://host/prefix/v1/'), 'https://host/prefix/v1')
        self.assertEqual(cli.api_url('http://localhost:8000/'), 'http://localhost:8000/v1')

    def test_output_budget_must_leave_room_for_input(self):
        for output in [4096, 8192]:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'less than'):
                cli.connection_settings(arguments(model='model', context_window=4096,
                                                   max_output_tokens=output), {})
        for number in ['0', '-1', 'not-a-number']:
            with self.assertRaises(argparse.ArgumentTypeError):
                cli.positive_integer(number)


class NativeConfigurationTest(unittest.TestCase):
    def test_native_config_keeps_goals_agents_compact_and_local_auth(self):
        config = cli.runtime_config('http://127.0.0.1:1234/v1', Path('/catalog'),
                                    Path('/workspace'), 65536, 16384)
        self.assertEqual(config['model_provider'], 'r_codex_local')
        self.assertTrue(config['features']['goals'])
        self.assertTrue(config['features']['multi_agent'])
        self.assertNotIn('model_reasoning_effort', config)
        provider = config['model_providers']['r_codex_local']
        self.assertEqual(provider['name'], 'OpenAI')
        self.assertEqual(provider['env_key'], 'R_CODEX_SESSION_TOKEN')
        self.assertEqual(provider['wire_api'], 'responses')
        self.assertLessEqual(config['model_auto_compact_token_limit'] + 16384, 65536)

    def test_catalog_advertises_text_only_and_uses_native_prompt(self):
        native = {'models': [{'slug': 'gpt-5.4', 'base_instructions': 'native-instructions'}]}
        with TemporaryDirectory() as td, patch.object(cli.subprocess, 'run',
                return_value=SimpleNamespace(stdout=json.dumps(native))):
            path = cli.build_catalog('/codex', Path(td), {}, 65536, 'backend-model')
            catalog = json.loads(path.read_text())['models'][0]
        self.assertEqual(catalog['slug'], 'text-model')
        self.assertEqual(catalog['input_modalities'], ['text'])
        self.assertEqual(catalog['base_instructions'], 'native-instructions')
        self.assertEqual(catalog['supported_reasoning_levels'], [])

    def test_catalog_enables_images_only_when_requested(self):
        native = {'models': [{'slug': 'gpt-5.4', 'base_instructions': 'native-instructions'}]}
        with TemporaryDirectory() as td, patch.object(cli.subprocess, 'run',
                return_value=SimpleNamespace(stdout=json.dumps(native))):
            path = cli.build_catalog('/codex', Path(td), {}, 65536, 'vision-model', image_input=True)
            catalog = json.loads(path.read_text())['models'][0]
        self.assertEqual(catalog['input_modalities'], ['text', 'image'])

    def test_client_environment_isolated_from_login_but_keeps_tool_credentials(self):
        with patch.dict(os.environ, {'CODEX_HOME': '/user/home', 'OPENAI_API_KEY': 'upstream',
                                    'SERPER_API_KEY': 'search', 'HTTPS_PROXY': 'http://proxy',
                                    'NO_PROXY': 'existing'}, clear=True):
            env = cli.environment(Path('/session'), 'local-token')
        self.assertEqual(env['CODEX_HOME'], '/session/.codex')
        self.assertEqual(env['R_CODEX_SESSION_TOKEN'], 'local-token')
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertEqual(env['SERPER_API_KEY'], 'search')
        self.assertEqual(env['HTTPS_PROXY'], 'http://proxy')
        self.assertIn('127.0.0.1', env['NO_PROXY'])
        self.assertIn('existing', env['NO_PROXY'])

    def test_launcher_uses_native_endpoint_and_persists_resume_settings(self):
        @contextmanager
        def served(provider, token):
            yield 'http://127.0.0.1:43123/v1'

        @contextmanager
        def hosted(*args):
            yield dict(endpoint='unix:///tmp/native.sock', thread_id='native-thread', child_pid=123)

        with TemporaryDirectory() as td:
            root = Path(td) / 'home'
            args = arguments(directory=td, url='http://upstream:8000', model='test-model',
                             context_window=65536, max_output_tokens=4096,
                             reasoning_effort='low', api_key_env='UPSTREAM_KEY', prompt='hello')
            provider = SimpleNamespace(meta={'root_thread_id': 'native-thread'})
            native = {'models': [{'slug': 'gpt-5.4', 'base_instructions': 'native'}]}
            with patch.dict(os.environ, {'R_CODEX_HOME': str(root), 'UPSTREAM_KEY': 'never-store-me'}, clear=True), \
                    patch.object(cli, 'Provider', return_value=provider) as make_provider, \
                    patch.object(cli, 'ensure_codex', return_value=Path('/native/codex')), \
                    patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(native))), \
                    patch.object(cli, 'serve', served), patch('r_codex.host.host', hosted), \
                    patch.object(cli, 'run_terminal', return_value=0) as terminal:
                self.assertEqual(cli.run(args), 0)
            directory = root / 'sessions/sample'
            manifest = json.loads((directory / 'session.json').read_text())
            config = (directory / '.codex/config.toml').read_text()
            self.assertEqual(manifest['url'], 'http://upstream:8000/v1')
            self.assertEqual(manifest['model'], 'test-model')
            self.assertEqual(manifest['api_key_env'], 'UPSTREAM_KEY')
            self.assertEqual(manifest['context_window'], 65536)
            self.assertEqual(manifest['max_output_tokens'], 4096)
            self.assertEqual(manifest['reasoning_effort'], 'low')
            self.assertEqual(manifest['status'], 'closed')
            self.assertEqual(make_provider.call_args.kwargs['api_key'], 'never-store-me')
            self.assertNotIn('never-store-me', config + json.dumps(manifest))
            self.assertNotIn('remory', config.lower())
            command = terminal.call_args.args[0]
            self.assertEqual(command[:4], ['/native/codex', 'resume', '--remote', 'unix:///tmp/native.sock'])
            self.assertEqual(command[-1], 'hello')


if __name__ == '__main__':
    unittest.main()
