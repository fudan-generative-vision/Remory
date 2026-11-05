"""Optional CPU smoke through the real TUI, PTY alias and provider."""
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time

import pytest


def test_native_session_command(tmp_path):
    binary = os.environ.get('REMORY_TEST_CODEX')
    if not binary:
        pytest.skip('Set REMORY_TEST_CODEX to the pinned Codex 0.149.0 binary')
    script = tmp_path / 'terminal_test.py'
    script.write_text('''
import sys
from pathlib import Path
import httpx
from remory.codex.cli import build_catalog, environment, serve, toml
from remory.codex.provider import Provider
from remory.codex.terminal import run_terminal
from test_codex import Tokenizer
directory, binary = Path(sys.argv[1]), sys.argv[2]
(directory / '.codex').mkdir()
def backend(request):
    raise AssertionError('Session commands must not invoke the model')
p = Provider('http://unused', {'identity':'fake', 'config':{'target_model':'fake'}},
    Tokenizer(), directory / 'provider', session_id='fake', transport=httpx.MockTransport(backend))
p.folder = object()
p.session_commands = True
env = environment(directory, 'local')
catalog = build_catalog(binary, directory, env, 262144, 'Fake local')
with serve(p, 'local') as url:
    config = dict(model='remory', model_provider='remory_local', model_catalog_json=str(catalog),
        check_for_update_on_startup=False, approval_policy='never', sandbox_mode='read-only',
        projects={str(directory): {'trust_level':'trusted'}},
        model_providers={'remory_local': {'name':'OpenAI','base_url':url,
        'env_key':'REMORY_CODEX_TOKEN','wire_api':'responses','requires_openai_auth':False,
        'request_max_retries':0,'stream_max_retries':0}})
    (directory / '.codex/config.toml').write_text('\\n'.join(f'{k} = {toml(v)}' for k,v in config.items()))
    run_terminal([binary, '-C', str(directory), '--no-alt-screen'], directory, env)
''')
    master, slave = pty.openpty()
    import fcntl
    import struct
    import termios
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 160, 0, 0))
    env = {**os.environ, 'TERM': 'xterm-256color', 'PYTHONPATH': os.pathsep.join([
        str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)])}
    process = subprocess.Popen([sys.executable, str(script), str(tmp_path), binary],
        stdin=slave, stdout=slave, stderr=slave, env=env, start_new_session=True)
    os.close(slave)
    transcript = bytearray()

    def pump(until, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], .1)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b''
                transcript.extend(chunk)
                if b'\x1b[6n' in chunk:
                    os.write(master, b'\x1b[1;1R')
                if b'\x1b[c' in chunk:
                    os.write(master, b'\x1b[?1;2c')
                if b'\x1b[?u' in chunk:
                    os.write(master, b'\x1b[?0u')
            if until():
                return
            if process.poll() is not None:
                break
        import re
        clean = re.sub(rb'\x1b\[[0-9;?]*[A-Za-z]', b'', bytes(transcript))
        pytest.fail((clean[:2500] + b'\n...\n' + clean[-1000:]).decode(errors='replace'))

    def enabled(expected):
        path = tmp_path / 'provider/provider.json'
        return path.exists() and json.loads(path.read_text()).get('agentic_compact') is expected

    try:
        pump(lambda: b'remory default' in transcript)
        os.write(master, b'/agentic_compact\r')
        pump(lambda: enabled(True))
        # Wait for the native final acknowledgement before the next command.
        pump(lambda: b'ON' in transcript)
        os.write(master, b'/agentic_compact off\r')
        pump(lambda: enabled(False))
        pump(lambda: b'OFF' in transcript)
        transcript.clear()
        os.write(master, b'/agentic_compact status\r')
        pump(lambda: b'OFF' in transcript)
        assert enabled(False)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
