"""Keep the native terminal interaction and show the owning r-codex resume command."""
import errno
from contextlib import suppress
import fcntl
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import termios
import time
import tty

from .terminal_commands import CommandScreen, InputPackets


class ResumeHint:
    prefix = b'To continue this session, run '
    ansi = re.compile(rb'\x1b\[[0-?]*[ -/]*[@-~]')

    def __init__(self, session):
        self.command = ('r-codex resume ' + shlex.quote(session)).encode()
        self.buffer = b''
        self.replaced = False

    def line(self, value):
        plain = self.ansi.sub(b'', value).rstrip(b'\r\n')
        if re.fullmatch(re.escape(self.prefix) + rb'codex resume(?:[ ,].*)?', plain):
            self.replaced = True
            ending = b'\r\n' if value.endswith(b'\r\n') else b'\n' if value.endswith(b'\n') else b''
            return self.prefix + self.command + b'\x1b[0m' + ending
        return value

    def feed(self, data):
        # Do not buffer terminal frames or parse input. Only the known exit
        # sentence can be held, including when writes split it across chunks.
        self.buffer += data
        output = bytearray()
        while self.buffer:
            start = self.buffer.find(self.prefix)
            if start < 0:
                keep = next((n for n in range(min(len(self.buffer), len(self.prefix)-1), 0, -1)
                             if self.buffer.endswith(self.prefix[:n])), 0)
                if keep:
                    output.extend(self.buffer[:-keep])
                    self.buffer = self.buffer[-keep:]
                else:
                    output.extend(self.buffer)
                    self.buffer = b''
                break
            output.extend(self.buffer[:start])
            self.buffer = self.buffer[start:]
            end = self.buffer.find(b'\n')
            if end < 0:
                if len(self.buffer) > 4096:
                    output.extend(self.buffer)
                    self.buffer = b''
                break
            output.extend(self.line(self.buffer[:end+1]))
            self.buffer = self.buffer[end+1:]
        return bytes(output)

    def finish(self):
        value, self.buffer = self.buffer, b''
        return self.line(value)


def run_terminal(argv, workspace, env, *, session, local_commands=False):
    """Keep native rendering and dispatch local /remory commands through its RPC."""
    output = sys.stdout.fileno()
    source = sys.stdin.fileno()
    master, slave = os.openpty()
    hint = ResumeHint(session)
    child = None
    packets = InputPackets()
    size = os.get_terminal_size(output)
    display = CommandScreen(size.columns, size.lines) if local_commands else None
    previous = {}
    saved_tty = termios.tcgetattr(source) if os.isatty(source) else None

    def resize(*_):
        if os.isatty(output):
            size = fcntl.ioctl(output, termios.TIOCGWINSZ, b'\0'*8)
            fcntl.ioctl(master, termios.TIOCSWINSZ, size)
            dimensions = os.get_terminal_size(output)
            if display is not None:
                display.resize(dimensions.columns, dimensions.lines)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGWINCH)

    def write(data):
        while data:
            data = data[os.write(output, data):]

    def copy_output():
        try:
            data = os.read(master, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                return False
            raise
        if not data:
            return False
        if display is not None:
            display.feed(data)
        write(hint.feed(data))
        return True

    def send(data):
        while data:
            data = data[os.write(master, data):]

    def refresh_composer():
        # A paste and Enter can arrive in one read. Let the native renderer
        # consume that text before examining its composer, including paste bursts.
        until = time.monotonic() + .12
        while time.monotonic() < until:
            ready, _, _ = select.select([master], [], [], max(0, until-time.monotonic()))
            if ready and not copy_output():
                break

    def terminate(signum, _):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise KeyboardInterrupt

    try:
        # The outer terminal is raw while relaying, so keep the slave's normal
        # newline translation for output after Codex restores its terminal mode.
        resize()
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, lambda *_: None)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, terminate)
        previous[signal.SIGWINCH] = signal.signal(signal.SIGWINCH, resize)
        tty.setraw(source)
        child = subprocess.Popen(argv, cwd=workspace, env=env,
                                 stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        slave = None
        while True:
            ready, _, _ = select.select([master, source], [], [], .05)
            if master in ready and not copy_output():
                break
            if source in ready:
                data = os.read(source, 65536)
                if not data:
                    break
                if display is None:
                    send(data)
                    continue
                for submit, packet in packets.feed(data):
                    if submit and display is not None:
                        refresh_composer()
                        packet = display.submission(packet)
                    send(packet)
            elif packets.pending:
                send(packets.flush_escape())
            if child.poll() is not None and not ready:
                break
        write(hint.finish())
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if saved_tty is not None:
            with suppress(OSError, termios.error):
                termios.tcsetattr(source, termios.TCSADRAIN, saved_tty)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        os.close(master)
        if slave is not None:
            os.close(slave)
