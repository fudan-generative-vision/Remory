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
import threading


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


def run_terminal(argv, workspace, env, *, session):
    """Relay output only; stdin and foreground signals still belong to Codex.

    An output PTY preserves native terminal detection and image rendering.
    Keyboard input is never piped, interpreted or re-encoded by this wrapper.
    """
    output = sys.stdout.fileno()
    source = sys.stdin.fileno()
    master, slave = os.openpty()
    hint = ResumeHint(session)
    child = relay = None
    stopped = threading.Event()
    previous = {}
    saved_tty = termios.tcgetattr(source) if os.isatty(source) else None

    def resize(*_):
        if os.isatty(output):
            size = fcntl.ioctl(output, termios.TIOCGWINSZ, b'\0'*8)
            fcntl.ioctl(master, termios.TIOCSWINSZ, size)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGWINCH)

    def write(data):
        while data:
            data = data[os.write(output, data):]

    def copy_output():
        try:
            while True:
                ready, _, _ = select.select([master], [], [], .1)
                if not ready:
                    if stopped.is_set():
                        break
                    continue
                try:
                    data = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not data:
                    break
                write(hint.feed(data))
            write(hint.finish())
        except OSError:
            # The real terminal disappeared; preserve the normal launcher
            # cleanup path instead of leaving an interactive child orphaned.
            if child is not None and child.poll() is None:
                child.terminate()

    def terminate(signum, _):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise KeyboardInterrupt

    try:
        # Avoid translating LF twice when relaying between terminal devices.
        attributes = termios.tcgetattr(slave)
        attributes[1] &= ~termios.OPOST
        termios.tcsetattr(slave, termios.TCSANOW, attributes)
        resize()
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, lambda *_: None)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, terminate)
        previous[signal.SIGWINCH] = signal.signal(signal.SIGWINCH, resize)
        child = subprocess.Popen(argv, cwd=workspace, env=env, stdout=slave, stderr=slave)
        os.close(slave)
        slave = None
        relay = threading.Thread(target=copy_output, daemon=True)
        relay.start()
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        stopped.set()
        if relay:
            relay.join(timeout=2)
        if saved_tty is not None:
            with suppress(OSError, termios.error):
                termios.tcsetattr(source, termios.TCSADRAIN, saved_tty)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        os.close(master)
        if slave is not None:
            os.close(slave)
