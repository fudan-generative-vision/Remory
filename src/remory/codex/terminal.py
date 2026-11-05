"""Native-terminal relay for r-codex's session command alias.

Pinned Codex cannot register new bare slash commands. Translate a command typed
at a fresh composer boundary to a provider-handled session message. All other
input, terminal output, signals and resize events remain native Codex's.
"""
import errno
import fcntl
import os
import pty
import select
import signal
import subprocess
import sys
import termios
import tty


class CommandAliases:
    alias = b'/agentic_compact'
    replacement = b'[Remory session] /agentic_compact'

    def __init__(self):
        self.pending = b''
        self.fresh = True
        self.escape = b''
        self.pasted = False
        self.command = None

    def expansion(self, text=None):
        # An injected command arrives much faster than human keystrokes. Mark
        # it as a paste so native paste-burst detection cannot eat the Enter.
        text = self.replacement if text is None else text
        if self.pasted:
            return text
        return b'\x1b[200~' + text + b'\x1b[201~'

    def feed(self, data):
        out = bytearray()
        for byte in data:
            char = bytes([byte])
            if self.command is not None:
                if byte in (8, 127):
                    self.command = self.command[:-1]
                elif byte in (3, 21):
                    self.command = None
                    self.fresh = True
                    out.extend(char)
                elif char in b'\r\n\t' or byte == 27:
                    text = self.command
                    if text == self.alias or text.startswith(self.alias + b' '):
                        text = self.replacement + text[len(self.alias):]
                    out.extend(self.expansion(text))
                    self.command = None
                    self.fresh = char in b'\r\n'
                    if byte == 27:
                        self.escape = char
                    else:
                        out.extend(char)
                else:
                    self.command += char
                continue
            if self.escape:
                self.escape += char
                if self.escape in (b'\x1b[200~', b'\x1b[201~'):
                    self.pasted = self.escape == b'\x1b[200~'
                    if not self.pasted:
                        # Paste terminators are a word boundary, not a submit.
                        out.extend(self.replacement if self.pending == self.alias else self.pending)
                        if self.pending:
                            self.fresh = False
                        self.pending = b''
                    out.extend(self.escape)
                    self.escape = b''
                elif self.escape == b'\x1b[':
                    continue
                elif self.escape.startswith(b'\x1b[') and len(self.escape) < 64:
                    if not 0x40 <= byte <= 0x7e:
                        continue
                    sequence, self.escape = self.escape, b''
                    # Startup device/cursor/keyboard capability replies are not
                    # composer edits. They must not disable the first command.
                    if byte in (ord('R'), ord('c')) or sequence.startswith((b'\x1b[?', b'\x1b[>')):
                        out.extend(sequence)
                    elif sequence in (b'\x1b[13u', b'\x1b[13;1u', b'\x1b[9u', b'\x1b[9;1u'):
                        key = b'\r' if sequence.startswith(b'\x1b[13') else b'\t'
                        out.extend(self.feed(key)[:-1] + sequence)
                    else:
                        out.extend(self.pending + sequence)
                        self.pending = b''
                        self.fresh = False
                else:
                    out.extend(self.pending + self.escape)
                    self.pending = self.escape = b''
                    # Cursor motion/history makes the composer position unknown.
                    self.fresh = False
                continue
            if byte == 27:
                self.escape = char
                continue
            if self.pending:
                if self.pending == self.alias and char in b' \r\n\t':
                    if char == b' ' and not self.pasted:
                        self.command = self.alias + char
                    else:
                        out.extend(self.expansion() + char)
                    self.pending = b''
                    self.fresh = char in b'\r\n' and not self.pasted
                elif byte in (8, 127):
                    self.pending = self.pending[:-1]
                elif byte in (3, 21):
                    self.pending = b''
                    self.fresh = True
                    out.extend(char)
                elif self.alias.startswith(self.pending + char):
                    self.pending += char
                else:
                    out.extend(self.pending + char)
                    self.pending = b''
                    self.fresh = char in b'\r\n' and not self.pasted
                continue
            if self.fresh and char == b'/':
                self.pending = char
                continue
            out.extend(char)
            if byte in (3, 21) or (char in b'\r\n' and not self.pasted):
                self.fresh = True
            elif char not in b'\r\n':
                self.fresh = False
        return bytes(out)

    def flush_escape(self):
        if not self.escape:
            return b''
        output = self.pending + self.escape
        self.pending = self.escape = b''
        self.fresh = False
        return output


def run_terminal(command, workspace, env):
    """Use a PTY so the alias works without replacing Codex's UI or bindings."""
    master, slave = pty.openpty()
    original = termios.tcgetattr(sys.stdin.fileno())
    previous, child = {}, None

    def resize(*_):
        size = fcntl.ioctl(sys.stdin.fileno(), termios.TIOCGWINSZ, bytes(8))
        fcntl.ioctl(master, termios.TIOCSWINSZ, size)

    def terminate(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise KeyboardInterrupt

    try:
        resize()
        termios.tcsetattr(slave, termios.TCSANOW, original)
        # Establish a controlling terminal after exec of this tiny helper. Avoid
        # preexec_fn: the HTTP provider already has threads when we get here.
        helper = ('import fcntl,os,sys,termios; '
                  'fcntl.ioctl(0,termios.TIOCSCTTY,0); '
                  'os.execvpe(sys.argv[1],sys.argv[1:],os.environ)')
        child = subprocess.Popen([sys.executable, '-c', helper, *command], cwd=workspace,
            env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        slave = None
        previous[signal.SIGWINCH] = signal.signal(signal.SIGWINCH, resize)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, terminate)
        tty.setraw(sys.stdin.fileno())
        aliases = CommandAliases()
        while True:
            ready, _, _ = select.select([sys.stdin.fileno(), master], [], [], .25)
            if not ready:
                pending = aliases.flush_escape()
                if pending:
                    os.write(master, pending)
            if master in ready:
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    break
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
            if sys.stdin.fileno() in ready:
                data = os.read(sys.stdin.fileno(), 65536)
                if not data:
                    child.send_signal(signal.SIGHUP)
                    break
                mapped = aliases.feed(data)
                if mapped:
                    os.write(master, mapped)
            if child.poll() is not None and master not in ready:
                break
        return child.wait()
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        os.close(master)
        if slave is not None:
            os.close(slave)
