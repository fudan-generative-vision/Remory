"""Packet boundaries for terminal input; no custom slash-command rewriting."""
import re


class InputPackets:
    """Keep pasted newlines and terminal escape sequences intact."""
    csi = re.compile(rb'\x1b\[[0-?]*[ -/]*[@-~]')

    def __init__(self):
        self.pending = b''
        self.paste = False

    def feed(self, data):
        value = self.pending + data
        self.pending = b''
        result, plain = [], bytearray()
        def flush():
            if plain:
                result.append((False, bytes(plain)))
                plain.clear()
        while value:
            if value.startswith(b'\x1b'):
                if value == b'\x1b' or value.startswith(b'\x1b['):
                    match = self.csi.match(value)
                    if not match:
                        self.pending = value
                        break
                    token = match[0]
                else:
                    token = value[:2]
                value = value[len(token):]
                if token == b'\x1b[200~':
                    self.paste = True
                elif token == b'\x1b[201~':
                    self.paste = False
                elif token in (b'\x1b[13u', b'\x1b[13;1u', b'\x1b[9u') and not self.paste:
                    flush()
                    result.append((True, token))
                    continue
                plain.extend(token)
            else:
                token, value = value[:1], value[1:]
                if token in (b'\r', b'\t') and not self.paste:
                    flush()
                    result.append((True, token))
                else:
                    plain.extend(token)
        flush()
        return result

    def flush_escape(self):
        value, self.pending = self.pending, b''
        return value
