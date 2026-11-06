"""Session commands intercepted before native model input, with optional training."""
import re

from ..online.control import command as online_command


def command(params):
    items = params.get('input')
    if (isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict)
            and items[0].get('type') == 'text'):
        text = items[0].get('text', '').strip()
        if re.match(r'^/agentic_compact(?:\s|$)', text):
            return ['agentic_compact', *text.split()[1:]]
    return online_command(params)


class SessionControl:
    def __init__(self, provider, thread_id, online=None):
        self.provider, self.thread_id, self.online = provider, thread_id, online

    def check_thread(self, thread_id):
        if thread_id != self.thread_id:
            raise ValueError('Session commands belong to the interactive root thread')

    def training(self):
        if self.online is None:
            raise ValueError('Online training is not enabled for this session; use --online-training when starting it')
        return self.online

    def execute(self, args):
        if args and args[0] == 'agentic_compact':
            return self.provider.agentic_compact_command(' '.join(args[1:]))
        return self.training().execute(args)

    def candidates(self):
        return self.training().candidates()

    def select(self, ident):
        return self.training().select(ident)
