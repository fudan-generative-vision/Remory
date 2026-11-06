"""Token-exact Qwen history cache with immutable committed round boundaries.

Only new assistant/observation pairs are tokenized. Speculative tool admission
may replace an uncommitted tail; neither it nor a shorter summary view can
rewrite committed history. No archived text is re-expanded in a model request.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


POLICY = "append_only_token_exact_round_cache_v1"


def _digest(message):
    return hashlib.sha256(json.dumps(message, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).digest()


class IncrementalHistory:
    def __init__(self, canonicalize, chat):
        self.canonicalize, self.chat = canonicalize, chat
        self.episode = None
        self.digests, self.committed = [], []
        self.canonical, self.ids, self.ends = [], [], []
        self.prefix, self.opening = [], []
        self.tokenized_pairs = 0

    def render(self, messages, episode):
        fields = ("episode", "digests", "canonical", "ids", "ends", "prefix", "opening", "tokenized_pairs")
        previous = {name: getattr(self, name) for name in fields}
        try:
            return self._render(messages, episode)
        except BaseException:
            for name, value in previous.items():
                setattr(self, name, value)
            raise

    def _render(self, messages, episode):
        if len(messages) < 2 or len(messages) % 2:
            raise ValueError("history requires system/user and complete interaction rounds")
        if self.episode not in (None, episode):
            raise RuntimeError("history episode changed")
        digests = [_digest(m) for m in messages]
        overlap = min(len(digests), len(self.committed))
        if digests[:overlap] != self.committed[:overlap]:
            raise RuntimeError("committed history changed")
        common = 0
        while common < min(len(digests), len(self.digests)) and digests[common] == self.digests[common]:
            common += 1
        common -= common % 2
        if common < 2:
            if self.digests and digests[:2] != self.digests[:2]:
                raise RuntimeError("persistent system/user history changed")
            canonical = self.canonicalize(messages[:2], episode)
            prefix = self.chat(canonical)
            opened = self.chat(canonical, opening=True)
            if opened[:len(prefix)] != prefix:
                raise RuntimeError("generation opening changed the persistent prefix")
            self.prefix, self.opening = list(prefix), opened[len(prefix):]
            self.canonical, self.ids = deepcopy(canonical), list(prefix)
            self.ends = [len(prefix), len(prefix)]
            common = 2
        # A shorter summary view must not discard a useful, checked longer cache.
        if common == len(messages) and len(self.digests) >= len(messages):
            end = self.ends[len(messages) - 1]
            return deepcopy(self.canonical[:len(messages)]), list(self.prefix), self.ids[:end]
        self.ids = self.ids[:self.ends[common - 1]]
        self.canonical, self.ends = self.canonical[:common], self.ends[:common]
        for index in range(common, len(messages), 2):
            pair = self.canonicalize([*messages[:2], *messages[index:index + 2]], episode)[2:]
            if [m["role"] for m in pair] not in (["assistant", "tool"], ["assistant", "user"]):
                raise ValueError("history requires assistant/(tool or user) pairs")
            initial = self.canonical[:2]
            assistant = self.chat(initial + pair[:1])
            rendered = self.chat(initial + pair)
            opened = self.chat(initial + pair, opening=True)
            if (assistant[:len(self.prefix)] != self.prefix
                    or rendered[:len(assistant)] != assistant
                    or opened != rendered + self.opening):
                raise RuntimeError("Qwen pair is not append-only token exact")
            # Bootstrap the append contract against the full-history oracle.
            if len(self.canonical) == 4 and self.chat(self.canonical + pair) != self.ids + rendered[len(self.prefix):]:
                raise RuntimeError("Qwen full-history template is not pair-append compatible")
            base = len(self.ids) - len(self.prefix)
            self.ends.extend([base + len(assistant), base + len(rendered)])
            self.ids.extend(rendered[len(self.prefix):])
            self.canonical.extend(deepcopy(pair))
            self.tokenized_pairs += 1
        self.digests, self.episode = digests, episode
        return deepcopy(self.canonical), list(self.prefix), list(self.ids)

    def commit(self, messages, episode):
        self.render(messages, episode)
        digests = [_digest(m) for m in messages]
        if len(digests) >= len(self.committed):
            self.committed = digests
