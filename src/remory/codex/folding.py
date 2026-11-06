"""Copy-on-write message folding with a small, declarative edit language.

The organizer never receives a reference to the actor's mutable history. It may
replace assistant spans and complete tool exchanges, but cannot edit user or
system messages, split tool exchanges, or execute transformation code.
"""
from copy import deepcopy
import json

from ..types import digest


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def message_units(messages):
    """Find indivisible assistant-call/result groups; reject orphaned tool data."""
    units = []
    index = 0
    while index < len(messages):
        start = index
        message = messages[index]
        role = message.get('role')
        if role not in {'system', 'developer', 'user', 'assistant', 'tool'}:
            raise ValueError('Unknown message role in folding snapshot')
        if role == 'tool':
            raise ValueError('Orphaned tool result in folding snapshot')
        calls = message.get('tool_calls', [])
        if calls:
            if role != 'assistant':
                raise ValueError('Only assistants may call tools')
            ids = [call.get('id') for call in calls]
            if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
                raise ValueError('Invalid or duplicate tool call IDs')
            pending = set(ids)
            while pending:
                index += 1
                if index >= len(messages):
                    raise ValueError('Unfinished tool exchange cannot be folded')
                result = messages[index]
                cid = result.get('tool_call_id')
                if result.get('role') != 'tool' or cid not in pending:
                    raise ValueError('Tool results must follow their calls exactly once')
                pending.remove(cid)
        units.append((start, index + 1, role in {'system', 'developer', 'user'}))
        index += 1
    return units


class FoldState:
    """One ephemeral organizer transaction. All inspection results are copies."""

    def __init__(self, snapshot, *, summary_validator=None):
        self._snapshot = deepcopy(snapshot)
        self._messages = self._snapshot['messages']
        self._units = message_units(self._messages)
        self._boundaries = self._snapshot.get('protected_boundaries', [])
        edges = {0, len(self._messages)} | {x for lo, hi, _ in self._units for x in (lo, hi)}
        if not isinstance(self._boundaries, list) or any(type(x) is not int or x not in edges
                                                       for x in self._boundaries):
            raise ValueError('Protected boundaries must be complete message-unit edges')
        self._source_hash = digest(self._snapshot)
        self._candidates = {}
        self._result = None
        self._aborted = None
        self._summary_validator = summary_validator

    @property
    def result(self):
        return deepcopy(self._result)

    def _open(self):
        if self._aborted is not None:
            raise ValueError('Folding was aborted: ' + self._aborted)
        if self._result is not None:
            raise ValueError('Folding was already committed')

    def inspect_context(self, target='index', offset=0, limit=8000):
        self._open()
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
            raise ValueError('Use a nonnegative offset and a limit between 1 and 12000')
        if target == 'index':
            value = [dict(start=start, end=end, protected=protected,
                          characters=len(_json(self._messages[start:end])))
                     for start, end, protected in self._units]
        elif target == 'messages':
            # Explicit indices avoid making the organizer count hundreds of
            # nested message objects to map evidence back to an edit range.
            value = [dict(message_index=index, message=message)
                     for index, message in enumerate(self._messages)]
        elif target == 'context':
            value = {k: v for k, v in self._snapshot.items() if k != 'messages'}
        else:
            raise ValueError('Unknown inspection target')
        text = _json(value)
        result = dict(target=target, offset=offset, total_characters=len(text),
                    text=text[offset:offset + limit],
                    next_offset=offset + limit if offset + limit < len(text) else None,
                    snapshot_sha256=self._source_hash, pagination_unit='characters')
        if target == 'index' and offset == 0:
            spans = []
            for start, end, protected in self._units:
                if protected:
                    continue
                if spans and spans[-1]['end'] == start and start not in self._boundaries:
                    spans[-1]['end'] = end
                else:
                    spans.append(dict(start=start, end=end))
            result['editable_spans'] = spans
        return result

    def test_transform(self, edits, summary):
        self._open()
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 32768:
            raise ValueError('Supply a nonempty checkpoint summary of at most 32768 characters')
        schema = self._snapshot.get('summary_contract', {}).get('output_schema')
        if schema:
            from jsonschema import ValidationError, validate
            try:
                validate(json.loads(summary), schema)
            except (ValueError, ValidationError) as exc:
                raise ValueError('Summary must match the existing output_schema: ' + str(exc)) from exc
        if self._summary_validator:
            self._summary_validator(summary)
        if not isinstance(edits, list) or len(edits) > 256:
            raise ValueError('edits must be a list with at most 256 replacements')
        if len(_json(edits)) > 262144:
            raise ValueError('Edit plan is too large')
        boundaries = {0, len(self._messages)}
        for start, end, _ in self._units:
            boundaries.update((start, end))
        folded, cursor = [], 0
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {'start', 'end', 'text'}:
                raise ValueError('Each edit must contain exactly start, end, text')
            start, end, text = edit['start'], edit['end'], edit['text']
            if type(start) is not int or type(end) is not int or not cursor <= start < end <= len(self._messages):
                raise ValueError('Edit ranges must be ordered, nonoverlapping, and in bounds')
            if start not in boundaries or end not in boundaries:
                raise ValueError('An edit cannot split a tool exchange')
            if any(start < boundary < end for boundary in self._boundaries):
                raise ValueError('An edit cannot cross an existing residual-memory boundary')
            if any(protected and start <= lo < end for lo, _, protected in self._units):
                raise ValueError('User, system and developer messages are protected')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Each removed span requires a nonempty factual replacement')
            folded.extend(deepcopy(self._messages[cursor:start]))
            folded.append({'role': 'assistant', 'content': '[context-summary]\n' + text.strip()})
            cursor = end
        folded.extend(deepcopy(self._messages[cursor:]))
        message_units(folded)
        before, after = len(_json(self._messages)), len(_json(folded))
        if edits and after > before:
            raise ValueError('The folded messages must not be larger than the original copy')
        candidate_id = digest(dict(snapshot=self._source_hash, edits=edits, summary=summary))
        candidate = dict(messages=folded, summary=summary.strip(), audit=dict(
            snapshot_sha256=self._source_hash, candidate_id=candidate_id,
            edits=deepcopy(edits), original_characters=before, folded_characters=after,
            assessment='AI-generated folding; semantic fidelity is not mechanically guaranteed'))
        # Keep a bounded number of trials, never a growing organizer transcript.
        if len(self._candidates) >= 8:
            self._candidates.pop(next(iter(self._candidates)))
        self._candidates[candidate_id] = candidate
        return dict(candidate_id=candidate_id, valid=True, original_characters=before,
                    folded_characters=after, message_count=len(folded))

    def commit_transform(self, candidate_id):
        self._open()
        if candidate_id not in self._candidates:
            raise ValueError('Test the candidate before committing it')
        self._result = self._candidates[candidate_id]
        self._candidates.clear()
        return dict(committed=True, candidate_id=candidate_id)

    def abort_folding(self, reason):
        self._open()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('An abort requires a reason')
        self._aborted = reason[:2000]
        self._candidates.clear()
        return dict(aborted=True, reason=self._aborted)
