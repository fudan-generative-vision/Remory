"""A native Codex Responses adapter with ordinary text-only compaction.

Inference uses an OpenAI-compatible Chat Completions endpoint. Checkpoints
contain a readable summary, never residual tensors or a remote memory handle.
"""
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import httpx

from .protocol import chat_output, content_text, messages_from_wire, sse, tools_from_wire


DEFAULT_INSTRUCTIONS = 'You are a coding assistant. Use the available tools to complete the user task.'
COMPACTION_PREFIX = 'text.v1.'
# Public local compaction instruction from the pinned Codex 0.149.0 binary,
# not OpenAI's remote compactor. No training/tokenizer dependency is required.
COMPACT_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            temporary.chmod(0o600)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Provider:
    def __init__(self, url, model, directory, *, session_id, max_new_tokens=16384,
                 api_key=None, transport=None, reasoning_effort=None):
        self.url = url.rstrip('/')
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError('Chat backend URL must use http:// or https://')
        if not isinstance(model, str) or not model:
            raise ValueError('A backend model name is required')
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        self.model, self.session_id = model, session_id
        self.max_new_tokens = max_new_tokens
        self.api_key, self.transport = api_key, transport
        self.reasoning_effort = reasoning_effort
        self.endpoint = (self.url if self.url.endswith('/v1') else self.url + '/v1')
        self.identity = digest({'url': self.endpoint, 'model': model})
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock, self._locks = threading.RLock(), {}
        self._closed = False
        self.meta_path = self.directory / 'provider.json'
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
            if self.meta.get('schema') != 'r_codex.text.provider.v1':
                raise ValueError('This is not a text provider session; start a new r-codex session')
            if self.meta.get('identity') != self.identity:
                raise ValueError('Backend model or URL changed; start a new r-codex session')
            if self.meta.get('session_id') != session_id:
                raise ValueError('Session identity changed')
        else:
            self.meta = {'schema': 'r_codex.text.provider.v1', 'identity': self.identity,
                         'session_id': session_id, 'root_thread_id': None, 'threads': {}}
            atomic_json(self.meta_path, self.meta)

    @contextmanager
    def client(self):
        headers = {'Authorization': 'Bearer ' + self.api_key} if self.api_key else {}
        with httpx.Client(timeout=1800, headers=headers, transport=self.transport,
                          trust_env=False) as client:
            yield client

    def _thread(self, headers):
        metadata = json.loads(headers.get('x-codex-turn-metadata', '{}'))
        if not isinstance(metadata, dict):
            raise ValueError('Invalid Codex turn metadata')
        thread = headers.get('thread-id') or metadata.get('thread_id') or 'root'
        if not isinstance(thread, str) or not 0 < len(thread) <= 128:
            raise ValueError('Invalid Codex thread ID')
        if metadata.get('thread_id', thread) != thread:
            raise ValueError('Conflicting Codex thread IDs')
        with self._lock:
            if self._closed:
                raise RuntimeError('Provider is closing')
            if self.meta['root_thread_id'] is None:
                self.meta['root_thread_id'] = thread
            if thread not in self.meta['threads']:
                # Native Codex owns the agent tree. Plain summaries may be
                # inherited by children without transferring server-side state.
                self.meta['threads'][thread] = {'parent': metadata.get('parent_thread_id')}
                atomic_json(self.meta_path, self.meta)
            return thread, self._locks.setdefault(thread, threading.RLock())

    def _path(self, thread):
        return self.directory / ('thread-' + hashlib.sha256(thread.encode()).hexdigest() + '.json')

    def _load(self, thread):
        path = self._path(thread)
        return json.loads(path.read_text()) if path.exists() else {'responses': {}}

    def _decode_checkpoint(self, item):
        marker = item.get('encrypted_content')
        if not isinstance(marker, str) or not marker.startswith(COMPACTION_PREFIX):
            raise ValueError('Foreign compaction checkpoint; only text.v1 checkpoints are supported')
        try:
            data = json.loads(base64.b64decode(marker[len(COMPACTION_PREFIX):],
                                             altchars=b'-_', validate=True))
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Malformed text compaction checkpoint') from exc
        if (not isinstance(data, dict) or data.get('schema') != 'r_codex.text.checkpoint.v1'
                or data.get('identity') != self.identity
                or not isinstance(data.get('summary'), str) or not data['summary'].strip()
                or not isinstance(data.get('leading'), list)):
            raise ValueError('Invalid text compaction checkpoint or backend identity changed')
        for message in data['leading']:
            if (not isinstance(message, dict) or message.get('role') not in {'system', 'developer'}
                    or not isinstance(message.get('content'), str)):
                raise ValueError('Invalid checkpoint instructions')
        return data

    def _context(self, request):
        if request.get('previous_response_id'):
            raise ValueError('previous_response_id is unsupported; send the full Responses input')
        items = request.get('input', [])
        if isinstance(items, str):
            items = [{'role': 'user', 'content': items}]
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError('Responses input must be a string or a list of items')
        triggers = [i for i, item in enumerate(items) if item.get('type') == 'compaction_trigger']
        if triggers and (triggers != [len(items) - 1] or set(items[-1]) != {'type'}):
            raise ValueError('compaction_trigger must be the last item without a payload')
        items = items[:-1] if triggers else items
        tools, routes = tools_from_wire(request.get('tools', []))
        instructions = request.get('instructions') or DEFAULT_INSTRUCTIONS
        if not isinstance(instructions, str):
            raise ValueError('Responses instructions must be text')
        leading = []
        while items and items[0].get('role') in {'system', 'developer'}:
            leading.append({'role': items[0]['role'], 'content': content_text(items[0].get('content'))})
            items = items[1:]
        indices = [i for i, item in enumerate(items) if item.get('type') == 'compaction']
        summary = None
        if indices:
            previous = self._decode_checkpoint(items[indices[-1]])
            leading = leading or previous['leading']
            summary = previous['summary']
            items = items[indices[-1] + 1:]
        prefix = [{'role': 'system', 'content': '\n\n'.join(
            [instructions, *(message['content'] for message in leading)])}]
        messages = messages_from_wire(items)
        if summary:
            messages.insert(0, {'role': 'user', 'content':
                '[Context checkpoint from the earlier conversation]\n' + summary})
        return prefix, messages, leading, tools, routes, bool(triggers)

    def _generate(self, messages, tools, request, *, compact=False):
        body = {'model': self.model, 'messages': messages, 'stream': False,
                'max_tokens': self.max_new_tokens}
        if tools and not compact:
            body['tools'] = tools
            if 'parallel_tool_calls' in request:
                body['parallel_tool_calls'] = bool(request['parallel_tool_calls'])
            choice = request.get('tool_choice')
            if isinstance(choice, str):
                body['tool_choice'] = choice
            elif choice is not None:
                raise ValueError('Only auto, none, or required tool_choice values are supported')
        reasoning = request.get('reasoning') or {}
        if not isinstance(reasoning, dict):
            raise ValueError('Responses reasoning must be an object')
        # Codex may attach a catalog default even for non-reasoning backends.
        # Only opt into this optional backend extension when configured by the user.
        effort = (reasoning.get('effort', self.reasoning_effort)
                  if self.reasoning_effort is not None else None)
        if effort:
            body['reasoning_effort'] = effort
        if compact:
            body['temperature'] = 0
        with self.client() as client:
            response = client.post(self.endpoint + '/chat/completions', json=body)
            response.raise_for_status()
            result = response.json()
        if not isinstance(result, dict) or not result.get('choices'):
            raise RuntimeError('Chat backend returned no completion')
        choice = result['choices'][0]
        reason = choice.get('finish_reason')
        if reason not in {'stop', 'tool_calls', 'length'}:
            raise RuntimeError('Chat generation did not complete: ' + str(reason))
        if compact and reason != 'stop':
            raise RuntimeError('Summary did not complete; the original context is retained')
        return choice['message'], result.get('usage', {})

    def handle(self, request, headers=None, *, force_compact=False):
        if not isinstance(request, dict):
            raise ValueError('Responses request must be an object')
        thread, lock = self._thread(headers or {})
        with lock:
            state = self._load(thread)
            request_key = digest({'request': request, 'force_compact': force_compact})
            if request_key in state['responses']:
                return state['responses'][request_key]
            prefix, messages, leading, tools, routes, trigger = self._context(request)
            compact = force_compact or trigger
            if compact:
                if not messages:
                    raise ValueError('There is no history to compact')
                message, usage = self._generate(prefix + messages + [
                    {'role': 'user', 'content': COMPACT_PROMPT}], [], request, compact=True)
                if message.get('tool_calls'):
                    raise RuntimeError('Summary attempted a tool call; the original context is retained')
                summary = content_text(message.get('content')).strip()
                # Servers without a reasoning parser can return inline thinking.
                if '</think>' in summary:
                    summary = summary.split('</think>', 1)[1].strip()
                if not summary or summary.startswith('<think>') or '<tool_call>' in summary:
                    raise RuntimeError('Summary is empty or unfinished; the original context is retained')
                checkpoint = {'schema': 'r_codex.text.checkpoint.v1', 'identity': self.identity,
                              'summary': summary, 'leading': leading}
                marker = COMPACTION_PREFIX + base64.urlsafe_b64encode(json.dumps(
                    checkpoint, ensure_ascii=False, separators=(',', ':')).encode()).decode()
                # encrypted_content is Codex's wire field name. This payload is
                # intentionally plain, inspectable text encoded for transport.
                output = [{'type': 'compaction', 'id': 'cmp_' + uuid.uuid4().hex,
                           'encrypted_content': marker}]
            else:
                message, usage = self._generate(prefix + messages, tools, request)
                output = chat_output(message, routes)
            input_tokens = usage.get('prompt_tokens', 0) or 0
            output_tokens = usage.get('completion_tokens', 0) or 0
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
            response = {'id': 'resp_' + uuid.uuid4().hex, 'object':
                'response.compaction' if force_compact else 'response',
                'created_at': int(time.time()), 'status': 'completed',
                'model': request.get('model', self.model), 'output': output,
                'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                          'total_tokens': input_tokens + output_tokens,
                          'input_tokens_details': {'cached_tokens': cached}}}
            state['responses'][request_key] = response
            atomic_json(self._path(thread), state)
            return response

    def close(self):
        with self._lock:
            self._closed = True


def create_app(provider, *, token):
    import secrets
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title='r-codex text provider')

    @app.middleware('http')
    async def local_auth(request, call_next):
        supplied = request.headers.get('authorization', '')
        if not secrets.compare_digest(supplied, 'Bearer ' + token):
            return JSONResponse({'error': {'message': 'Local session token required'}}, status_code=401)
        return await call_next(request)

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/v1/models')
    def models():
        return {'data': [{'id': provider.model, 'object': 'model'}]}

    @app.post('/v1/responses')
    @app.post('/v1/responses/compact')
    async def responses(request: Request):
        try:
            body = await request.json()
            result = await run_in_threadpool(provider.handle, body, dict(request.headers),
                force_compact=request.url.path.endswith('/compact'))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except (RuntimeError, httpx.HTTPError) as exc:
            raise HTTPException(502, str(exc)) from exc
        if body.get('stream') and not request.url.path.endswith('/compact'):
            return StreamingResponse(sse(result), media_type='text/event-stream')
        return result
    return app
