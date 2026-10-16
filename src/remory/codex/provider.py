"""A local Responses provider for native Codex, backed by one Remory server."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import httpx

from ..adapters.harness import ChatTemplate, GlmChatTemplate
from ..client import Client
from ..types import digest
from .protocol import content_text, messages_from_wire, parse_output, sse, tools_from_wire


class CodexChatTemplate(ChatTemplate):
    """Render Qwen's system/tools prefix without retaining a synthetic user turn.

    The upstream template requires a user even for a system-only prefix or a
    post-compaction tool continuation. A temporary boundary is removed before
    tokenization; it is never submitted to the model.
    """
    def summary(self, text):
        # Quoted vision markers in code/tool output are ordinary text, not image
        # attachments. Actual images use the separate _image request below.
        for marker in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>'):
            text = text.replace(marker, marker.replace('<', r'\u003c').replace('>', r'\u003e'))
        return super().summary(text)

    def split(self, prefix_messages, messages, *, generation=False):
        boundary = {'role': 'user', 'content': 'REMORY_TEMPLATE_BOUNDARY'}
        kwargs = dict(tools=self.tools, tokenize=False, **self.template_kwargs)
        rendered_prefix = self.tokenizer.apply_chat_template(prefix_messages + [boundary],
            add_generation_prompt=False, **kwargs)
        marker = '<|im_start|>user\nREMORY_TEMPLATE_BOUNDARY<|im_end|>\n'
        if not rendered_prefix.endswith(marker):
            raise ValueError('Unsupported Qwen chat-template boundary')
        prefix_text = rendered_prefix[:-len(marker)]
        full = self.tokenizer.apply_chat_template(prefix_messages + [boundary] + messages,
            add_generation_prompt=generation, **kwargs)
        if not full.startswith(rendered_prefix):
            raise ValueError('The native template changed its system/tools prefix')
        prefix = self.summary(prefix_text)
        full_ids = self.summary(prefix_text + full[len(rendered_prefix):])
        if full_ids[:len(prefix)] != prefix:
            raise ValueError('Tokenization changed the system/tools boundary')
        return prefix, full_ids[len(prefix):]


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
    def __init__(self, url, info, tokenizer, directory, *, session_id,
                 max_new_tokens=16384, transport=None):
        self.url, self.info, self.tokenizer = url.rstrip('/'), info, tokenizer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session_id, self.max_new_tokens = session_id, max_new_tokens
        self.transport = transport
        self.config = info['config']
        self.glm = self.config.get('schema') == 'remory_glm53_local_evaluation_v1'
        self._lock = threading.RLock()
        self._locks = {}
        self._closed = False
        self.meta_path = self.directory / 'provider.json'
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
            if self.meta['identity'] != info['identity']:
                raise ValueError('Server model/checkpoint changed; start a new r-codex session')
            if self.meta['session_id'] != session_id:
                raise ValueError('Session identity changed')
        else:
            self.meta = {'identity': info['identity'], 'session_id': session_id,
                         'root_thread_id': None, 'threads': {}}
            atomic_json(self.meta_path, self.meta)

    @contextmanager
    def client(self):
        with Client(self.url, session_id=self.session_id, timeout=1800,
                    transport=self.transport, trust_env=False) as client:
            yield client

    def _thread(self, headers):
        metadata = json.loads(headers.get('x-codex-turn-metadata', '{}'))
        thread_id = headers.get('thread-id') or metadata.get('thread_id') or 'root'
        if not isinstance(thread_id, str) or not 0 < len(thread_id) <= 128:
            raise ValueError('Invalid Codex thread ID')
        if metadata.get('thread_id', thread_id) != thread_id:
            raise ValueError('Conflicting Codex thread IDs')
        parent = metadata.get('parent_thread_id')
        with self._lock:
            if self._closed:
                raise RuntimeError('Provider is closing')
            if self.meta['root_thread_id'] is None:
                if parent:
                    raise ValueError('A parent session must be opened before a sub-agent')
                self.meta['root_thread_id'] = thread_id
            elif thread_id != self.meta['root_thread_id']:
                known = self.meta['threads'].get(thread_id)
                if not known and parent != self.meta['root_thread_id']:
                    raise ValueError('Only direct children of this session are supported')
            if thread_id not in self.meta['threads']:
                self.meta['threads'][thread_id] = {'parent': parent}
                atomic_json(self.meta_path, self.meta)
            return thread_id, self._locks.setdefault(thread_id, threading.RLock())

    def _path(self, thread):
        return self.directory / ('thread-' + hashlib.sha256(thread.encode()).hexdigest() + '.json')

    def _load(self, thread):
        path = self._path(thread)
        if path.exists():
            return json.loads(path.read_text())
        return {'checkpoints': {}, 'responses': {}, 'images': {}, 'folded': []}

    @staticmethod
    def _wire_items(items):
        if isinstance(items, str):
            return [{'role': 'user', 'content': items}]
        return [{k: v for k, v in item.items() if k not in {'id', 'status'}}
                for item in items if item.get('type') != 'compaction_trigger']

    def _image(self, url, state):
        if not isinstance(url, str) or not url.startswith(('data:image/', 'https://', 'http://')):
            raise ValueError('Image input requires a data URL or HTTP image URL')
        key = hashlib.sha256(url.encode()).hexdigest()
        if key not in state['images']:
            if self.glm:
                raise ValueError('Image observations currently require the Qwen actor')
            prompt = ('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n'
                      'Describe this image for a coding agent. Include visible text, layout, '
                      'errors, and details useful for debugging. Do not invent unseen content.'
                      '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            with httpx.Client(timeout=1800, transport=self.transport, trust_env=False) as http:
                response = http.post(self.url + '/generate', json={'text': prompt,
                    'image_data': [url], 'sampling_params': {'temperature': 0, 'max_new_tokens': 2048}})
                response.raise_for_status()
                result = response.json()
            reason = result.get('meta_info', {}).get('finish_reason', {})
            if isinstance(reason, dict) and reason.get('type') in {'abort', 'error'}:
                raise RuntimeError('SGLang could not inspect the image')
            text = result['text'].split('</think>')[-1].strip()
            if not text:
                raise RuntimeError('The model returned no image observation')
            state['images'][key] = '[Image observation by the deployed Qwen model]\n' + text
        return state['images'][key]

    def _context(self, request, state):
        items = request.get('input', [])
        if isinstance(items, str):
            items = [{'role': 'user', 'content': items}]
        if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
            raise ValueError('Responses input must be a list of items')
        trigger = [i for i, x in enumerate(items) if x.get('type') == 'compaction_trigger']
        if trigger and (trigger != [len(items) - 1] or set(items[-1]) != {'type'}):
            raise ValueError('compaction_trigger must be the last item without a payload')
        if trigger:
            items = items[:-1]
        complete_items = items
        tools, routes = tools_from_wire(request.get('tools', []))
        instructions = request.get('instructions') or 'You are a coding assistant. Use the available tools to complete the user task.'
        leading = []
        while items and items[0].get('role') in {'system', 'developer'}:
            leading.append(content_text(items[0].get('content')))
            items = items[1:]
        prefix = [{'role': 'system', 'content': '\n\n'.join([instructions, *leading])}]
        codec = (GlmChatTemplate(self.tokenizer, tools=tools) if self.glm else
                 CodexChatTemplate(self.tokenizer, tools=tools, template_kwargs={
                     'enable_thinking': True, 'preserve_thinking': True}))
        indices = [i for i, x in enumerate(items) if x.get('type') == 'compaction']
        previous = None
        if indices:
            index = indices[-1]
            marker = items[index].get('encrypted_content', '')
            if not isinstance(marker, str) or not marker.startswith('remory.v1.'):
                raise ValueError('Foreign compaction checkpoint')
            previous = state['checkpoints'].get(marker)
            if previous is None:
                raise ValueError('Unknown checkpoint for this thread; resume the original session')
            if previous['tools_sha256'] != digest(tools) or (
                    previous.get('instructions', instructions) != instructions):
                raise ValueError('Instructions or tools changed after compact; start a new session')
            if leading and previous['prefix'] != prefix:
                raise ValueError('Developer instructions changed after compact; start a new session')
            prefix = previous['prefix']
            items = items[index + 1:]
        automatic = state.get('auto_checkpoint')
        if automatic:
            base = automatic['wire_base']
            if self._wire_items(complete_items[:len(base)]) == base:
                if automatic['tools_sha256'] != digest(tools) or automatic['prefix'] != prefix:
                    raise ValueError('Instructions or tools changed after sub-agent folding')
                previous = automatic
                items = complete_items[len(base):]
        messages = messages_from_wire(items, image_reader=lambda url: self._image(url, state))
        for message in messages:
            if message['role'] == 'system':
                message['role'] = 'user'
                message['content'] = '[Developer instructions from the harness]\n' + message['content']
        return codec, prefix, messages, previous, routes, bool(trigger), tools

    def _compact(self, codec, prefix, messages, previous, tools, client, branches=()):
        if not messages and previous is None:
            raise ValueError('There is no history to compact')
        prefix_ids, history_ids = codec.split(prefix, messages)
        contract = self.config.get('summary_contract', {})
        suffix = contract.get('generation_suffix_ids')
        summary_messages = messages
        if branches:
            summary_messages = messages + [{'role': 'user', 'content':
                '[Completed sub-agent results]\n' + '\n\n'.join(b['summary'] for b in branches)}]
        if suffix:
            _, summary_history = codec.split(prefix, summary_messages)
            tail = summary_history + suffix
        else:
            _, tail = codec.split(prefix, summary_messages + [{'role': 'user', 'content':
                'Write a concise checkpoint with the task, progress, constraints and next steps.'}],
                generation=True)
        kwargs = ({'handle': previous['handle'], 'continuation_ids': tail} if previous
                  else {'input_ids': prefix_ids + tail})
        sampling = {'temperature': 0}
        if contract.get('output_schema'):
            sampling['json_schema'] = json.dumps(contract['output_schema'])
        result = client.generate(**kwargs,
                                 max_new_tokens=min(contract.get('max_summary_tokens', 4096),
                                                    self.max_new_tokens),
                                 sampling=sampling)
        reason = result.get('meta_info', {}).get('finish_reason')
        reason = reason.get('type') if isinstance(reason, dict) else reason
        if reason != 'stop':
            raise RuntimeError('Summary did not complete; the original context is retained')
        summary = result['text'].split('</think>')[-1]
        for end in ['<|im_end|>', '<|endoftext|>', '[eop]']:
            summary = summary.removesuffix(end).rstrip()
        if not summary.strip():
            raise RuntimeError('Summary is empty; the original context is retained')
        memory = client.compact(prefix_ids=prefix_ids, history_ids=history_ids,
            summary_ids=codec.summary(summary), previous=previous['handle'] if previous else None,
            branches=[branch['handle'] for branch in branches])
        return {'handle': memory['handle'], 'summary': summary.strip(), 'prefix': prefix,
                'tools_sha256': digest(tools), 'receipt': memory['receipt']}, result

    def handle(self, request, headers=None, *, force_compact=False):
        if not isinstance(request, dict):
            raise ValueError('Responses request must be an object')
        thread, lock = self._thread(headers or {})
        with lock:
            state = self._load(thread)
            request_key = digest({'request': request, 'force_compact': force_compact})
            if request_key in state['responses']:
                return state['responses'][request_key]
            codec, prefix, messages, previous, routes, compact, tools = self._context(request, state)
            with self.client() as client:
                branches = []
                if thread == self.meta['root_thread_id'] and not self.glm:
                    with self._lock:
                        children = deepcopy(self.meta['threads'])
                    for child in children.values():
                        completion = child.get('completion')
                        if (child.get('parent') == thread and completion
                                and completion['handle'] not in state.get('folded', [])):
                            branches.append(completion)
                if branches and not (compact or force_compact):
                    previous, _ = self._compact(codec, prefix, messages, previous, tools, client, branches)
                    previous['instructions'] = request.get('instructions') or 'You are a coding assistant. Use the available tools to complete the user task.'
                    previous['wire_base'] = self._wire_items(request.get('input', []))
                    state['auto_checkpoint'] = previous
                    state.setdefault('folded', []).extend(b['handle'] for b in branches)
                    messages = []
                if compact or force_compact:
                    checkpoint, generated = self._compact(codec, prefix, messages, previous, tools, client, branches)
                    checkpoint['instructions'] = request.get('instructions') or 'You are a coding assistant. Use the available tools to complete the user task.'
                    marker = 'remory.v1.' + checkpoint['handle']
                    state['checkpoints'][marker] = checkpoint
                    state.pop('auto_checkpoint', None)
                    state.setdefault('folded', []).extend(b['handle'] for b in branches)
                    output = [{'type': 'compaction', 'id': 'cmp_' + uuid.uuid4().hex,
                               'encrypted_content': marker}]
                else:
                    prefix_ids, tail = codec.split(prefix, messages, generation=True)
                    kwargs = ({'handle': previous['handle'], 'continuation_ids': tail} if previous
                              else {'input_ids': prefix_ids + tail})
                    generated = client.generate(**kwargs, max_new_tokens=self.max_new_tokens,
                        sampling={'temperature': 0.6, 'top_p': 0.95, 'top_k': 20})
                    reason = generated.get('meta_info', {}).get('finish_reason')
                    kind = reason.get('type') if isinstance(reason, dict) else reason
                    if kind not in {'stop', 'length'}:
                        raise RuntimeError('SGLang generation did not complete')
                    output = parse_output(generated['text'], routes, thinking=True)
                    state['last_input'] = deepcopy(request)
                    state['last_output'] = output
                    # Seal a completed child before its final answer reaches the
                    # parent. On the next parent request its memory is folded once.
                    if (thread != self.meta['root_thread_id'] and not self.glm
                            and any(x.get('channel') == 'final' for x in output)
                            and not any(x['type'] in {'function_call', 'custom_tool_call'} for x in output)):
                        completed = messages + messages_from_wire(output)
                        checkpoint, _ = self._compact(codec, prefix, completed, previous, tools, client)
                        state['completion'] = checkpoint
            meta = generated.get('meta_info', {})
            input_tokens = meta.get('prompt_tokens', meta.get('input_tokens', 0))
            output_tokens = meta.get('completion_tokens', meta.get('output_tokens', 0))
            response = {'id': 'resp_' + uuid.uuid4().hex, 'object':
                'response.compaction' if force_compact else 'response',
                'created_at': int(time.time()), 'status': 'completed',
                'model': request.get('model', 'remory'), 'output': output,
                'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                          'total_tokens': input_tokens + output_tokens,
                          'input_tokens_details': {'cached_tokens': 0}}}
            state['responses'][request_key] = response
            atomic_json(self._path(thread), state)
            with self._lock:
                self.meta['threads'][thread]['last_activity'] = time.time()
                if state.get('completion'):
                    self.meta['threads'][thread]['completion'] = state['completion']
                atomic_json(self.meta_path, self.meta)
            return response

    def close(self):
        with self._lock:
            self._closed = True


def create_app(provider, *, token):
    import secrets
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title='Remory Codex provider')

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
        return {'data': [{'id': 'remory', 'object': 'model'}]}

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
