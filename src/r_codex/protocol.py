"""Translate native Codex Responses items to OpenAI-compatible chat messages."""
import json
import re
import uuid

REASONING_BOUNDARY_POLICY = 'ignore_explicitly_quoted_think_close_v1'


def _reasoning_boundary(text):
    """Keep explicitly quoted transcript delimiters inside the reasoning text.

    A copied JSON transcript can contain literal backslash-n around </think>.
    Those characters are data, unlike the actual newline-delimited boundary.
    Do not use rsplit: the visible tool arguments may also quote </think>.
    Only skip unambiguous local quoting; malformed calls still fail closed.
    """
    for match in re.finditer(r'</think>', text):
        before, after = text[:match.start()], text[match.end():]
        escaped_line = (before.endswith(r'\n')
                        and re.match(r'\\+(?:n|r\\+n)', after) is not None)
        quoted_marker = any(before.endswith(q) and after.startswith(q)
                            for q in ('`', '"', "'"))
        if not escaped_line and not quoted_marker:
            return match.start()
    return None


def tools_from_wire(tools):
    native, routes = [], {}
    def add(tool, namespace=None):
        kind = tool.get('type')
        if kind == 'namespace':
            for child in tool.get('tools', []):
                add(child, tool['name'])
            return
        if kind not in {'function', 'custom'}:
            raise ValueError(f'Unsupported Codex tool type: {kind}; use a function or MCP tool')
        name = tool['name']
        alias = f'{namespace}__{name}' if namespace else name
        if alias in routes:
            raise ValueError('Duplicate tool alias: ' + alias)
        parameters = tool.get('parameters') if kind == 'function' else {
            'type': 'object', 'properties': {'input': {'type': 'string'}},
            'required': ['input'], 'additionalProperties': False}
        if not isinstance(parameters, dict):
            raise ValueError('Tool parameters are missing: ' + alias)
        description = tool.get('description', '')
        routes[alias] = {'name': name, 'namespace': namespace, 'type': kind,
                         'parameters': parameters}
        native.append({'type': 'function', 'function': {
            'name': alias, 'description': description, 'parameters': parameters}})
    for tool in tools:
        add(tool)
    return native, routes


def content_text(content, *, image_reader=None):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError('Expected message text or content blocks')
    text = []
    for part in content:
        kind = part.get('type')
        if kind in {'input_text', 'output_text', 'text'}:
            text.append(part['text'])
        elif kind in {'input_image', 'image_url'}:
            if image_reader is None:
                raise ValueError('This provider has no image reader configured')
            url = part.get('image_url', part.get('url'))
            if isinstance(url, dict):
                url = url.get('url')
            text.append(image_reader(url))
        else:
            raise ValueError(f'Unsupported content block: {kind}')
    return '\n'.join(text)


def chat_content(content):
    """Keep image data/URLs as image blocks; no model-specific image tokenizer."""
    if content is None or isinstance(content, str):
        return content or ''
    if not isinstance(content, list):
        raise ValueError('Expected message text or content blocks')
    blocks = []
    for part in content:
        kind = part.get('type')
        if kind in {'input_text', 'output_text', 'text'}:
            blocks.append({'type': 'text', 'text': part['text']})
        elif kind in {'input_image', 'image_url'}:
            image = part.get('image_url', part.get('url'))
            if isinstance(image, str):
                image = {'url': image}
            if not isinstance(image, dict) or not isinstance(image.get('url'), str):
                raise ValueError('Image input requires an image URL')
            if not image['url'].startswith(('data:image/', 'https://', 'http://')):
                raise ValueError('Image input requires a data URL or HTTP image URL')
            image = dict(image)
            if part.get('detail'):
                image['detail'] = part['detail']
            blocks.append({'type': 'image_url', 'image_url': image})
        else:
            raise ValueError(f'Unsupported content block: {kind}')
    if all(part['type'] == 'text' for part in blocks):
        return '\n'.join(part['text'] for part in blocks)
    return blocks


def messages_from_wire(items, *, image_reader=None):
    messages, calls = [], {}
    pending = set()
    tool_images = []
    for item in items:
        kind = item.get('type', 'message' if 'role' in item else None)
        if kind == 'message':
            role = item.get('role')
            if role == 'developer':
                role = 'system'
            if role not in {'user', 'assistant', 'system'}:
                raise ValueError('Unsupported message role: ' + str(role))
            text = (content_text(item.get('content'), image_reader=image_reader)
                    if image_reader is not None else chat_content(item.get('content')))
            if role == 'assistant' and messages and messages[-1]['role'] == 'assistant':
                previous = messages[-1]['content']
                if not isinstance(previous, str) or not isinstance(text, str):
                    raise ValueError('Assistant image outputs are not supported')
                messages[-1]['content'] = previous + ('\n' if previous and text else '') + text
            else:
                messages.append({'role': role, 'content': text})
        elif kind in {'function_call', 'custom_tool_call'}:
            alias = (item['namespace'] + '__' if item.get('namespace') else '') + item['name']
            args = ({'input': item['input']} if kind == 'custom_tool_call'
                    else json.loads(item['arguments']))
            if not isinstance(args, dict):
                raise ValueError('Tool arguments must be an object')
            call = {'id': item['call_id'], 'type': 'function',
                    'function': {'name': alias, 'arguments': json.dumps(args, ensure_ascii=False)}}
            if item['call_id'] in calls:
                raise ValueError('Duplicate tool call ID')
            calls[item['call_id']] = alias
            pending.add(item['call_id'])
            if messages and messages[-1]['role'] == 'assistant':
                messages[-1].setdefault('tool_calls', []).append(call)
            else:
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
        elif kind in {'function_call_output', 'custom_tool_call_output'}:
            if item['call_id'] not in pending:
                raise ValueError('Tool output has no pending call: ' + item['call_id'])
            pending.remove(item['call_id'])
            output = item.get('output', '')
            if not isinstance(output, (list, str)):
                output = json.dumps(output, ensure_ascii=False)
            content = (content_text(output, image_reader=image_reader)
                       if image_reader is not None else chat_content(output))
            if isinstance(content, list):
                tool_images.extend(part for part in content if part['type'] == 'image_url')
                content = '\n'.join(part['text'] for part in content if part['type'] == 'text')
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'],
                             'content': content})
            if not pending and tool_images:
                messages.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Image observations returned by the preceding tools.'},
                    *tool_images]})
                tool_images = []
        elif kind == 'reasoning':
            # Codex returns the provider's readable reasoning with the transcript.
            summary = item.get('summary', [])
            text = '\n'.join(p.get('text', '') for p in summary)
            if text:
                if messages and messages[-1]['role'] == 'assistant':
                    previous = messages[-1].get('reasoning_content', '')
                    messages[-1]['reasoning_content'] = previous + ('\n' if previous else '') + text
                else:
                    messages.append({'role': 'assistant', 'content': '', 'reasoning_content': text})
        elif kind in {'compaction', 'compaction_trigger'}:
            raise ValueError('Compaction items must be resolved before rendering')
        else:
            raise ValueError('Unsupported Responses item: ' + str(kind))
    if pending:
        raise ValueError('Missing tool outputs: ' + ', '.join(sorted(pending)))
    return messages


def _coerce(text, schema):
    if schema.get('type') == 'string':
        return text
    try:
        return json.loads(text)
    except ValueError:
        if schema.get('type') in {None, 'string'}:
            return text
        raise ValueError('Invalid JSON tool argument') from None


def parse_output(text, routes, *, thinking=True):
    """Parse native Qwen XML; accept standard JSON tool calls and GLM XML too."""
    reasoning = ''
    boundary = _reasoning_boundary(text)
    if boundary is not None:
        reasoning, text = text[:boundary], text[boundary + len('</think>'):]
        reasoning = reasoning.removeprefix('<think>').strip()
    elif thinking and ('</think>' in text or '<tool_call>' not in text):
        # A length-limited unfinished reasoning trace must not become a final answer.
        raise ValueError('The model did not finish its reasoning')
    for marker in ['<|im_end|>', '<|endoftext|>', '<|fim_suffix|>', '[eop]']:
        text = text.removesuffix(marker).rstrip()
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if text.count('<tool_call>') != len(blocks):
        raise ValueError('Incomplete native tool call')
    output = []
    for block in blocks:
        block = block.strip()
        if block.startswith('{'):
            call = json.loads(block)
            call = call.get('function', call)
            name, args = call['name'], call.get('arguments', {})
            if isinstance(args, str):
                args = json.loads(args)
        else:
            match = re.fullmatch(r'<function=([^>]+)>\s*(.*?)\s*</function>', block, re.S)
            if match:
                name, params = match.groups()
                pairs = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', params, re.S)
                if re.sub(r'<parameter=[^>]+>.*?</parameter>', '', params, flags=re.S).strip():
                    raise ValueError('Malformed native tool arguments')
            else:
                name, _, params = block.partition('<arg_key>')
                name = name.strip()
                pairs = re.findall(r'([^<]+)<arg_value>(.*?)(?=<arg_key>|$)', params, re.S)
            if name not in routes:
                raise ValueError('Unknown generated tool: ' + name)
            properties = routes[name]['parameters'].get('properties', {})
            args = {key.strip(): _coerce(value.strip(), properties.get(key.strip(), {}))
                    for key, value in pairs}
            if len(args) != len(pairs):
                raise ValueError('Duplicate native tool parameter')
        if name not in routes or not isinstance(args, dict):
            raise ValueError('Unknown tool or malformed arguments')
        from jsonschema import ValidationError, validate
        route = routes[name]
        try:
            validate(args, route['parameters'])
        except ValidationError as exc:
            raise ValueError(f'Invalid arguments for {name}: {exc.message}') from exc
        call_id = 'call_' + uuid.uuid4().hex
        item = {'id': 'fc_' + uuid.uuid4().hex, 'call_id': call_id,
                'name': route['name'], 'status': 'completed'}
        if route['namespace']:
            item['namespace'] = route['namespace']
        if route['type'] == 'custom':
            item.update(type='custom_tool_call', input=args['input'])
        else:
            item.update(type='function_call', arguments=json.dumps(args, ensure_ascii=False))
        output.append(item)
    prose = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S).strip()
    if prose:
        output.insert(0, {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message',
            'role': 'assistant', 'status': 'completed', 'channel': 'commentary' if blocks else 'final',
            'content': [{'type': 'output_text', 'text': prose, 'annotations': []}]})
    if reasoning:
        output.insert(0, {'id': 'rs_' + uuid.uuid4().hex, 'type': 'reasoning',
            'summary': [{'type': 'summary_text', 'text': reasoning}]})
    if not any(item['type'] != 'reasoning' for item in output):
        raise ValueError('The model returned no complete message or tool call')
    return output


def chat_output(message, routes):
    """Convert structured Chat Completions outputs to native Codex Responses."""
    text = message.get('content') or ''
    if isinstance(text, list):
        text = content_text(text)
    if not isinstance(text, str):
        raise ValueError('Invalid assistant content from chat backend')
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    if not isinstance(reasoning, str):
        raise ValueError('Invalid reasoning content from chat backend')
    calls = message.get('tool_calls') or []
    if text.lstrip().startswith('<think>') and _reasoning_boundary(text) is None:
        raise ValueError('The model did not finish its reasoning')
    # Some local OpenAI-compatible servers omit a tool-call parser. Accept their
    # native XML format, while preferring structured tool calls when provided.
    if not calls and ('<tool_call>' in text or '</think>' in text):
        result = parse_output(text, routes, thinking=False)
        if reasoning:
            result.insert(0, {'id': 'rs_' + uuid.uuid4().hex, 'type': 'reasoning',
                'summary': [{'type': 'summary_text', 'text': reasoning}]})
        return result
    output, identifiers = [], set()
    for call in calls:
        if call.get('type', 'function') != 'function':
            raise ValueError('Unsupported generated chat tool type')
        function = call['function']
        name = function['name']
        if name not in routes:
            raise ValueError('Unknown generated tool: ' + name)
        args = function.get('arguments', '{}')
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be an object')
        from jsonschema import ValidationError, validate
        route = routes[name]
        try:
            validate(args, route['parameters'])
        except ValidationError as exc:
            raise ValueError(f'Invalid arguments for {name}: {exc.message}') from exc
        call_id = call.get('id') or 'call_' + uuid.uuid4().hex
        if call_id in identifiers:
            raise ValueError('Duplicate generated tool call ID')
        identifiers.add(call_id)
        item = {'id': 'fc_' + uuid.uuid4().hex, 'call_id': call_id,
                'name': route['name'], 'status': 'completed'}
        if route['namespace']:
            item['namespace'] = route['namespace']
        if route['type'] == 'custom':
            item.update(type='custom_tool_call', input=args['input'])
        else:
            item.update(type='function_call', arguments=json.dumps(args, ensure_ascii=False))
        output.append(item)
    if text:
        output.insert(0, {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message',
            'role': 'assistant', 'status': 'completed', 'channel': 'commentary' if calls else 'final',
            'content': [{'type': 'output_text', 'text': text, 'annotations': []}]})
    if reasoning:
        output.insert(0, {'id': 'rs_' + uuid.uuid4().hex, 'type': 'reasoning',
            'summary': [{'type': 'summary_text', 'text': reasoning}]})
    if not any(item['type'] != 'reasoning' for item in output):
        raise ValueError('The model returned no complete message or tool call')
    return output


def response_events(response):
    """The same events are used for buffered and streaming native Codex requests."""
    yield {'type': 'response.created', 'response': {**response, 'output': [], 'status': 'in_progress'}}
    for index, item in enumerate(response['output']):
        opened = {**item, 'status': 'in_progress'}
        if item['type'] == 'message':
            opened['content'] = []
        elif item['type'] == 'function_call':
            opened['arguments'] = ''
        elif item['type'] == 'custom_tool_call':
            opened['input'] = ''
        yield {'type': 'response.output_item.added', 'output_index': index,
               'item': opened}
        if item['type'] == 'message':
            for part_index, part in enumerate(item['content']):
                yield {'type': 'response.output_text.delta', 'item_id': item['id'],
                       'output_index': index, 'content_index': part_index, 'delta': part['text']}
        elif item['type'] == 'function_call':
            yield {'type': 'response.function_call_arguments.delta', 'item_id': item['id'],
                   'output_index': index, 'delta': item['arguments']}
        elif item['type'] == 'custom_tool_call':
            yield {'type': 'response.custom_tool_call_input.delta', 'item_id': item['id'],
                   'output_index': index, 'delta': item['input']}
        yield {'type': 'response.output_item.done', 'output_index': index, 'item': item}
    yield {'type': 'response.completed', 'response': response}


def sse(response):
    for sequence, event in enumerate(response_events(response)):
        event['sequence_number'] = sequence
        yield ('event: ' + event['type'] + '\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode()
    yield b'data: [DONE]\n\n'
