"""CPU preparation of portable inputs for the unchanged Qwen training entries.

The JSONL importer is an additional path for user-owned documents and trajectories.
It does not generate answers/summaries or reproduce the original dataset selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import zlib

from context_compression import qwen38_long_repeat_data as repeat
from context_compression.qwen38_allturn_data import (
    SCHEMA as ALLTURN_SCHEMA, digest, load_example, render_row,
)
from context_compression.qwen38_residual_data import SummaryCache, identity, residual_events
from context_compression.codex_compact_v0149 import HANDOFF_SCHEMA, structured_summary_contract
from context_compression.prepare_qwen38_residual_summaries import validate_dataset
from context_compression.prepare_qwen38_summary_selection import SCHEMA as SELECTION_SCHEMA, load_selection
from context_compression.qwen38_residual_curriculum import build_context_order, load_context_order
from context_compression.qwen38_residual_data import open_summary_manifest
from context_compression.stage2_batch_order import global_batches

ROOT = Path(__file__).resolve().parent
PRETRAINED_FILES = {
    'config.json': '873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980',
    'model.safetensors': '67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_summary(value):
    fields = HANDOFF_SCHEMA['properties']
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('summary must contain the five handoff fields')
    for key, spec in fields.items():
        items = value[key]
        if (not isinstance(items, list) or not spec.get('minItems', 0) <= len(items) <= 12
                or any(not isinstance(s, str) or not s.strip() for s in items)):
            raise ValueError(f'invalid summary field: {key}')


def stage1_data(input_path, target_model, output_dir, tokenizer, *, allow_short=False):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    db_path = out / 'repeat.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE examples(id INTEGER PRIMARY KEY, document_id TEXT, '
               'document_sha256 TEXT UNIQUE, split TEXT, token_start INT, original_tokens INT, '
               'token_count INT, tokens BLOB, url TEXT, edu_score REAL)')
    seen, stats, buckets = set(), Counter(), Counter()
    try:
        for row in rows(input_path):
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError('each document needs nonempty text')
            sha = hashlib.sha256(row['text'].encode()).hexdigest()
            if sha in seen:
                raise ValueError('duplicate document; train/validation overlap is not allowed')
            seen.add(sha)
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            if not ids or (not allow_short and len(ids) < 8192):
                raise ValueError('Stage I requires at least 8192 tokens; short examples need --allow-short-examples')
            count = min(len(ids), 65536)
            start = int(sha[16:32], 16) % (len(ids) - count + 1)
            split = row.get('split', repeat.split_for(sha))
            if split not in ('train', 'validation'):
                raise ValueError('split must be train or validation')
            db.execute('INSERT INTO examples(document_id,document_sha256,split,token_start,'
                       'original_tokens,token_count,tokens,url,edu_score) VALUES(?,?,?,?,?,?,?,?,?)',
                       (str(row['id']), sha, split, start, len(ids), count,
                        repeat.encode_ids(ids[start:start + count]), row.get('url', ''), None))
            stats[split + '_documents'] += 1
            stats[split + '_tokens'] += count
            buckets[f'{split}:{repeat.bucket(count)}'] += 1
        counts = dict(db.execute('SELECT split, COUNT(*) FROM examples GROUP BY split'))
        if not counts.get('train') or not counts.get('validation'):
            raise ValueError('provide document-disjoint train and validation examples')
        db.commit()
    finally:
        db.close()
    manifest = dict(schema=repeat.SCHEMA, database=db_path.name, database_sha256=digest(db_path),
                    tokenizer=repeat.tokenizer_receipt(target_model), counts=counts,
                    stats=dict(stats), length_buckets=dict(buckets),
                    source={'kind': 'user_jsonl', 'sha256': digest(Path(input_path))},
                    selection={'minimum_tokens': 1 if allow_short else 8192, 'maximum_tokens': 65536,
                               'document_deduplication': 'sha256(text)',
                               'short_format_examples': allow_short,
                               'education_filter': 'not_applied_by_jsonl_importer'})
    manifest['manifest_sha256'] = repeat.canonical(manifest)
    write_json(out / 'repeat.manifest.json', manifest)
    return out / 'repeat.manifest.json'


def make_stage2_example(row, tokenizer):
    messages = row['messages']
    if len(messages) < 2 or messages[-1].get('role') != 'assistant':
        raise ValueError('messages must end with the held-out assistant continuation')
    # The last assistant is the ONLY scored response; earlier messages form history.
    raw = dict(id=row['id'], messages=messages, tools=row.get('tools', []),
               validation={'valid': True}, truncated=False,
               generation_start_message_index=len(messages) - 1,
               generated_assistant_message_indices=[len(messages) - 1])
    example = render_row(raw, 'coding', tokenizer)
    return example, next(residual_events(example))


def stage2_data(input_path, target_model, output_dir, tokenizer, *, allow_short=False):
    input_sha = digest(Path(input_path))
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    db_path = out / 'continuations.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE examples(ordinal INTEGER PRIMARY KEY, key TEXT UNIQUE, corpus TEXT, '
               'responses INTEGER, tokens INTEGER, prefix_tokens INTEGER, max_response INTEGER, payload BLOB)')
    stats = Counter()
    try:
        for ordinal, row in enumerate(rows(input_path)):
            validate_summary(row['summary'])
            example, event = make_stage2_example(row, tokenizer)
            if not allow_short and len(event.history_ids) < 8192:
                raise ValueError('Stage II requires at least 8192 history tokens; use --allow-short-examples for format demos')
            target = len(event.target_ids)
            db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?,?,?)',
                       (ordinal, example['key'], 'coding', 1, target, len(event.prefix_ids), target,
                        zlib.compress(json.dumps(example, separators=(',', ':')).encode(), 1)))
            stats.update(trajectories=1, responses=1, target_tokens=target)
            stats['max_prefix_tokens'] = max(stats['max_prefix_tokens'], len(event.prefix_ids))
            stats['max_response_tokens'] = max(stats['max_response_tokens'], target)
        if not stats['trajectories']:
            raise ValueError('empty continuation dataset')
        if digest(Path(input_path)) != input_sha:
            raise ValueError('input changed during preparation')
        db.commit()
    finally:
        db.close()
    model = Path(target_model).resolve()
    tokenizer_files = repeat.tokenizer_receipt(model)
    for name in ['added_tokens.json']:
        if (model / name).is_file():
            tokenizer_files[name] = digest(model / name)
    manifest = dict(schema=ALLTURN_SCHEMA, database=str(db_path), database_sha256=digest(db_path),
                    stats={'coding': dict(stats)}, tokenizer_files=tokenizer_files,
                    tokenizer_model=str(model), data_implementation_sha256=digest(
                        ROOT / 'context_compression/qwen38_allturn_data.py'),
                    sources=[{'kind': 'user_jsonl', 'sha256': input_sha}],
                    coverage='last_assistant_response_per_input_row',
                    short_format_examples=allow_short)
    manifest_path = out / 'continuations.manifest.json'
    write_json(manifest_path, manifest)
    # Imported handoffs are explicitly identified; no model generation is claimed.
    contract = structured_summary_contract(tokenizer, database_sha256=manifest['database_sha256'],
                                          tokenizer_files=tokenizer_files,
                                          generator_model='user_provided_precomputed_handoff',
                                          max_summary_tokens=4096)
    contract['summary_origin'] = 'provided_by_caller_history_only_required'
    cache = SummaryCache(out / 'summaries.sqlite', contract)
    database = sqlite3.connect(db_path)
    try:
        # Stream again instead of retaining every long history in host memory.
        for ordinal, row in enumerate(rows(input_path)):
            event = next(residual_events(load_example(database, ordinal)))
            ids = tokenizer.encode(json.dumps(row['summary'], ensure_ascii=False), add_special_tokens=False)
            cache.put(event, ids, finish_reason='stop')
        if digest(Path(input_path)) != input_sha:
            raise ValueError('input changed during summary import')
        summary_manifest = cache.seal(database)
        plan = build_context_order(database, database_sha256=manifest['database_sha256'])
    finally:
        cache.close()
        database.close()
    write_json(out / 'summary.manifest.json', summary_manifest)
    write_json(out / 'context-order.json', plan)
    selection = dict(schema=SELECTION_SCHEMA, database_sha256=manifest['database_sha256'],
                     source_manifest=str(manifest_path), source_cache=str(out / 'summaries.sqlite'),
                     summary_manifest_sha256=digest(out / 'summary.manifest.json'),
                     policy='all_provided_validated_independent_events',
                     original_responses=stats['responses'], selected_responses=stats['responses'],
                     selected_target_tokens=stats['target_tokens'],
                     ordinals=list(range(stats['trajectories'])), excluded=[])
    selection['selection_sha256'] = identity(selection)
    write_json(out / 'selection.json', selection)
    return manifest_path


def pretrained_receipt(path):
    path = Path(path).resolve()
    config = read_json(path / 'config.json')
    if (config['hidden_size'], config['num_hidden_layers'], config['intermediate_size']) != (5120, 5, 17408):
        raise ValueError('expected the Qwen3.8-27B DFlash2 backbone')
    if config['dflash_config']['target_layer_ids'] != [5, 19, 33, 47, 61]:
        raise ValueError('unexpected target feature layers')
    files = {name: digest(path / name) for name in PRETRAINED_FILES}
    if files != PRETRAINED_FILES:
        raise ValueError('pretrained files differ from the pinned DFlash2 revision')
    return dict(directory=str(path), files=files)


def schedule(config, total, warmup):
    opt = config['optimizer']
    opt['total_steps'] = total
    if warmup is not None:
        opt['warmup_steps'] = warmup
    if not 0 <= opt['warmup_steps'] < total:
        raise ValueError(f'warmup must be below {total} total steps; set --warmup-steps explicitly for a small dataset')


def stage1_config(args):
    c = read_json(ROOT / 'configs/stage1.json')
    manifest, db = repeat.load_dataset(args.manifest, args.target_model)
    db.close()
    if manifest.get('selection', {}).get('short_format_examples'):
        raise ValueError('short format examples are not a production training corpus')
    backbone = {**c.pop('pretrained'), **pretrained_receipt(args.pretrained)}
    c.update(target_model=str(args.target_model.resolve()), dataset=manifest,
             manifest_sha256=digest(args.manifest), tokenizer=manifest['tokenizer'])
    c['compressor_initialization'] = dict(
        kind='pretrained_DFlash2_backbone', path=backbone['directory'],
        repository=backbone['repository'], revision=backbone['revision'],
        config_sha256=backbone['files']['config.json'],
        weights_sha256=backbone['files']['model.safetensors'])
    schedule(c, math.ceil(manifest['counts']['train'] / c['optimizer']['grad_accumulation']), args.warmup_steps)
    write_json(args.output, c)


def stage2_config(args):
    c = read_json(ROOT / 'configs/stage2.json')
    manifest, db = validate_dataset(args.manifest, args.target_model)
    db.close()
    if manifest.get('short_format_examples'):
        raise ValueError('short format examples are not a production training corpus')
    c['pretrained'].update(pretrained_receipt(args.pretrained))
    selected, selection = load_selection(args.selection, manifest, args.summary_manifest)
    order, curriculum = load_context_order(args.context_order_plan, dataset_manifest=manifest, canary_ordinals=selected)
    if len(order) < 4 or any(row[3] < 8192 for row in order):
        raise ValueError('Stage II needs at least four events with >=8192 history tokens')
    cache, summaries = open_summary_manifest(args.summary_manifest, dataset_manifest=manifest, canary_ordinals=selected)
    cache.close()
    checkpoint = args.stage1_checkpoint.resolve()
    trainer = read_json(checkpoint / 'trainer_state.json')
    run = read_json(checkpoint / 'run_config.json')
    contract = {k: v for k, v in run.items() if k not in ('run_contract_sha256', 'output_dir', 'resume_from')}
    if (trainer['schema'] != 'qwen38_pretrained_dflash2_stage1_3b_four_gpu_v1'
            or trainer['world_size'] != 4 or trainer['global_step'] <= 0
            or identity(contract) != trainer['run_contract_sha256']
            or run['run_contract_sha256'] != trainer['run_contract_sha256']):
        raise ValueError('invalid Stage I checkpoint contract')
    parent = run['source_contract']
    if parent['target_model'] != str(args.target_model.resolve()):
        raise ValueError('Stage I and Stage II must use the same frozen target path')
    c['compressor']['output_embedding_rms'] = parent['compressor']['output_embedding_rms']
    c.update(target_model=str(args.target_model.resolve()), manifest=str(args.manifest.resolve()),
             manifest_sha256=digest(args.manifest), database_sha256=manifest['database_sha256'],
             data_stats=manifest['stats'], summary_manifest=str(args.summary_manifest.resolve()),
             summary_manifest_sha256=digest(args.summary_manifest), summary_cache_sha256=summaries['cache_sha256'],
             summary_contract=summaries['contract'], training_selection=selection,
             context_curriculum=curriculum, canary_ordinals=[])
    c['residual']['coverage'] = manifest.get('coverage', 'independent_compression_events')
    if 'derivation' in manifest:
        c['residual']['data_derivation'] = manifest['derivation']
    c['stage1_checkpoint'] = dict(directory=str(checkpoint), global_step=trainer['global_step'],
        run_contract_sha256=trainer['run_contract_sha256'], files={name: digest(checkpoint / name)
        for name in ['model.pt', 'run_config.json', 'trainer_state.json']})
    c['initialization'] = dict(kind='StageI_checkpoint', stage1_checkpoint_used=True,
        stage1_global_step=trainer['global_step'], global_step=0, optimizer='fresh_AdamW',
        data_cursor='start_of_epoch_zero', new_modules=['summary_attention', 'depth_embedding'])
    total = sum(1 for _ in global_batches(order, 0, c['optimizer']['accumulate_min_tokens'])) * c['epochs']
    schedule(c, total, args.warmup_steps)
    receipt = dict(schema='qwen38_stage1_to_residual_initialization_v1', source_world_size=4,
                   global_step=0, cursor=dict(epoch=0, shard_index=0, event_index=0, chunk_index=0),
                   source_checkpoint=str(checkpoint), source_global_step=trainer['global_step'])
    write_json(args.output, dict(source_contract=c, initialization_receipt=receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for stage in (1, 2):
        p = sub.add_parser(f'stage{stage}-data')
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--target-model', type=Path, required=True)
        p.add_argument('--output-dir', type=Path, required=True)
        p.add_argument('--allow-short-examples', action='store_true')
        p = sub.add_parser(f'stage{stage}-config')
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--target-model', type=Path, required=True)
        p.add_argument('--pretrained', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--warmup-steps', type=int)
        if stage == 2:
            p.add_argument('--summary-manifest', type=Path, required=True)
            p.add_argument('--selection', type=Path, required=True)
            p.add_argument('--context-order-plan', type=Path, required=True)
            p.add_argument('--stage1-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if args.command.endswith('-config'):
        (stage1_config if args.command.startswith('stage1') else stage2_config)(args)
        print(args.output)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
        fn = stage1_data if args.command.startswith('stage1') else stage2_data
        print(fn(args.input, args.target_model, args.output_dir, tokenizer,
                 allow_short=args.allow_short_examples))


if __name__ == '__main__':
    main()
