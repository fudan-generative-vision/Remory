"""CPU checks for import coverage, held-out answers, and artifact integrity.

The tiny tokenizer isolates the data contracts. The real Qwen tokenizer is
checked separately by running the documented JSONL preparation commands.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare
from context_compression.qwen38_residual_data import summary_request


class Tokenizer:
    def encode(self, text, **kwargs):
        # A reversible character codec with one special end-of-message token.
        parts = text.split('<|im_end|>')
        ids = []
        for index, part in enumerate(parts):
            if index:
                ids.append(1)
            ids.extend(ord(c) + 10 for c in part)
        return ids

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        text = ''
        for message in messages:
            text += '<|im_start|>' + message['role'] + '\n'
            if message['role'] == 'assistant':
                text += '<think>\n' + message.get('reasoning_content', '') + '\n</think>\n\n'
            text += message['content'] + '<|im_end|>\n'
        if add_generation_prompt:
            text += '<|im_start|>assistant\n<think>\n'
        return text


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


@pytest.fixture
def tokenizer_dir(tmp_path):
    path = tmp_path / 'tokenizer'
    path.mkdir()
    (path / 'tokenizer.json').write_text('{"fixture": true}')
    return path


def test_document_import_is_loadable_and_disjoint(tmp_path, tokenizer_dir):
    path = prepare.stage1_data(EXAMPLES / 'stage1.documents.jsonl', tokenizer_dir,
                               tmp_path / 'data', Tokenizer(), allow_short=True)
    manifest, db = prepare.repeat.load_dataset(path, tokenizer_dir)
    assert manifest['counts'] == {'train': 1, 'validation': 1}
    assert db.execute('SELECT COUNT(DISTINCT document_sha256) FROM examples').fetchone()[0] == 2
    for (key,) in db.execute('SELECT id FROM examples'):
        assert prepare.repeat.load_example(db, key)['token_ids']
    db.close()


def test_duplicate_documents_cannot_leak_into_validation(tmp_path, tokenizer_dir):
    rows = list(prepare.rows(EXAMPLES / 'stage1.documents.jsonl'))
    rows[1]['text'] = rows[0]['text']
    source = tmp_path / 'duplicate.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='duplicate document'):
        prepare.stage1_data(source, tokenizer_dir, tmp_path / 'data', Tokenizer(), allow_short=True)


@pytest.mark.parametrize('stage', [1, 2])
def test_short_examples_require_explicit_opt_in(stage, tmp_path, tokenizer_dir):
    fn = prepare.stage1_data if stage == 1 else prepare.stage2_data
    source = 'stage1.documents.jsonl' if stage == 1 else 'stage2.continuations.jsonl'
    with pytest.raises(ValueError, match='8192'):
        fn(EXAMPLES / source, tokenizer_dir, tmp_path / 'data', Tokenizer())


@pytest.mark.parametrize('stage', [1, 2])
def test_short_fixture_cannot_become_production_configuration(stage, tmp_path, tokenizer_dir):
    fn = prepare.stage1_data if stage == 1 else prepare.stage2_data
    configure = prepare.stage1_config if stage == 1 else prepare.stage2_config
    source = 'stage1.documents.jsonl' if stage == 1 else 'stage2.continuations.jsonl'
    path = fn(EXAMPLES / source, tokenizer_dir, tmp_path / 'data', Tokenizer(), allow_short=True)
    with pytest.raises(ValueError, match='not a production training corpus'):
        configure(SimpleNamespace(manifest=path, target_model=tokenizer_dir))


def test_continuations_summaries_and_selection_round_trip(tmp_path, tokenizer_dir):
    out = tmp_path / 'data'
    path = prepare.stage2_data(EXAMPLES / 'stage2.continuations.jsonl', tokenizer_dir,
                               out, Tokenizer(), allow_short=True)
    manifest, db = prepare.validate_dataset(path, tokenizer_dir)
    selected, _ = prepare.load_selection(out / 'selection.json', manifest, out / 'summary.manifest.json')
    order, _ = prepare.load_context_order(out / 'context-order.json', dataset_manifest=manifest,
                                          canary_ordinals=selected)
    assert selected == [0, 1] and len(order) == 2
    cache, summary_manifest = prepare.open_summary_manifest(out / 'summary.manifest.json',
                                                            dataset_manifest=manifest)
    assert summary_manifest['coverage']['responses'] == 2
    try:
        event = next(prepare.residual_events(prepare.load_example(db, 0)))
        original = cache.get(event)
        # The current target answer must not enter the summarizer's input.
        row = copy.deepcopy(next(prepare.rows(EXAMPLES / 'stage2.continuations.jsonl')))
        row['messages'][-1]['content'] = 'A DIFFERENT HELD-OUT ANSWER'
        _, changed_answer = prepare.make_stage2_example(row, Tokenizer())
        assert event.target_ids != changed_answer.target_ids
        assert summary_request(event, cache.contract) == summary_request(changed_answer, cache.contract)
        assert cache.get(changed_answer) == original
        row['messages'][-2]['content'] += ' Changed history.'
        _, changed_history = prepare.make_stage2_example(row, Tokenizer())
        with pytest.raises(ValueError, match='different causal history'):
            cache.get(changed_history)
    finally:
        cache.close()
        db.close()
    with (out / 'summaries.sqlite').open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare.open_summary_manifest(out / 'summary.manifest.json', dataset_manifest=manifest)


def test_summary_must_use_handoff_schema():
    with pytest.raises(ValueError, match='five handoff fields'):
        prepare.validate_summary({'answer': 'not a handoff'})
    row = next(prepare.rows(EXAMPLES / 'stage2.continuations.jsonl'))
    row['summary']['current_progress'] = []
    with pytest.raises(ValueError, match='current_progress'):
        prepare.validate_summary(row['summary'])


def test_small_dataset_requires_explicit_warmup():
    config = {'optimizer': {'warmup_steps': 500}}
    with pytest.raises(ValueError, match='warmup'):
        prepare.schedule(config, 10, None)
    prepare.schedule(config, 10, 1)
    assert config['optimizer'] == {'warmup_steps': 1, 'total_steps': 10}
