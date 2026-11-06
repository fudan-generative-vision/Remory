"""CPU-only public Remory API over an attested legacy residual SGLang client.

This bridge owns new immutable checkpoints only. It does not import legacy
session handles, load model weights, install hooks, or modify shared services.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading

import numpy as np

from ..engine import Remory
from ..store import MemoryStore
from ..types import Contract, Generation, Prepared, digest, token_ids


class LegacyBackend:
    def __init__(self, client, tokenizer, checkpoint):
        self.client, self.tokenizer = client, tokenizer
        self.checkpoint = str(Path(checkpoint).resolve())
        self.lock = threading.RLock()
        attestation = client.attest_residual_server(self.checkpoint)
        self.config = deepcopy(client._residual_config)
        self.attestation = deepcopy(attestation)
        self.context_limit = attestation['context_limit']
        self.contract = Contract.from_config(self.config, identity=digest({
            'transport': 'legacy-residual-cpu-bridge.v1', 'endpoint': client.base_url,
            'checkpoint': self.checkpoint, 'attestation': attestation}))

    def verify(self):
        """Fail closed if the shared pool's encoder/model changes after startup."""
        current = self.client.attest_residual_server(self.checkpoint)
        if (current != self.attestation or self.client._residual_config != self.config):
            raise RuntimeError('Legacy backend identity changed; retain the original checkpoint')

    def generate(self, source, *, max_new_tokens, sampling=None):
        with self.lock:
            self.verify()
            kwargs = dict(max_new_tokens=max_new_tokens, sampling_overrides=dict(sampling or {}))
            if source.memory is None:
                result = self.client.generate_discrete(list(source.input_ids), **kwargs)
            else:
                result = self.client.generate_with_memory(list(source.input_ids),
                    memory_positions=list(source.memory_positions),
                    memory_embeddings=source.memory.tolist(),
                    expected_hidden_size=self.contract.hidden_size, **kwargs)
            return Generation(text=self.tokenizer.decode(result.token_ids, skip_special_tokens=False),
                output_ids=list(result.token_ids), usage=dict(result.meta_info),
                finish_reason=result.finish_reason)


def _cpu_tensor(array):
    import torch
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))


class LegacyBridge(Remory):
    def __init__(self, client, tokenizer, checkpoint, store, *, tensor_factory=_cpu_tensor):
        backend = LegacyBackend(client, tokenizer, checkpoint)
        super().__init__(backend, store)
        self.model_config = deepcopy(backend.config)
        self.tensor_factory = tensor_factory
        # No `online` attribute: public server rejects online training requests.

    def compact(self, *, owner, prefix_ids, history_ids, summary_ids, previous=None, branches=()):
        self.store._key(owner)
        prefix = token_ids(prefix_ids, 'prefix_ids')
        history = token_ids(history_ids, 'history_ids', empty=previous is not None or bool(branches))
        summary = token_ids(summary_ids, 'summary_ids')
        if not isinstance(branches, (list, tuple)) or len(branches) > 16:
            raise ValueError('branches must contain at most 16 memory handles')
        if len(set(branches)) != len(branches) or previous in branches:
            raise ValueError('branch memory must not be included twice')
        c = self.contract
        if previous:
            old = self.store.get(owner, previous)
            self._check_memory(old)
            if old.prefix_ids != prefix:
                raise ValueError('original system/tools/task prefix changed across compaction')
            source = self._prepare(owner=owner, handle=previous, continuation_ids=history)
        else:
            source = Prepared(prefix + history)
        for handle in branches:
            child = self.store.get(owner, handle)
            self._check_memory(child)
            lead = source.input_ids + c.summary_before + child.summary_ids + c.summary_after + c.memory_before
            positions = tuple(range(len(lead), len(lead) + len(child.embeddings)))
            memory = (np.concatenate([source.memory, child.embeddings])
                      if source.memory is not None else child.embeddings)
            source = Prepared(lead + (c.placeholder_id,) * len(positions) + c.memory_after,
                              source.memory_positions + positions, memory,
                              source.slot_depths + child.slot_depths)
        self._check_length(max(len(source.input_ids),
            len(prefix) + len(c.summary_before) + len(summary) + len(c.summary_after),
            len(prefix) + c.block_tokens))
        # Preflight the exact restored width before the legacy client allocates anything.
        from ..pyramid import plan_pyramid
        history_length = len(source.input_ids) - len(prefix)
        depths = [0] * ((history_length + c.block_tokens - 1) // c.block_tokens)
        for position, depth in zip(source.memory_positions, source.slot_depths):
            index = (position - len(prefix)) // c.block_tokens
            depths[index] = max(depths[index], depth)
        plan = plan_pyramid(history_length, budget=c.memory_budget, block=c.block_tokens,
                            ratio=c.ratio, max_depth=None, leaf_input_depths=depths)
        self._check_length(len(prefix) + len(c.summary_before) + len(summary) + len(c.summary_after)
                           + len(c.memory_before) + plan.soft_tokens + len(c.memory_after))
        with self.backend.lock:
            self.backend.verify()
            kwargs = {}
            if source.memory is not None:
                kwargs = dict(source_memory=self.tensor_factory(source.memory),
                              source_memory_positions=list(source.memory_positions),
                              source_memory_depths=list(source.slot_depths))
            result = self.backend.client.build_residual_memory(list(prefix),
                list(source.input_ids[len(prefix):]), list(summary),
                contract=self.model_config['summary_contract'], memory_budget=c.memory_budget,
                block_tokens=c.block_tokens, max_depth=c.max_depth, **kwargs)
            values = result['memory_embeddings']
            if hasattr(values, 'detach'):
                values = values.detach().float().cpu().numpy()
            receipt = deepcopy(result['receipt'])
            depths = tuple(receipt['slot_depths'])
            memory = self._matrix(values, len(depths))
            if len(memory) != plan.soft_tokens:
                raise RuntimeError('Legacy client returned inconsistent frontier size')
            receipt.update(schema='remory.legacy_bridge.compaction.v1', previous=previous,
                           branches=list(branches), bridge_identity=c.identity,
                           encoder_checkpoint=self.backend.checkpoint,
                           legacy_endpoint=self.backend.client.base_url,
                           source_sha256=digest(source.input_ids))
            return self.store.put(owner=owner, identity=c.identity, prefix_ids=prefix,
                summary_ids=summary, embeddings=memory, slot_depths=depths, receipt=receipt)


def build_bridge(*, legacy_workspace, checkpoint, endpoint, store_path, tokenizer=None):
    """Connect by GET attestation only; model requests begin only at public API use."""
    import sys
    workspace = str(Path(legacy_workspace).resolve(strict=True))
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from context_compression.qwen38_residual_sglang_client import SummaryResidualSGLangClient
    from context_compression.qwen38_residual_serving_contract import load_residual_serving_contract
    config = load_residual_serving_contract(checkpoint, verify_weights=False)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['target_model'], local_files_only=True)
    client = SummaryResidualSGLangClient(endpoint,
        eos_token_id=config['summary_contract']['end_token_id'], enable_request_cancellation=False)
    return LegacyBridge(client, tokenizer, checkpoint, MemoryStore(store_path))


def retain_compatible_identity(engine, previous):
    """Explicit deployment migration: weights may change, memory ABI may not."""
    import re
    if (previous.get('config') != engine.model_config or
            not re.fullmatch(r'[0-9a-f]{64}', previous.get('identity', ''))):
        raise ValueError('Previous bridge identity has an incompatible memory contract')
    engine.contract = replace(engine.contract, identity=previous['identity'])
    engine.backend.contract = engine.contract


def main():
    import argparse
    import uvicorn
    from ..server import create_app
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-workspace', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--store', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8421)
    parser.add_argument('--compatible-identity-info', type=Path,
                        help='Explicit prior /v1/remory/info receipt for an identical memory ABI')
    args = parser.parse_args()
    engine = build_bridge(legacy_workspace=args.legacy_workspace, checkpoint=args.checkpoint,
                          endpoint=args.endpoint, store_path=args.store)
    if args.compatible_identity_info:
        retain_compatible_identity(engine, json.loads(args.compatible_identity_info.read_text()))
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
