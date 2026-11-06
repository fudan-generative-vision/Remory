"""Use the online-1228 summary protocol without training snapshots or branches."""
import hashlib
import json
from pathlib import Path
from context_compression.qwen38_online_protocol import (
    PROTOCOL, summary_suffix, summary_sampling, generate_summary,
)


def install(generator_class):
    original_init = generator_class.__init__
    original_decode = generator_class._decode

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._summary_suffix_ids = summary_suffix(self.tokenizer, self.summary_contract)

    def decode(self, ids, images, reserve, *, summary_request=False, post_compact_overhead=0):
        if not summary_request:
            return original_decode(self, ids, images, reserve, summary_request=False,
                                   post_compact_overhead=post_compact_overhead)
        suffix = self._summary_suffix_ids
        if list(ids[-len(suffix):]) != suffix:
            raise RuntimeError('summary suffix changed')
        identity = hashlib.sha256(json.dumps(dict(
            ids=list(ids), frontier=self._installed_identity,
            images=[image.image_data for image in images], protocol=PROTOCOL,
        ), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        run_root = Path(__file__).resolve().parents[2]
        episode = hashlib.sha256(str(self.episode_id).encode()).hexdigest()[:24]
        attempt_dir = run_root / 'summary-attempts' / episode / identity

        def request(attempt):
            from terminal_bench_eval.summary_policy import summary_output_budget
            self._ensure_open()
            request_ids = list(ids[:-len(suffix)]) + summary_suffix(self.tokenizer, self.summary_contract, attempt)
            self._summary_budget = summary_output_budget(
                input_tokens=len(request_ids) + reserve, server_info=self.server_info,
                post_compact_overhead=post_compact_overhead)
            kwargs = dict(max_new_tokens=self._summary_budget,
                          sampling_overrides=summary_sampling(self.summary_contract))
            if images:
                kwargs['image_data'] = [image.image_data for image in images]
            stream = getattr(self.client, 'set_generation_stream_context', None)
            if stream is not None:
                stream(phase='summary', episode_id=self.episode_id,
                       round_index=getattr(self, '_observed_round_index', None))
            if self._context_memory is None:
                return self.client.generate_discrete(request_ids, **kwargs)
            return self.client.generate_with_memory(
                request_ids, memory_positions=self.memory_positions,
                memory_embeddings=self._context_memory,
                expected_hidden_size=self.deployment['compressor']['target_hidden_size'], **kwargs)

        result, _, _ = generate_summary(request, self.tokenizer, self.summary_contract, attempt_dir)
        return result

    generator_class.__init__ = initialize
    generator_class._decode = decode
