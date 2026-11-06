"""SGLang runtime for SpecForge token-embedding ELF drafter checkpoints.

The endpoint of this model is a *normalized raw target input embedding*, not
an LM-head hidden state.  Consequently the final residual stream must remain
unnormalized, and proposal token IDs must be recovered by the worker's
checkpoint-selected token-embedding retrieval path.  The supported
train/serve ABIs are centered-L2 retrieval and frozen tied-input-embedding
dot-product retrieval.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dflow import DFlowDraftModel
from sglang.srt.models.dflow_token_embedding_contract import (
    validate_token_embedding_objective_abi,
)


_COMMON_CONTRACT = {
    "clean_latent_source": "token_embedding",
    "decode_adapter_type": "identity",
    "layerwise_noise_conditioning": False,
    "one_step_direct": False,
    "output_norm_type": "identity",
    "target_embedding_contract": "raw_tied_input_embedding",
    "token_embedding_method": "elf",
    "token_embedding_output": True,
}


class DFlowTokenEmbeddingDraftModel(DFlowDraftModel):
    """Six-layer DFlow whose x-prediction endpoint lives in embedding space."""

    is_dflow_token_embedding_model = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        raw = getattr(config, "dflow_config", None)
        if not isinstance(raw, dict):
            try:
                raw = dict(raw or {})
            except (TypeError, ValueError) as exc:
                raise ValueError("token-embedding ELF requires dflow_config") from exc

        mismatches = {
            key: (raw.get(key), expected)
            for key, expected in _COMMON_CONTRACT.items()
            if raw.get(key) != expected
        }
        try:
            retrieval_metric, _training_objective = (
                validate_token_embedding_objective_abi(raw)
            )
        except ValueError as exc:
            mismatches["objective_abi"] = (str(exc), "valid objective ABI")
        if str(raw.get("normalization_type", "")) != "scalar":
            mismatches["normalization_type"] = (
                raw.get("normalization_type"),
                "scalar",
            )
        valid_vocab_size = raw.get("valid_vocab_size")
        if (
            isinstance(valid_vocab_size, bool)
            or not isinstance(valid_vocab_size, int)
            or not 1 <= valid_vocab_size <= int(config.vocab_size)
        ):
            mismatches["valid_vocab_size"] = (
                valid_vocab_size,
                f"integer in [1, {int(config.vocab_size)}]",
            )
        if mismatches:
            raise ValueError(
                "Token-embedding ELF checkpoint contract mismatch "
                f"(actual, expected): {mismatches}."
            )

        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.output_norm_identity = True
        self.out_norm = nn.Identity()
        self.token_embedding_method = "elf"
        self.retrieval_metric = retrieval_metric
        self.retrieval_temperature = float(raw.get("retrieval_temperature", 1.0))
        self.valid_vocab_size = int(valid_vocab_size)
        self.serving_num_flow_steps = int(raw.get("serving_num_flow_steps", 8))
        self.serving_denoiser_noise_scale = float(
            raw.get("serving_denoiser_noise_scale", 2.0)
        )
        self.serving_t_eps = float(raw.get("serving_t_eps", 0.05))
        # Historical L2 checkpoints predate this explicit field and used
        # self-conditioning.  New discrete dot+CE checkpoints record it
        # explicitly, including the training-matched false setting.
        self.serving_self_conditioning = raw.get("serving_self_conditioning", True)

        if (
            not math.isfinite(self.retrieval_temperature)
            or self.retrieval_temperature <= 0.0
        ):
            raise ValueError("retrieval_temperature must be positive")
        if self.serving_num_flow_steps < 1:
            raise ValueError("serving_num_flow_steps must be positive")
        if self.serving_denoiser_noise_scale < 0.0:
            raise ValueError("serving_denoiser_noise_scale must be non-negative")
        if self.serving_t_eps <= 0.0:
            raise ValueError("serving_t_eps must be positive")
        if not isinstance(self.serving_self_conditioning, bool):
            raise ValueError("serving_self_conditioning must be a JSON boolean")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        """Return the raw final residual stream, matching ``out_norm=Identity``."""

        if input_embeds is None:
            raise ValueError(
                "DFlowTokenEmbeddingDraftModel requires flow-conditioned "
                "input_embeds."
            )
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0 and residual is not None:
            # SGLang's fused decoder layer carries the accumulated residual
            # separately.  The HF trainer performs this final addition and
            # then applies Identity; calling self.norm here changes embedding
            # norms and therefore changes nearest-neighbour token IDs.
            hidden_states = hidden_states + residual

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(weights)


EntryClass = DFlowTokenEmbeddingDraftModel
