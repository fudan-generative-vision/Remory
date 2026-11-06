"""SGLang runtime for fixed-MASK token-embedding ELF checkpoints.

These checkpoints use the same token-embedding endpoint model as regular ELF,
but their rollout state is initialized with the normalized frozen target
``<|MASK|>`` embedding instead of Gaussian noise.  The buffer is saved in the
checkpoint so serving never needs a target-model forward or an RNG draw to
reconstruct it.
"""

from __future__ import annotations

import torch

from sglang.srt.models.dflow_token_embedding import DFlowTokenEmbeddingDraftModel


class DFlowTokenEmbeddingFixedMaskDraftModel(DFlowTokenEmbeddingDraftModel):
    """Token-embedding ELF with a persistent frozen-MASK initial latent."""

    fixed_mask_initial_state = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        raw = getattr(config, "dflow_config", None)
        if not isinstance(raw, dict):
            try:
                raw = dict(raw or {})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "fixed-MASK token-embedding ELF requires dflow_config"
                ) from exc

        if raw.get("initial_state_type") != "frozen_target_mask_embedding":
            raise ValueError(
                "fixed-MASK token-embedding ELF requires "
                "initial_state_type='frozen_target_mask_embedding'"
            )
        if raw.get("training_input_source") != "frozen_target_mask_embedding":
            raise ValueError(
                "fixed-MASK token-embedding ELF requires "
                "training_input_source='frozen_target_mask_embedding'"
            )
        if raw.get("serving_input_noise_active") is not False:
            raise ValueError(
                "fixed-MASK token-embedding ELF requires "
                "serving_input_noise_active=false"
            )
        if raw.get("source_mask_trainable") is not False:
            raise ValueError(
                "fixed-MASK token-embedding ELF source mask must be frozen"
            )

        mask_token_id = raw.get("mask_token_id")
        if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int):
            raise ValueError("fixed-MASK token-embedding ELF requires integer mask_token_id")
        if not 0 <= int(mask_token_id) < int(config.vocab_size):
            raise ValueError(
                f"mask_token_id={mask_token_id} is outside vocab_size={config.vocab_size}"
            )

        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.mask_token_id = int(mask_token_id)
        self.register_buffer(
            "fixed_mask_latent",
            torch.zeros(int(config.hidden_size), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fixed_mask_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    def make_fixed_mask_initial_state(
        self, anchor_norm: torch.Tensor, *, block_size: int
    ) -> torch.Tensor:
        """Return ``[verified anchor; frozen normalized MASK x (B - 1)]``."""

        if not bool(self.fixed_mask_initialized.item()):
            raise RuntimeError(
                "fixed-MASK token-embedding ELF checkpoint has an "
                "uninitialized fixed_mask_latent buffer"
            )
        if anchor_norm.ndim != 3 or int(anchor_norm.shape[1]) != 1:
            raise ValueError(
                "fixed-MASK initial state requires anchor_norm [batch, 1, hidden], "
                f"got {tuple(anchor_norm.shape)}"
            )
        width = int(block_size)
        if width < 1:
            raise ValueError(f"block_size must be positive, got {width}")
        mask = self.fixed_mask_latent.to(
            device=anchor_norm.device, dtype=anchor_norm.dtype
        ).view(1, 1, -1)
        return torch.cat(
            [anchor_norm, mask.expand(int(anchor_norm.shape[0]), width - 1, -1)],
            dim=1,
        )


EntryClass = DFlowTokenEmbeddingFixedMaskDraftModel
