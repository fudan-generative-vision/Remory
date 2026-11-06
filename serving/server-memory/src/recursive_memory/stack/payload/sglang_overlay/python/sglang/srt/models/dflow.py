"""Native SGLang runtime for SpecForge ``DFlowDraftModel`` checkpoints.

The transformer/KV-cache path is shared with DFlash, but DFlow has a distinct
training/serving ABI: normalized Gaussian latents, self-conditioning, timestep
and mode conditioning, plus a learned adapter before the target LM head.  The
worker prepares those dynamic inputs outside the CUDA graph and copies the
result into SGLang's existing draft ``input_embeds`` graph buffer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel


_MODE_TO_ID = {
    "denoise": 0,
    "decode": 1,
    "distill": 2,
    "acceptance": 3,
}

_VERIFIED_ROOT_BLOCK_SIZE = 16
_VERIFIED_ROOT_CONFIG_CONTRACT = {
    "one_step_self_cond": "verified_root",
    "verified_root_conditioning": "target_embedding_rms",
    "verified_root_projection": "self_cond_proj_second_half",
    "verified_root_init": "copy_w_z",
    "verified_root_broadcast_slots": _VERIFIED_ROOT_BLOCK_SIZE,
}
_VERIFIED_ROOT_CONTEXTUAL_X0_CONTRACT = {
    "clean_latent_source": "target_last_hidden",
    "clean_latent_alignment": "next_token_h[p-1]_to_y[p]",
    "serving_anchor_source": "last_target_feature_h[p-1]",
    "decode_adapter_type": "identity",
    "decode_linear_adapter": False,
    "decode_norm_trainable": False,
    "normalization_type": "scalar",
    "one_step_input_noise_scale": 0.0,
}


def _contract_value_equal(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, Integral)
            and not isinstance(actual, bool)
            and int(actual) == expected
        )
    if isinstance(expected, float):
        return (
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    return actual == expected


def _verified_root_conditioning_enabled(
    dflow_config: dict,
    *,
    block_size,
) -> bool:
    """Validate and resolve the explicit verified-root serving ABI.

    A legacy checkpoint is one that contains none of the four VRC-only keys
    and does not select ``one_step_self_cond=verified_root``.  Any partial VRC
    opt-in fails closed instead of silently taking the legacy zero-condition
    path.
    """

    vrc_only_keys = set(_VERIFIED_ROOT_CONFIG_CONTRACT) - {"one_step_self_cond"}
    claims_vrc = (
        dflow_config.get("one_step_self_cond") == "verified_root"
        or any(key in dflow_config for key in vrc_only_keys)
        or any(key.startswith("verified_root_") for key in dflow_config)
    )
    if not claims_vrc:
        return False

    mismatches = {
        key: (dflow_config.get(key), expected)
        for key, expected in _VERIFIED_ROOT_CONFIG_CONTRACT.items()
        if not _contract_value_equal(dflow_config.get(key), expected)
    }
    for key in dflow_config:
        if (
            key.startswith("verified_root_")
            and key not in _VERIFIED_ROOT_CONFIG_CONTRACT
        ):
            mismatches[key] = (dflow_config[key], "unsupported marker")
    if not _contract_value_equal(block_size, _VERIFIED_ROOT_BLOCK_SIZE):
        mismatches["block_size"] = (block_size, _VERIFIED_ROOT_BLOCK_SIZE)

    # VRC was trained only through the exact one-step folded path.  Requiring
    # these already-public markers here prevents a syntactically complete VRC
    # marker set from entering a dynamically conditioned or iterative path.
    required_base_contract = {
        **_VERIFIED_ROOT_CONTEXTUAL_X0_CONTRACT,
        "fixed_conditioning_folded": True,
        "one_step_direct": True,
        "one_step_input_t": 0.0,
        "one_step_mode": "distill",
    }
    for key, expected in required_base_contract.items():
        if not _contract_value_equal(dflow_config.get(key), expected):
            mismatches[key] = (dflow_config.get(key), expected)
    for key in ("progressive_layer_denoising", "layerwise_noise_conditioning"):
        actual = dflow_config.get(key, False)
        if not _contract_value_equal(actual, False):
            mismatches[key] = (actual, False)

    if mismatches:
        raise ValueError(
            "DFlow verified-root conditioning config contract mismatch "
            f"(actual, expected): {mismatches}."
        )
    return True


def _fixed_conditioning_folded_enabled(
    dflow_config: dict,
    *,
    verified_root_conditioning: bool = False,
) -> bool:
    """Validate the checkpoint opt-in for fixed one-step conditioning.

    Unmarked checkpoints keep the original per-token conditioning path.  A
    marked checkpoint may fold only the exact conditioning tuple used by the
    no-layer-noise one-step training contract.
    """

    marker = dflow_config.get("fixed_conditioning_folded", False)
    if not isinstance(marker, bool):
        raise ValueError(
            "DFlow dflow_config.fixed_conditioning_folded must be a JSON "
            f"boolean, got {marker!r}."
        )
    if not marker:
        return False

    input_t = dflow_config.get("one_step_input_t")
    input_t_is_zero = (
        isinstance(input_t, Real)
        and not isinstance(input_t, bool)
        and math.isfinite(float(input_t))
        and float(input_t) == 0.0
    )
    violations = []
    if dflow_config.get("one_step_direct") is not True:
        violations.append("one_step_direct=true")
    if dflow_config.get("progressive_layer_denoising", False) is not False:
        violations.append("progressive_layer_denoising=false")
    if dflow_config.get("layerwise_noise_conditioning", False) is not False:
        violations.append("layerwise_noise_conditioning=false")
    if not input_t_is_zero:
        violations.append("one_step_input_t=0.0")
    expected_self_cond = "verified_root" if verified_root_conditioning else "zero"
    if dflow_config.get("one_step_self_cond") != expected_self_cond:
        violations.append(f'one_step_self_cond="{expected_self_cond}"')
    if dflow_config.get("one_step_mode") != "distill":
        violations.append('one_step_mode="distill"')
    if violations:
        raise ValueError(
            "DFlow fixed_conditioning_folded checkpoint violates its exact "
            "serving contract; required "
            + ", ".join(violations)
            + "."
        )
    return True


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10_000
) -> torch.Tensor:
    """Match SpecForge's sinusoidal DFlow timestep embedding exactly."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DFlowDraftModel(DFlashDraftModel):
    """SGLang implementation of SpecForge's flow-conditioned DFlash drafter."""

    is_dflow_draft_model = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        dflow_config = getattr(config, "dflow_config", {}) or {}
        if not isinstance(dflow_config, dict):
            dflow_config = dict(dflow_config)
        # Validate the explicit VRC ABI before allocating the transformer.  A
        # partial or drifted marker set must never instantiate a legacy model.
        verified_root_conditioning = _verified_root_conditioning_enabled(
            dflow_config,
            block_size=getattr(config, "block_size", None),
        )
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.clean_latent_source = str(
            dflow_config.get("clean_latent_source", "token_embedding")
        )
        if self.clean_latent_source not in {
            "token_embedding",
            "target_last_hidden",
        }:
            raise ValueError(
                "Unsupported DFlow clean_latent_source="
                f"{self.clean_latent_source!r}."
            )

        # These layers are replicated on every TP rank, matching the full
        # hidden-state representation at the entrance to tensor-parallel blocks.
        self.self_cond_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=True),
        )
        self.mode_embed = nn.Embedding(len(_MODE_TO_ID), hidden_size)
        self.one_step_direct = bool(dflow_config.get("one_step_direct", False))
        self.progressive_layer_denoising = bool(
            dflow_config.get("progressive_layer_denoising", False)
        )

        self.layerwise_noise_conditioning = bool(
            dflow_config.get("layerwise_noise_conditioning", False)
        )
        self.verified_root_conditioning = verified_root_conditioning
        self.fixed_conditioning_folded = _fixed_conditioning_folded_enabled(
            dflow_config,
            verified_root_conditioning=self.verified_root_conditioning,
        )
        layerwise_denoise_t = dflow_config.get("layerwise_denoise_t")
        if getattr(self, "layerwise_noise_conditioning", False):
            num_layers = int(config.num_hidden_layers)
            if layerwise_denoise_t is None:
                layerwise_denoise_t = [
                    float(layer_idx + 1) / float(num_layers)
                    for layer_idx in range(num_layers)
                ]
            if len(layerwise_denoise_t) != num_layers:
                raise ValueError(
                    "layerwise_denoise_t must contain one value per draft layer: "
                    f"expected {num_layers}, got {len(layerwise_denoise_t)}"
                )
            layerwise_denoise_t = torch.as_tensor(
                layerwise_denoise_t, dtype=torch.float32
            )
            if not bool(
                torch.isfinite(layerwise_denoise_t).all()
                and (layerwise_denoise_t >= 0).all()
                and (layerwise_denoise_t <= 1).all()
            ):
                raise ValueError(
                    "layerwise_denoise_t values must be finite and in [0, 1]."
                )
            if not bool(
                (layerwise_denoise_t[1:] >= layerwise_denoise_t[:-1]).all()
            ):
                raise ValueError("layerwise_denoise_t must be non-decreasing.")
        else:
            layerwise_denoise_t = torch.empty(0, dtype=torch.float32)
        self.register_buffer(
            "layerwise_denoise_t",
            layerwise_denoise_t,
            persistent=True,
        )

        # DFlashDraftModel calls this final norm ``norm``.  Historical DFlow
        # checkpoints serialize the same tensor as ``out_norm.weight``;
        # load_weights maps the name below.  The continuous-time contextual
        # JiT checkpoints are different: their endpoint is already in the
        # target LM-head input space and explicitly owns an *identity* output
        # norm.  Keeping ``norm`` allocated preserves the parent layout, but
        # it must be bypassed (and must not be loaded/required) for that ABI.
        output_norm_value = dflow_config.get("output_norm_type", "rmsnorm")
        if not isinstance(output_norm_value, str):
            raise ValueError("dflow_config.output_norm_type must be a string.")
        self.output_norm_type = output_norm_value.lower()
        if self.output_norm_type in {"identity", "none", "no_adapter"}:
            self.output_norm_type = "identity"
            self.output_norm_identity = True
        elif self.output_norm_type == "rmsnorm":
            self.output_norm_identity = False
        else:
            raise ValueError(
                "Unsupported DFlow output_norm_type="
                f"{output_norm_value!r}; expected 'rmsnorm' or 'identity'."
            )
        self.out_norm = self.norm

        decode_adapter_type = dflow_config.get("decode_adapter_type", "rmsnorm")
        self.decode_adapter_type = decode_adapter_type
        if decode_adapter_type in (None, "none", "identity", "no_adapter"):
            self.decode_norm = nn.Identity()
        elif decode_adapter_type == "rmsnorm":
            self.decode_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        elif decode_adapter_type == "linear":
            self.decode_norm = nn.Linear(hidden_size, hidden_size, bias=True)
        else:
            raise ValueError(
                f"Unsupported DFlow decode_adapter_type={decode_adapter_type!r}."
            )
        if (
            self.clean_latent_source == "target_last_hidden"
            and not isinstance(self.decode_norm, nn.Identity)
        ):
            raise ValueError(
                "Contextual x0 is already post-final-norm; its DFlow decode "
                "adapter must be identity."
            )

        # Version-2 continuous-JiT checkpoints have no independently trained
        # decode pass.  Their final Euler endpoint is fed directly to the
        # frozen target LM head.  Store and validate this as a model-owned ABI
        # rather than letting the worker infer it from a loose combination of
        # flags; otherwise a generic DFlow checkpoint could silently lose or
        # gain a physical draft forward.
        terminal_mode_value = dflow_config.get(
            "serving_terminal_mode", "decode_forward"
        )
        if not isinstance(terminal_mode_value, str):
            raise ValueError(
                "dflow_config.serving_terminal_mode must be a string."
            )
        self.serving_terminal_mode = terminal_mode_value.lower()
        if self.serving_terminal_mode not in {"decode_forward", "direct_lm_head"}:
            raise ValueError(
                "Unsupported DFlow serving_terminal_mode="
                f"{terminal_mode_value!r}; expected 'decode_forward' or "
                "'direct_lm_head'."
            )

        serving_self_conditioning = dflow_config.get(
            "serving_self_conditioning", True
        )
        if not isinstance(serving_self_conditioning, bool):
            raise ValueError(
                "dflow_config.serving_self_conditioning must be a JSON boolean."
            )
        self.serving_self_conditioning = serving_self_conditioning

        serving_num_flow_steps = dflow_config.get("serving_num_flow_steps", 8)
        if type(serving_num_flow_steps) is not int or serving_num_flow_steps < 1:
            raise ValueError(
                "dflow_config.serving_num_flow_steps must be a positive JSON integer."
            )
        self.serving_num_flow_steps = serving_num_flow_steps

        self.serving_denoiser_noise_scale = float(
            dflow_config.get("serving_denoiser_noise_scale", 2.0)
        )
        self.serving_t_eps = float(dflow_config.get("serving_t_eps", 0.05))
        if (
            not math.isfinite(self.serving_denoiser_noise_scale)
            or self.serving_denoiser_noise_scale < 0.0
        ):
            raise ValueError(
                "dflow_config.serving_denoiser_noise_scale must be finite and "
                "non-negative."
            )
        if not math.isfinite(self.serving_t_eps) or self.serving_t_eps <= 0.0:
            raise ValueError(
                "dflow_config.serving_t_eps must be finite and positive."
            )

        if self.serving_terminal_mode == "direct_lm_head":
            if dflow_config.get("serving_contract_version") != 2:
                raise ValueError(
                    "direct_lm_head serving requires serving_contract_version=2."
                )
            if self.clean_latent_source != "target_last_hidden":
                raise ValueError(
                    "direct_lm_head serving requires "
                    "clean_latent_source='target_last_hidden'."
                )
            if not isinstance(self.decode_norm, nn.Identity):
                raise ValueError(
                    "direct_lm_head serving requires an identity decode adapter."
                )
            if not self.output_norm_identity:
                raise ValueError(
                    "direct_lm_head serving requires output_norm_type='identity'."
                )
            target_depth = int(getattr(config, "num_target_layers"))
            target_layer_ids = [
                int(layer_id) for layer_id in dflow_config.get("target_layer_ids", [])
            ]
            if not target_layer_ids or target_layer_ids[-1] != target_depth:
                raise ValueError(
                    "direct_lm_head contextual serving requires the post-final-norm "
                    f"target-depth sentinel {target_depth}, got {target_layer_ids}."
                )
            if dflow_config.get("training_objective") != (
                "continuous_t_jit_velocity_to_contextual_last_hidden"
            ):
                raise ValueError(
                    "direct_lm_head serving requires the matching contextual "
                    "continuous-t JiT training objective."
                )

            recorded_forwards = dflow_config.get("serving_drafter_forwards")
            if (
                type(recorded_forwards) is not int
                or recorded_forwards != self.serving_num_flow_steps
            ):
                raise ValueError(
                    "direct_lm_head serving_drafter_forwards must equal "
                    "serving_num_flow_steps; got "
                    f"{recorded_forwards!r} vs {self.serving_num_flow_steps}."
                )

        normalization_type = str(dflow_config.get("normalization_type", "scalar"))
        self.normalization_type = normalization_type
        mean_value = dflow_config.get("latent_mean", 0.0)
        std_value = dflow_config.get("latent_std", 1.0)
        mean = torch.as_tensor(mean_value, dtype=torch.float32)
        std = torch.as_tensor(std_value, dtype=torch.float32)
        if normalization_type == "scalar":
            if mean.numel() != 1 or std.numel() != 1:
                raise ValueError("DFlow scalar normalization requires scalar stats.")
            mean = mean.reshape(())
            std = std.reshape(())
        elif normalization_type == "channel":
            if mean.numel() != hidden_size or std.numel() != hidden_size:
                raise ValueError(
                    "DFlow channel normalization requires one value per hidden channel."
                )
            mean = mean.reshape(hidden_size)
            std = std.reshape(hidden_size)
        else:
            raise ValueError(
                f"Unsupported DFlow normalization_type={normalization_type!r}."
            )
        self.register_buffer("latent_mean", mean, persistent=True)
        self.register_buffer("latent_std", std.clamp_min(1e-6), persistent=True)

    def normalize_latent(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.to(device=value.device, dtype=value.dtype)
        std = self.latent_std.to(device=value.device, dtype=value.dtype)
        return (value - mean) / std.clamp_min(1e-6)

    def denormalize_latent(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.to(device=value.device, dtype=value.dtype)
        std = self.latent_std.to(device=value.device, dtype=value.dtype)
        return value * std.clamp_min(1e-6) + mean

    def prepare_flow_input(
        self,
        z_t: torch.Tensor,
        *,
        flow_t: float,
        flow_mode_id: int,
        self_cond: Optional[torch.Tensor] = None,
        verified_root_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply dynamic DFlow conditioning before CUDA-graph replay."""
        if z_t.ndim != 2 or int(z_t.shape[-1]) != int(self.config.hidden_size):
            raise ValueError(
                f"DFlow z_t must be [tokens, hidden], got {tuple(z_t.shape)}."
            )
        fixed_conditioning_folded = getattr(
            self, "fixed_conditioning_folded", False
        )
        if not isinstance(fixed_conditioning_folded, bool):
            raise RuntimeError(
                "DFlow fixed_conditioning_folded runtime marker must be boolean."
            )
        verified_root_conditioning = getattr(
            self, "verified_root_conditioning", False
        )
        if not isinstance(verified_root_conditioning, bool):
            raise RuntimeError(
                "DFlow verified_root_conditioning runtime marker must be boolean."
            )
        if verified_root_conditioning != (verified_root_embedding is not None):
            requirement = "requires" if verified_root_conditioning else "forbids"
            raise ValueError(
                f"This DFlow checkpoint {requirement} "
                "verified_root_embedding."
            )
        if fixed_conditioning_folded:
            expected_mode_id = _MODE_TO_ID["distill"]
            if self_cond is not None:
                raise ValueError(
                    "Folded DFlow conditioning requires self_cond=None; the "
                    "checkpoint contract supplies the fixed zero value."
                )
            if (
                not isinstance(flow_t, Real)
                or isinstance(flow_t, bool)
                or not math.isfinite(float(flow_t))
                or float(flow_t) != 0.0
            ):
                raise ValueError(
                    "Folded DFlow conditioning requires flow_t=0.0, got "
                    f"{flow_t!r}."
                )
            if (
                not isinstance(flow_mode_id, Integral)
                or isinstance(flow_mode_id, bool)
                or int(flow_mode_id) != expected_mode_id
            ):
                raise ValueError(
                    "Folded DFlow conditioning requires distill mode id "
                    f"{expected_mode_id}, got {flow_mode_id!r}."
                )

            compute_dtype = self.self_cond_proj.weight.dtype
            z_t = z_t.to(dtype=compute_dtype)
            hidden_size = int(self.config.hidden_size)
            if verified_root_conditioning:
                hidden_states = self._prepare_verified_root_input(
                    z_t,
                    verified_root_embedding,
                )
            else:
                # Keep the legacy folded checkpoint path byte-for-byte in the
                # same operation order when no VRC markers are present.
                hidden_states = F.linear(
                    z_t,
                    self.self_cond_proj.weight[:, :hidden_size],
                    self.self_cond_proj.bias,
                )
            # Fixed t/mode conditioning is identical for every token.  Compute
            # each vector once and rely on broadcast instead of materializing
            # token-sized condition tensors.
            t = torch.zeros((1,), dtype=torch.float32, device=z_t.device)
            time_cond = timestep_embedding(t, hidden_size).to(
                dtype=compute_dtype
            )
            time_cond = self.time_mlp(time_cond)
            mode_ids = torch.full(
                (1,),
                expected_mode_id,
                dtype=torch.long,
                device=z_t.device,
            )
            return hidden_states + time_cond + self.mode_embed(mode_ids)

        if self_cond is None:
            self_cond = torch.zeros_like(z_t)
        if self_cond.shape != z_t.shape:
            raise ValueError(
                "DFlow self_cond must match z_t, got "
                f"z_t={tuple(z_t.shape)} self_cond={tuple(self_cond.shape)}."
            )
        if not (0 <= int(flow_mode_id) < len(_MODE_TO_ID)):
            raise ValueError(f"Invalid DFlow mode id {flow_mode_id}.")

        compute_dtype = self.self_cond_proj.weight.dtype
        z_t = z_t.to(dtype=compute_dtype)
        if (
            getattr(self, "one_step_direct", False)
            and getattr(self, "progressive_layer_denoising", False)
        ):
            # The layer-progressive direct model starts from z0 itself.  Its
            # training forward explicitly bypasses self/time/mode
            # conditioning, so serving must do the same.
            return z_t

        # The no-layer-noise one-step snapshot predates the progressive
        # bypass: it was trained with self_cond=0, t=0 and mode="distill".
        # Keep that conditioning path active when one_step_direct=true but
        # progressive_layer_denoising is absent/false.
        self_cond = self_cond.to(dtype=compute_dtype)
        hidden_states = self.self_cond_proj(torch.cat([z_t, self_cond], dim=-1))

        t = torch.full(
            (int(z_t.shape[0]),),
            float(flow_t),
            dtype=torch.float32,
            device=z_t.device,
        )
        time_cond = timestep_embedding(t, int(self.config.hidden_size)).to(
            dtype=compute_dtype
        )
        time_cond = self.time_mlp(time_cond)
        mode_ids = torch.full(
            (int(z_t.shape[0]),),
            int(flow_mode_id),
            dtype=torch.long,
            device=z_t.device,
        )
        return hidden_states + time_cond + self.mode_embed(mode_ids)

    def _prepare_verified_root_input(
        self,
        z_t: torch.Tensor,
        verified_root_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``W_z z + b + W_root RMS(E(y_p))`` for one proposal call.

        ``z_t`` has sample-major layout ``[M, batch, 16, hidden]`` after
        reshaping.  The target embedding remains at base-batch width, so the
        reused ``W_root`` half is evaluated exactly once per base block.  Its
        projected code is then broadcast across both the sample and slot axes.
        """

        if verified_root_embedding is None:
            raise ValueError(
                "Verified-root DFlow input is missing verified_root_embedding."
            )
        if verified_root_embedding.requires_grad:
            raise ValueError(
                "Verified-root embeddings must be detached outputs of the "
                "frozen target embedding table."
            )
        if verified_root_embedding.ndim != 2:
            raise ValueError(
                "verified_root_embedding must have shape [batch, hidden], got "
                f"{tuple(verified_root_embedding.shape)}."
            )

        hidden_size = int(self.config.hidden_size)
        batch_size, root_hidden_size = verified_root_embedding.shape
        if int(root_hidden_size) != hidden_size:
            raise ValueError(
                "Verified-root embedding hidden size mismatch: expected "
                f"{hidden_size}, got {root_hidden_size}."
            )
        if batch_size <= 0:
            raise ValueError("Verified-root base batch must be positive.")
        runtime_block_size = getattr(
            self.config,
            "block_size",
            getattr(self, "block_size", None),
        )
        if runtime_block_size is None:
            raise RuntimeError("Verified-root runtime is missing block_size.")
        block_size = int(runtime_block_size)
        if block_size != _VERIFIED_ROOT_BLOCK_SIZE:
            raise RuntimeError(
                "Verified-root runtime block width drifted from 16: "
                f"got {block_size}."
            )
        tokens_per_sample = int(batch_size) * block_size
        if int(z_t.shape[0]) % tokens_per_sample != 0:
            raise ValueError(
                "Verified-root latent rows are not a complete sample-major "
                "[M,batch,16] layout: "
                f"rows={z_t.shape[0]}, batch={batch_size}."
            )
        noise_samples = int(z_t.shape[0]) // tokens_per_sample
        if noise_samples <= 0:
            raise ValueError("Verified-root noise-sample count must be positive.")

        compute_dtype = self.self_cond_proj.weight.dtype
        root_compute = verified_root_embedding.to(dtype=compute_dtype)
        # Match SpecForge exactly: the mean-square and rsqrt are FP32, use the
        # checkpoint RMS epsilon, then cast back before the learned projection.
        root_float = root_compute.float()
        root_normalized = root_float * torch.rsqrt(
            root_float.square().mean(dim=-1, keepdim=True)
            + float(self.config.rms_norm_eps)
        )
        root_normalized = root_normalized.to(dtype=compute_dtype)

        source_code = F.linear(
            z_t,
            self.self_cond_proj.weight[:, :hidden_size],
            self.self_cond_proj.bias,
        )
        root_code = F.linear(
            root_normalized,
            self.self_cond_proj.weight[:, hidden_size:],
            bias=None,
        )
        source_code = source_code.view(
            noise_samples,
            int(batch_size),
            block_size,
            hidden_size,
        )
        return (
            source_code + root_code.view(1, int(batch_size), 1, hidden_size)
        ).reshape(-1, hidden_size)

    def prepare_hidden_for_lm_head(self, x_pred_norm: torch.Tensor) -> torch.Tensor:
        """Match SpecForge ``decode_logits`` before using the target LM head."""
        hidden = self.denormalize_latent(x_pred_norm)
        return self.decode_norm(hidden)

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
        # ``input_embeds`` is already flow-conditioned by prepare_flow_input.
        if input_embeds is None:
            raise ValueError("DFlowDraftModel requires flow-conditioned input_embeds.")
        hidden_states = input_embeds
        residual = None

        layer_time_cond = None
        if self.layerwise_noise_conditioning:
            layer_t = self.layerwise_denoise_t.to(device=hidden_states.device)
            layer_time_cond = timestep_embedding(
                layer_t, int(self.config.hidden_size)
            ).to(dtype=hidden_states.dtype)
            layer_time_cond = self.time_mlp(layer_time_cond)

        for layer_idx, layer in enumerate(self.layers):
            if layer_time_cond is not None:
                # Adding to the MLP branch is equivalent to adding to the full
                # residual stream at the next fused pre-norm boundary.
                hidden_states = hidden_states + layer_time_cond[layer_idx]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if self.output_norm_identity:
                # The fused pre-norm layers hold the residual addition until
                # the final boundary.  This is the exact raw endpoint that
                # HF/SpecForge's `out_norm = Identity()` returns.
                if residual is not None:
                    hidden_states = hidden_states + residual
            elif residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load both the shared DFlash tower and DFlow-specific tensors."""
        params = dict(self.named_parameters())
        required_flow_params = {
            "self_cond_proj.weight",
            "self_cond_proj.bias",
            "time_mlp.0.weight",
            "time_mlp.0.bias",
            "time_mlp.2.weight",
            "time_mlp.2.bias",
            "mode_embed.weight",
            "decode_norm.weight",
            "norm.weight",
        }
        required_flow_buffers = {"latent_mean", "latent_std"}
        # Fixed-MASK token-embedding ELF checkpoints carry a persistent,
        # non-parameter initial latent.  Keep this generic loader aware of
        # the optional buffers so the subclass can reuse the exact streamed
        # DFlow weight-loading path without accepting an uninitialized source.
        if hasattr(self, "fixed_mask_latent"):
            required_flow_buffers.add("fixed_mask_latent")
        if hasattr(self, "fixed_mask_initialized"):
            required_flow_buffers.add("fixed_mask_initialized")
        if bool(getattr(self, "output_norm_identity", False)):
            # Raw token-embedding ELF endpoints deliberately have no final
            # RMSNorm.  The parent module still owns an unused ``norm``
            # parameter for layout compatibility, but the checkpoint must not
            # be required to provide (or silently initialize) it.
            required_flow_params.discard("norm.weight")
        if getattr(self, "layerwise_noise_conditioning", False):
            required_flow_buffers.add("layerwise_denoise_t")
        loaded: set[str] = set()

        for raw_name, loaded_weight in weights:
            name = (
                raw_name[len("model.") :] if raw_name.startswith("model.") else raw_name
            )

            # SpecForge keeps context_proj and the backwards-compatible fc alias
            # in the checkpoint.  The SGLang base model stores only fc.
            if name == "context_proj.weight":
                name = "fc.weight"
            elif name == "out_norm.weight":
                name = "norm.weight"

            if name in required_flow_buffers:
                target = getattr(self, name)
                target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
                loaded.add(name)
                continue

            if name in params and (
                name.startswith("self_cond_proj.")
                or name.startswith("time_mlp.")
                or name.startswith("mode_embed.")
                or name.startswith("decode_norm.")
                or name == "norm.weight"
            ):
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded.add(name)
                continue

            # DFlow partitions and streams base tensors through this loader one
            # at a time, then applies its own checkpoint-level contract below.
            base_loaded = super().load_weights(
                [(name, loaded_weight)], require_complete=False
            )
            loaded.update(base_loaded)

        missing = required_flow_params - loaded
        if isinstance(self.decode_norm, nn.Identity):
            missing.discard("decode_norm.weight")
        if missing:
            raise ValueError(
                "DFlow checkpoint is missing required runtime tensors: "
                f"{sorted(missing)}"
            )
        missing_buffers = required_flow_buffers - loaded
        if missing_buffers:
            raise ValueError(
                "DFlow checkpoint is missing required runtime buffers: "
                f"{sorted(missing_buffers)}"
            )
        return loaded


EntryClass = DFlowDraftModel
