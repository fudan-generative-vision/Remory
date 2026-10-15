"""Train a Qwen3.5-family r16 recursive compressor on regenerated traces.

This first implementation intentionally keeps the frozen target under native
Transformers TP2 and shards the randomly initialized DFlash compressor with
FSDP over the same two ranks.  Both ranks process the same example because
they jointly execute one tensor-parallel target model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, masking_utils
from transformers.integrations import sdpa_attention as transformers_sdpa_attention
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5_modeling
from transformers.models.qwen3_5_moe import (
    modeling_qwen3_5_moe as qwen3_5_moe_modeling,
)

from context_compression.grepseek_modeling import (
    DEFAULT_QWEN36_27B_TARGET_LAYER_IDS,
    DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS,
    GrepSeekDFlashCompressor,
)
from context_compression.losses import (
    frozen_lm_head_chunked_forward_kl,
    frozen_lm_head_chunked_token_ce,
)
from context_compression.mindforge_glm52_data import (
    MINDFORGE_CONTRACT,
    MINDFORGE_DATASET_FILE,
    MINDFORGE_DATASET_FILE_BYTES,
    MINDFORGE_DATASET_FILE_SHA256,
    MINDFORGE_DATASET_REPO,
    MINDFORGE_DATASET_REVISION,
    QWEN38_ENABLE_THINKING,
    QWEN38_PRESERVE_THINKING,
    MindForgeDecisionEvent,
    MindForgeRenderedEvent,
    render_mindforge_decision_event,
)
from context_compression.qwen35_text_target import (
    Qwen35TextTarget,
    load_qwen35_text_target,
    qwen35_modeling_module,
    qwen35_text_model_type,
)
from context_compression.qwen36_agentic_data import (
    QWEN36_PRESERVE_THINKING,
    QWEN36_TOOL_RESULT_BATCH_CLOSING_IDS,
    AgenticCompressionEvent,
    Qwen36AgenticExample,
    render_qwen36_agentic_example,
    slot_count_for_physical_tokens,
)
from context_compression.qwen36_agentic_data import (
    recurrent_slot_count as agentic_recurrent_slot_count,
)
from context_compression.qwen36_recursive_data import (
    COMPRESSION_RATIO,
    EVICTION_BLOCK_TOKENS,
    RAW_WORKING_MEMORY_TOKENS,
    Qwen36TokenClockExample,
    TokenClockEvent,
    build_qwen36_token_clock_example,
    recurrent_slot_counts,
    token_clock_events,
)
from context_compression.qwen36_teacher_parity import (
    PARITY_FORMAT_VERSION,
    PARITY_SCHEMA,
)
from context_compression.qwen36_teacher_parity import (
    cached_teacher_chunk_ranges as _cached_teacher_chunk_ranges,
)

FORMAT_VERSION = 2
SCHEMA_BY_TARGET_FAMILY = {
    "qwen36": "qwen36_27b_recursive_compressor_cached_teacher_v2",
    "qwen36_moe": "qwen36_35b_a3b_recursive_compressor_cached_teacher_v2",
    "qwen38": "qwen38_27b_qwen36_dflash_recursive_compressor_cached_teacher_v2",
}
LEGACY_TOKEN_CLOCK_DATA_CONTRACT = "legacy_token_clock"
QWEN36_AGENTIC_DATA_CONTRACT = "qwen36_agentic_tool_batches_v1"
MINDFORGE_FRONTIER_CE_DATA_CONTRACT = "mindforge_glm52_frontier_actions_v1"
QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT = "qwen38_vibe_wm4_action_kl_v1"
WM4_RESUME_BINDING_SCHEMA = "qwen38_vibe_wm4_resume_binding_v1"
# A WM4 checkpoint can be loaded only under the identical training semantic
# contract. Every run-contract key, including initialization and the entire
# data/model/implementation/optimizer/curriculum/selector/seed closure,
# remains exact. The one state-derived representation difference is explained
# next to the closed compressor architecture projection.
WM4_RESUME_COMPRESSOR_ARCHITECTURE_KEYS = frozenset(
    {
        "architecture",
        "target_layer_ids",
        "target_hidden_size",
        "selected_source_width",
        "compression_ratio",
        "local_window",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "layer_types",
        "gradient_checkpointing",
        "output_embedding_rms",
        "slot_padding_side",
    }
)
# ``compressor`` is included here only because its one state-derived exported
# buffer, ``output_embedding_rms``, changes from the portable BF16 value to
# the freshly built target FP32 value before the checkpoint restores that
# buffer.  The semantic projection below has a closed architecture projection
# which excludes only that buffer; arbitrary compressor changes are rejected.
WM4_RESUME_ALLOWED_RUN_CONTRACT_DELTA_KEYS = frozenset({"compressor"})
# These controls affect checkpoint placement or stop/save observation only.
# They are deliberately recorded and self-hashed separately from the semantic
# run contract so a WM4 A/B/C or formal mid-run resume cannot silently widen
# the set of allowed differences.
WM4_RESUME_OPERATIONAL_CONTROL_KEYS = frozenset(
    {
        "output_dir",
        "resume_from",
        "init_compressor",
        "max_steps",
        "save_interval",
        "skip_final_checkpoint",
        "log_interval",
        "verify_parameters_finite_after_step",
    }
)
QWEN36_AGENTIC_AUDIT_FORMAT_VERSION = 2
QWEN36_AGENTIC_AUDIT_SCHEMA = "qwen36_a3b_agentic_cpu_audit_v2"
DATA_CONTRACTS = (
    LEGACY_TOKEN_CLOCK_DATA_CONTRACT,
    QWEN36_AGENTIC_DATA_CONTRACT,
    MINDFORGE_FRONTIER_CE_DATA_CONTRACT,
    QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT,
)
OBJECTIVES = ("kl", "ce", "ce_kl")
OPTIMIZER_MEMORY_MODES = ("default", "low_peak")
WM4_LOW_PEAK_ADAM_MOMENT_RESIDENCY = (
    "cpu_between_steps_restore_after_activation_release_v1"
)
WM4_FORWARD_ENTRY_CACHE_TRIM = (
    "synchronize_then_empty_cache_before_each_action_v1"
)
WM4_ADAMW_MOMENT_STATE_KEYS = frozenset({"step", "exp_avg", "exp_avg_sq"})
QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION = "qwen38_step1000_to_progressive_v1"
QWEN38_PROGRESSIVE_RECURSIVE_EVENT_CAPS: tuple[int | None, ...] = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    None,
)
QWEN38_STEP1000_SOURCE_CONTRACT = {
    "run_contract_sha256": (
        "05a840cee1559bb769f49ce97b9432dee73fb3a7e818ead4864d781703beec44"
    ),
    "format_version": 2,
    "schema": "qwen38_27b_qwen36_dflash_recursive_compressor_cached_teacher_v2",
    "world_size": 2,
    "global_step": 1_000,
    "next_epoch": 0,
    "next_line_index": 24_612,
    "data_sha256": ("41ce010d490dc16811e2587802eae4aa5d4ea1f707e022bade780aecddcace3c"),
    "data_size": 516_420_430,
    "run_config_file_sha256": (
        "d3dbe357c99bc824a741712dc1c77295f88c2513debc6f8ba0d860f84071ace9"
    ),
    "trainer_state_file_sha256": (
        "287a2f0146855c58b22ab3ebbedd14e9f531ef507b1d2697c98bf2a781943308"
    ),
    "rank_file_sha256": {
        "rank-00000.pt": (
            "ba05c2ad478f56752c9c929192c0ed51cce442abe9c7a3cd08242e65c25e5007"
        ),
        "rank-00001.pt": (
            "29d4457350c58ec41f34fac174f3f9e730d6e3b4c42be74551cfdb52a5bb4b63"
        ),
    },
    "implementation_local_file_sha256": {
        "compressor": (
            "b1b4541e1dea8a8cbc632865831173e3795c86304ab05d711e77e2b8efa6cfdc"
        ),
        "data": ("7f63452b3c25e33e355023aea277001791092c8a89944771afcb07fecb11098c"),
        "loss": ("21e91ea420f842c9ecd751c5461efe8d244afa96c6e5de548a6c170142ac743f"),
        "parity_validator": (
            "0edc15c72e2e13ac0655b1b834a0f413e7ca8f7a075dc7712f33f1de440d4ad7"
        ),
        "trainer": ("1b796a0c8d288a686303ff3b278ca3cc228f189933c07c43a789304fe39b4b79"),
    },
    "implementation_identity_sha256": (
        "a01ee72b1484d4361a84cec84e1409a000b8cd6ac8dc6e82b35fb9222ccfbbbb"
    ),
}
QWEN38_STEP1000_SOURCE_RUNTIME = {
    "python": "3.11.11",
    "qwen35_cached_full_attention_patch": (
        "torch_causal_lower_right_native_gqa_registry_sdpa_v2"
    ),
    "packages": {
        "causal-conv1d": "1.6.2.post1",
        "flash-linear-attention": "0.5.1",
        "torch": "2.9.1",
        "transformers": "5.3.0",
        "triton": "3.5.1",
    },
    "torch_build": {
        "cuda": "12.8",
        "cudnn": 91002,
        "device_capability": [9, 0],
        "device_name": "NVIDIA H800",
        "git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
        "nccl": [2, 27, 5],
    },
}
QWEN38_STEP1000_SOURCE_DEPENDENCY_FILE_SHA256 = {
    "causal_conv1d_cuda": (
        "8b0b86808daa02e2ee90383bac54700e8e0b12d294b2fc10221e8821f1cb79ae"
    ),
    "causal_conv1d_interface": (
        "221f41aa963a8f84683d9c6b2b4bf448692feb963fefcd778fa1b4ed2688c859"
    ),
    "fla_common_chunk_delta_h": (
        "0100ab70fba8ff977935688e2bb8713da56906970870aaed471f95ec04735e6f"
    ),
    "fla_gated_delta_chunk": (
        "4f5ece54e49e843274396a709d98a604fdea5ec9820417b63544c9cf2d0d8052"
    ),
    "fla_gated_delta_chunk_fwd": (
        "109bd5ec13c822a628777479e79ff36e4a3571da265f64406f61e8f64b5e89d4"
    ),
    "fla_gated_delta_fused_recurrent": (
        "9c15dc68d37a150da5317431a51fe7def362264ac92770e32261c8557cd5889b"
    ),
    "torch_attention_bias": (
        "b17a2bf96c9cdee3ef7e29896e294193253da46c947422c4dead9e7ac49c943c"
    ),
    "torch_c_extension": (
        "4349b125953ce9e0f7e0e94f5a78519107f15a234e1fb3703e86c12c405b09cd"
    ),
    "torch_init": ("3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288"),
    "transformers_cache_utils": (
        "e4fe20ca1eeb17ea0ce7e62d1dbd0d6eab32e7072044f293612403b8e0c477fd"
    ),
    "transformers_masking_utils": (
        "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7"
    ),
    "transformers_qwen3_5": (
        "0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a"
    ),
    "transformers_sdpa_attention": (
        "87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564"
    ),
}
QWEN38_PROGRESSIVE_LINEAGE_SCHEMA = "qwen38_step1000_progressive_recursive_lineage_v1"
QWEN38_PROGRESSIVE_DESTINATION_LOCAL_FILE_SHA256 = {
    "compressor": "49467f741a6fe20be20bc45af26d7545d8e9fac3e6f62e445734366fafdc6aee",
    "data": "7f63452b3c25e33e355023aea277001791092c8a89944771afcb07fecb11098c",
    "loss": "5620bbb4b4464558b56dba76a66472493a5c34988742749f2342feaf91a52cca",
    "parity_validator": (
        "0edc15c72e2e13ac0655b1b834a0f413e7ca8f7a075dc7712f33f1de440d4ad7"
    ),
    "target_loader": (
        "53c01efa121933e57fa6bf21e73a46c63bd9edc1e54186a42b7d2ef9993423b7"
    ),
}
QWEN38_PROGRESSIVE_ADDED_DEPENDENCY_FILE_SHA256 = {
    "transformers_qwen3_5_moe": (
        "1096a45073dad015f521a86890d491d1a4c655e7b98a7a56139493fa6edf3ac4"
    ),
}
QWEN38_PROGRESSIVE_SEMANTIC_AUDIT = {
    "schema": "qwen38_progressive_semantic_ast_audit_v2",
    "ast_python": "3.11.11",
    "legacy_snapshot": {
        "compressor_sha256": (
            "b1b4541e1dea8a8cbc632865831173e3795c86304ab05d711e77e2b8efa6cfdc"
        ),
        "loss_sha256": (
            "21e91ea420f842c9ecd751c5461efe8d244afa96c6e5de548a6c170142ac743f"
        ),
    },
    "unchanged_qwen38_ast_sha256": {
        "loss.frozen_lm_head_chunked_forward_kl": (
            "826d5f81ec532633cf3588639b0f25c0fe371ab9a295e2d81d06a61ca129e899"
        ),
        "model.build_qwen36_27b_dflash_config": (
            "4bf866c1ba1177ff9daf69caa95bc3abdda6da142539a549d786855c869376eb"
        ),
        "model.extract_dflash_hidden_features": (
            "b9aa7763751174b3084861863ea184234cb34f5b192d9286ffb769ca9bd4f1ba"
        ),
        "model.build_local_compression_layout": (
            "f222d4bb0011789c441fbde3b476f5c78d96af74059dded69acbcc100f9d9c81"
        ),
        "model.GrepSeekDFlashAttention": (
            "ac1d9571ba0120bd0e420cd3b8516ea763aa051f8bc2600661407cf26142ab2f"
        ),
        "model.GrepSeekDFlashDecoderLayer": (
            "79772ff476cb8c2b85e0dd7a34c99403da42a39f865d5d49e9f1e214afa278aa"
        ),
        "model.GrepSeekDFlashCompressor.__init__": (
            "ae35500cf316bf4709ae28fbbda4ff7c977624f9fd49ad00a96ac00867f9edd0"
        ),
        "model.GrepSeekDFlashCompressor.forward": (
            "c4801c6df3817f1d85b51f5eddc778a682cd3730443e8da5a1db4811a17c1cb1"
        ),
        "model.GrepSeekDFlashCompressor.export_config": (
            "53513f425a96047f4efbf273b22bb2b46406c50de5c100d629452e0fdf70feb0"
        ),
    },
    "allowed_non_qwen38_file_deltas": [
        "add Qwen3.6-35B-A3B compressor geometry without changing Qwen3.8 paths",
        "activation-checkpoint token-CE only; the KL objective AST is unchanged",
    ],
    "authorized_target_loader_refactor": {
        "classification": "previously_unreceipted_loader_now_exactly_pinned",
        "source_limitation": (
            "the step-1000 receipt pins the full legacy trainer and the dense "
            "Transformers modeling file, but does not separately pin a target-loader "
            "module; no byte-identical step-1000 trainer source was recoverable"
        ),
        "nearest_frozen_predecessor": {
            "date": "2026-08-20",
            "trainer_sha256": (
                "8bc7e7531a26155dab64479aa3c82924e421210afc2f85be5fd483297882bbcd"
            ),
            "dense_load": "Qwen3_5ForCausalLM.from_pretrained",
            "not_claimed_as_step1000_source": True,
        },
        "frozen_target": {
            "target_content_sha256": (
                "70722571062ccd3efa3c482ffe5ea15ae95329080ccce1f257c34e62e8b54074"
            ),
            "config_sha256": (
                "e2b9a2c27711af3dad9401e03ac8190cc1ccfa864f6051644a21162bb4e6bad7"
            ),
            "model_type": "qwen3_5_text",
            "architecture": "Qwen3_5ForCausalLM",
            "transformers_modeling_sha256": (
                "0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a"
            ),
        },
        "destination_loader_sha256": (
            "53c01efa121933e57fa6bf21e73a46c63bd9edc1e54186a42b7d2ef9993423b7"
        ),
        "dense_dispatch_contract": {
            "class": "Qwen3_5ForCausalLM",
            "local_files_only": True,
            "callsite_kwargs": {
                "attn_implementation": "sdpa",
                "dtype": "bfloat16",
                "tp_plan": "auto",
            },
            "ast_sha256": {
                "TEXT_TARGET_CLASS_BY_MODEL_TYPE": (
                    "50c1dd4fc10893b29440905f8c42b1fe5f40fb0046e721e012f405b5f9e24469"
                ),
                "qwen35_text_config": (
                    "fa42d2fa852a8ef231aebabdbe69d0ea3f922e21d7b0dedf4e7d45e3c6d42c23"
                ),
                "qwen35_text_model_type": (
                    "25b3e89fcb81ce01bf9775ba8443ca24bd8145733d49b5623f591ed1245ae7e7"
                ),
                "qwen35_text_target_class": (
                    "160c3a9564d68853fdc4a067eb6c88328fa1621c385e62244f0c433083c015eb"
                ),
                "load_qwen35_text_target": (
                    "299aeb5cd4bb5f154e3fa1df54297d6825604bad46d15ed0624a540a69977de4"
                ),
            },
        },
        "verification": (
            "CPU contract test intercepts Qwen3_5ForCausalLM.from_pretrained and "
            "proves the wrapper forwards the same target path and callsite kwargs, "
            "adding only local_files_only=true; the wrapper performs no numerical "
            "operation of its own"
        ),
    },
}


def _checkpoint_schema_for_data_contract(
    target_family: str,
    data_contract: str,
) -> str:
    """Keep agentic checkpoints impossible to resume under the legacy ABI."""

    base = SCHEMA_BY_TARGET_FAMILY[target_family]
    if data_contract == LEGACY_TOKEN_CLOCK_DATA_CONTRACT:
        return base
    if data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
        return f"{base}__qwen36_agentic_tool_batches_v1"
    if data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
        return f"{base}__mindforge_glm52_frontier_actions_v1"
    if data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        return f"{base}__qwen38_vibe_wm4_action_kl_v1"
    raise ValueError(f"unsupported data contract: {data_contract}")


QWEN35_CACHED_FULL_ATTENTION_PATCH = (
    "torch_causal_lower_right_native_gqa_registry_sdpa_v2"
)
QWEN38_MONOLITHIC_12K_PRODUCTION_CONTRACT = {
    "teacher_prefill_chunk_tokens": 12_288,
    "kl_tokens": 256,
    "lm_head_chunk_size": 16,
    "max_target_sequence_tokens": 12_288,
    "max_teacher_sequence_tokens": 12_288,
    "max_history_tokens": 262_144,
}
MINDFORGE_PRIMARY_H8192_PROFILE = "primary-h8192-v1"
MINDFORGE_FULL_VIEW_H4096_PROFILE = "full-view-h4096-v1"
MINDFORGE_PILOT_PROFILES = (
    MINDFORGE_PRIMARY_H8192_PROFILE,
    MINDFORGE_FULL_VIEW_H4096_PROFILE,
)
MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT = {
    "history_window_tokens": 8_192,
    "max_action_tokens": 1_024,
    "max_target_sequence_tokens": 12_288,
    "max_teacher_sequence_tokens": 12_288,
    "teacher_prefill_chunk_tokens": 12_288,
    "lm_head_chunk_size": 16,
    "compression_ratio": 16,
    "eviction_block_tokens": 1_024,
    "final_working_memory_tokens": 1_024,
}
QWEN38_VIBE_WM4_ACTION_KL_CONTRACT = {
    "working_memory_rounds": 4,
    "epochs": 10,
    "target_tokens": 256,
    "max_target_sequence_tokens": 12_288,
    "max_teacher_sequence_tokens": 12_288,
    "teacher_prefill_chunk_tokens": 12_288,
    "lm_head_chunk_size": 16,
    "max_history_tokens": 262_144,
    "compression_ratio": 16,
    "eviction_block_tokens": 1_024,
}
MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT_H4096 = {
    **MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT,
    "history_window_tokens": 4_096,
}
MINDFORGE_DECISION_MANIFEST_SCHEMA = "mindforge_glm52_qwen38_ce_decision_manifest_v1"
MINDFORGE_DECISION_MANIFEST_VIEW_SCHEMA = (
    "mindforge_glm52_qwen38_ce_decision_manifest_view_v1"
)
MINDFORGE_DECISION_MANIFEST_FORMAT_VERSION = 1
MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256 = (
    "352df26d29e0ee085cb4ee7d5142a674f0dd4b757c2687196500b9f83e93d832"
)
MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256 = (
    "e80f356308f663602e1ff55436ba3a1421b9d0a451d9fc57147d290fb1fcb2d3"
)
MINDFORGE_SELECTED_ACTIONS_PER_TRAJECTORY = 4
MINDFORGE_PRIMARY_TRAIN_TRAJECTORIES = 512
MINDFORGE_PRIMARY_VALIDATION_TRAJECTORIES = 16
MINDFORGE_PRIMARY_VALIDATION_ACTIONS_PER_TRAJECTORY = 2
MINDFORGE_RUNTIME_ABI = {
    "torch_public_version": "2.9.1",
    "transformers_version": "5.3.0",
}
QWEN36_MOE_MONOLITHIC_12K_PRODUCTION_CONTRACT = {
    "teacher_prefill_chunk_tokens": 12_288,
    "lm_head_chunk_size": 16,
    "max_target_sequence_tokens": 12_288,
    "max_teacher_sequence_tokens": 12_288,
    "max_history_tokens": 262_144,
}
QWEN36_35B_A3B_DFLASH_UPSTREAM_RECEIPT = {
    "config_sha256": (
        "20784b063040f7871efec4bf63342e5cb941907925943540176dfbb8aaf186dc"
    ),
    "geometry": {
        "block_size": 16,
        "bos_token_id": None,
        "eos_token_id": 248044,
        "head_dim": 128,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "mask_token_id": 248077,
        "max_position_embeddings": 262144,
        "max_window_layers": 6,
        "num_attention_heads": 32,
        "num_hidden_layers": 6,
        "num_key_value_heads": 8,
        "num_target_layers": 40,
        "pad_token_id": None,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        },
        "sliding_window": 4096,
        "target_layer_ids": list(DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS),
        "vocab_size": 248320,
    },
    "repository": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "revision": "0ce151d016ba8f0f0454d160a81b52ad4588c924",
}
QWEN36_35B_A3B_DFLASH_UPSTREAM_RECEIPT_SHA256 = (
    "12af37596c6c784f2d4b2073dbc050412b6667f2ed5f768f2eaf811bf6238c37"
)


def _target_family_from_manifest(target_model: Path) -> str:
    """Derive the target family from the immutable text-snapshot manifest."""

    manifest_path = target_model.expanduser().resolve() / "text_snapshot_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid target text-snapshot manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("target text-snapshot manifest must be a JSON object")
    schema = manifest.get("schema")
    source_name = Path(str(manifest.get("source", ""))).name
    derived_model_type = qwen35_text_model_type(target_model)
    if (
        schema
        in {
            "qwen36_text_only_symlink_snapshot",
            "qwen3_5_text_only_symlink_snapshot",
        }
        and derived_model_type == "qwen3_5_text"
        and source_name.startswith("Qwen3.6-27B")
    ):
        return "qwen36"
    if (
        schema == "qwen3_5_moe_text_only_symlink_snapshot"
        and derived_model_type == "qwen3_5_moe_text"
        and manifest.get("source_text_model_type") == "qwen3_5_moe_text"
        and manifest.get("derived_model_type") == "qwen3_5_moe_text"
        and manifest.get("derived_architecture") == "Qwen3_5MoeForCausalLM"
        and source_name.startswith("Qwen3.6-35B-A3B")
    ):
        return "qwen36_moe"
    if (
        schema == "qwen3_5_text_only_symlink_snapshot"
        and derived_model_type == "qwen3_5_text"
        and source_name.startswith("Qwen3.8-27B")
    ):
        return "qwen38"
    raise ValueError(
        "unsupported or ambiguous target text-snapshot family: "
        f"schema={schema!r}, source={source_name!r}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument(
        "--target-family",
        choices=tuple(SCHEMA_BY_TARGET_FAMILY),
        default="qwen36",
        help=(
            "Frozen target/template family. qwen36_moe selects the 35B-A3B "
            "text target; qwen38 intentionally retains the randomly initialized "
            "Qwen3.6-DFlash compressor geometry."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--data-contract",
        choices=DATA_CONTRACTS,
        default=LEGACY_TOKEN_CLOCK_DATA_CONTRACT,
        help=(
            "Input and compression-event ABI. The legacy contract reads JSONL "
            "and uses the 1024-token clock; qwen36_agentic_tool_batches_v1 "
            "reads the pinned A3B tool-calling parquet and supervises only its "
            "final chosen assistant continuation; qwen38_vibe_wm4_action_kl_v1 "
            "reads a frozen accepted coding-trajectory snapshot and matches the "
            "serving system+user+soft-memory+last-four-rounds boundary."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="kl",
        help=(
            "Training loss. The default kl path preserves the existing online "
            "raw-teacher objective. ce and ce_kl are reserved for the pinned "
            "MindForge frontier-action contract; its kl arm uses the same gold "
            "action and selected states as the CE arms."
        ),
    )
    parser.add_argument(
        "--kl-loss-weight",
        type=float,
        default=0.05,
        help="KL coefficient in objective=ce_kl; CE always has coefficient one.",
    )
    parser.add_argument(
        "--mindforge-event-manifest",
        type=Path,
        help=(
            "Frozen, audited set of late MindForge assistant cutpoints. MindForge "
            "training refuses to scan or supervise unlisted actions."
        ),
    )
    parser.add_argument(
        "--mindforge-pilot-profile",
        choices=MINDFORGE_PILOT_PROFILES,
        default=MINDFORGE_PRIMARY_H8192_PROFILE,
        help=(
            "Explicit frozen MindForge training lineage. The default preserves "
            "the primary 8K manifest; the 4K profile accepts only the pinned full "
            "history-window VIEW artifact."
        ),
    )
    parser.add_argument(
        "--expected-mindforge-manifest-sha256",
        help="Mandatory byte SHA256 pin for the production MindForge manifest.",
    )
    parser.add_argument(
        "--expected-mindforge-manifest-contract-sha256",
        help="Mandatory canonical contract SHA256 pin inside that manifest.",
    )
    parser.add_argument(
        "--mindforge-history-window-tokens",
        type=int,
        default=8192,
        help=(
            "Exact raw transcript-token suffix replayed for every selected "
            "MindForge action. The selected pilot profile fixes this width; it is "
            "not the "
            "trajectory's full raw history."
        ),
    )
    parser.add_argument(
        "--mindforge-max-action-tokens",
        type=int,
        default=1024,
        help=(
            "Maximum complete native assistant action. Longer actions are "
            "rejected whole; tool-call tokens are never truncated."
        ),
    )
    parser.add_argument(
        "--source-dataset-id",
        help="Immutable source dataset repository recorded for parquet provenance.",
    )
    parser.add_argument(
        "--source-dataset-revision",
        help="Full immutable source dataset commit recorded for parquet provenance.",
    )
    parser.add_argument(
        "--agentic-audit-artifact",
        "--agentic-data-audit",
        dest="agentic_data_audit",
        type=Path,
        help="Merged full-parquet CPU audit required by the agentic contract.",
    )
    parser.add_argument("--expected-agentic-total-rows", type=int)
    parser.add_argument("--expected-agentic-valid-rows", type=int)
    parser.add_argument("--expected-agentic-rejected-rows", type=int)
    parser.add_argument(
        "--expected-agentic-reject-category",
        action="append",
        default=[],
        help="Pinned NAME=COUNT reject category; repeat for every expected category.",
    )
    parser.add_argument("--expected-agentic-reject-rows-sha256")
    parser.add_argument("--expected-agentic-audit-sha256")
    parser.add_argument("--expected-agentic-audit-contract-sha256")
    parser.add_argument("--expected-agentic-12288-cap-stats-sha256")
    parser.add_argument(
        "--expected-data-sha256",
        help="Exact raw SHA-256 of the --data file bytes for every data contract.",
    )
    parser.add_argument(
        "--expected-data-manifest-canonical-sha256",
        help=(
            "Exact canonical JSON SHA-256 of a Qwen3.8 WM4 frozen data "
            "manifest; this is distinct from --expected-data-sha256."
        ),
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="CPU-only exact input-manifest validation; never initializes CUDA.",
    )
    parser.add_argument(
        "--vibe-working-memory-rounds",
        type=int,
        default=4,
        help=(
            "Exact count of complete normalized coding rounds retained raw by the "
            "Qwen3.8 vibe serving-state contract."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--curriculum-mode",
        choices=("auto", "one_shot", "recursive"),
        default="auto",
    )
    parser.add_argument(
        "--recursive-event-caps",
        help=(
            "Comma-separated maximum chronological recursive events for each "
            "epoch after epoch 1; use 'all' for no curriculum cap."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--optimizer-memory-mode",
        choices=OPTIMIZER_MEMORY_MODES,
        default="default",
        help=(
            "Optimizer allocation policy. low_peak keeps AdamW semantics while "
            "disabling foreach temporaries and releasing unused CUDA cache at "
            "each optimizer boundary."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--kl-tokens", type=int, default=256)
    parser.add_argument("--lm-head-chunk-size", type=int, default=16)
    parser.add_argument("--max-target-sequence-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-teacher-sequence-tokens",
        type=int,
        default=262144,
        help="Maximum complete raw-teacher context, capped by the target model.",
    )
    parser.add_argument(
        "--teacher-prefill-chunk-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum tokens in one no-grad teacher forward; in monolithic-only "
            "mode this must equal the teacher sequence hard cap."
        ),
    )
    parser.add_argument(
        "--teacher-execution-mode",
        choices=("cached", "monolithic_only"),
        default="cached",
        help=(
            "Teacher forward contract. monolithic_only disables cache and drops "
            "events above the configured teacher hard cap."
        ),
    )
    parser.add_argument("--max-history-tokens", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-transition",
        choices=(QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION,),
        help=(
            "Explicit, fail-closed run-contract transition used only to fork the "
            "pinned Qwen3.8 step-1000 checkpoint into progressive recursion."
        ),
    )
    parser.add_argument(
        "--expected-progressive-trainer-sha256",
        help=(
            "Exact SHA256 of this trainer source, required and contract-bound for "
            "the Qwen3.8 progressive transition and every subsequent resume."
        ),
    )
    parser.add_argument(
        "--expected-vibe-trainer-sha256",
        help=(
            "Mandatory exact trainer source SHA256 for the Qwen3.8 vibe action-KL "
            "contract."
        ),
    )
    parser.add_argument(
        "--init-compressor",
        type=Path,
        help=(
            "Portable compressor.pt loaded strictly as model weights only. A new "
            "optimizer, scheduler, RNG stream, and step counter are always created."
        ),
    )
    parser.add_argument(
        "--resume-at-recursive-epoch",
        action="store_true",
        help=(
            "Restore model/optimizer/RNG from an epoch-1 checkpoint but intentionally "
            "skip the remaining one-shot rows and start epoch 2 at line 0."
        ),
    )
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument(
        "--verify-parameters-finite-after-step",
        action="store_true",
        help=(
            "Canary-only: after the final controlled optimizer update, prove all "
            "FSDP parameter shards finite across ranks before any final save."
        ),
    )
    parser.add_argument("--disable-target-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--disable-compressor-gradient-checkpointing", action="store_true"
    )
    parser.add_argument(
        "--verify-cached-teacher",
        action="store_true",
        help="Smoke-only: compare cached and monolithic teacher outputs.",
    )
    parser.add_argument(
        "--cached-teacher-max-output-kl",
        type=float,
        default=1e-4,
        help="Maximum KL(monolithic teacher || cached teacher) per token.",
    )
    parser.add_argument(
        "--cached-teacher-min-top1-agreement",
        type=float,
        default=1.0,
        help="Minimum cached/monolithic teacher top-1 agreement.",
    )
    parser.add_argument(
        "--fixed-probe-line-index",
        "--parity-line-index",
        dest="parity_line_index",
        type=int,
        help="Smoke-only zero-based source record selected for a fixed probe.",
    )
    parser.add_argument(
        "--fixed-probe-event-index",
        "--parity-event-index",
        dest="parity_event_index",
        type=int,
        help="Smoke-only final recursive event; causal predecessors run first.",
    )
    parser.add_argument(
        "--fixed-probe-expected-example-id",
        "--parity-expected-example-id",
        dest="parity_expected_example_id",
    )
    parser.add_argument(
        "--fixed-probe-expected-row-sha256",
        help=(
            "Canonical source-row SHA256 required by an agentic parquet " "fixed probe."
        ),
    )
    parser.add_argument(
        "--fixed-probe-expected-teacher-tokens",
        "--parity-expected-teacher-tokens",
        dest="parity_expected_teacher_tokens",
        type=int,
    )
    parser.add_argument(
        "--fixed-probe-expected-student-tokens",
        "--parity-expected-student-tokens",
        dest="parity_expected_student_tokens",
        type=int,
    )
    parser.add_argument(
        "--fixed-probe-expected-cached-chunks",
        "--parity-expected-cached-chunks",
        dest="parity_expected_cached_chunks",
        type=int,
    )
    parser.add_argument(
        "--cached-teacher-parity-artifact",
        type=Path,
        action="append",
        default=[],
        help="Validated smoke evidence required by a formal cached-teacher run.",
    )
    parser.add_argument(
        "--required-parity-chunk-tokens",
        type=int,
        action="append",
        default=[],
    )
    return parser


def _parse_expected_reject_categories(values: Sequence[str]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for value in values:
        name, separator, count_text = value.partition("=")
        if (
            not separator
            or not name.startswith("reject_")
            or not count_text.isdigit()
            or name in categories
        ):
            raise ValueError(
                "expected agentic reject categories must be unique NAME=COUNT values"
            )
        categories[name] = int(count_text)
    return categories


def _objective_uses_raw_teacher(objective: str) -> bool:
    if objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    return objective in {"kl", "ce_kl"}


def _objective_uses_ce(objective: str) -> bool:
    if objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    return objective in {"ce", "ce_kl"}


def _parse_recursive_event_caps(value: str | None) -> tuple[int | None, ...]:
    if value is None:
        return ()
    fields = value.split(",")
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("recursive-event-caps must be a non-empty comma list")
    caps: list[int | None] = []
    for field in fields:
        normalized = field.strip().lower()
        if normalized == "all":
            caps.append(None)
            continue
        try:
            cap = int(normalized)
        except ValueError as error:
            raise ValueError(
                "recursive-event-caps entries must be positive integers or 'all'"
            ) from error
        if cap <= 0:
            raise ValueError(
                "recursive-event-caps entries must be positive integers or 'all'"
            )
        caps.append(cap)
    return tuple(caps)


def _recursive_event_cap_for_epoch(
    caps: Sequence[int | None],
    *,
    epoch: int,
) -> int | None:
    """Return the cap for a zero-based logical epoch after the one-shot epoch."""

    if epoch < 1:
        raise ValueError("a recursive event cap requires logical epoch 2 or later")
    schedule_index = epoch - 1
    if schedule_index >= len(caps):
        raise ValueError("recursive event cap schedule does not cover this epoch")
    return caps[schedule_index]


def _apply_recursive_event_cap(
    events: Sequence[TokenClockEvent | AgenticCompressionEvent],
    cap: int | None,
) -> tuple[TokenClockEvent | AgenticCompressionEvent, ...]:
    if cap is not None and cap <= 0:
        raise ValueError("recursive event cap must be positive or None")
    selected = tuple(events if cap is None else events[:cap])
    if selected != tuple(events[: len(selected)]):
        raise RuntimeError("recursive curriculum must retain a chronological prefix")
    return selected


def _validate_args(args: argparse.Namespace) -> None:
    recursive_event_caps = _parse_recursive_event_caps(args.recursive_event_caps)
    if (
        args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
        and (args.resume_at_recursive_epoch or args.resume_transition is not None)
    ):
        raise ValueError(
            "Qwen3.8 WM4 normal resume forbids cursor overrides and "
            "resume-transition"
        )
    if recursive_event_caps:
        if recursive_event_caps != QWEN38_PROGRESSIVE_RECURSIVE_EVENT_CAPS:
            raise ValueError(
                "the Qwen3.8 progressive run requires recursive-event-caps="
                "2,3,4,5,6,7,8,9,10,all"
            )
        if (
            args.target_family != "qwen38"
            or args.data_contract != LEGACY_TOKEN_CLOCK_DATA_CONTRACT
            or args.objective != "kl"
            or args.curriculum_mode != "auto"
            or args.epochs != 11
        ):
            raise ValueError(
                "progressive recursive caps require Qwen3.8 legacy KL, "
                "curriculum-mode=auto, and epochs=11"
            )
        if args.resume_from is None:
            raise ValueError("progressive recursive caps require resume-from")
        expected_trainer_sha256 = args.expected_progressive_trainer_sha256
        if (
            not isinstance(expected_trainer_sha256, str)
            or len(expected_trainer_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_trainer_sha256
            )
        ):
            raise ValueError(
                "progressive recursion requires a lowercase 64-hex "
                "expected-progressive-trainer-sha256"
            )
        observed_trainer_sha256 = _sha256(Path(__file__).resolve())
        if observed_trainer_sha256 != expected_trainer_sha256:
            raise ValueError(
                "progressive trainer SHA256 differs from its launch-bound identity: "
                f"expected={expected_trainer_sha256}, "
                f"observed={observed_trainer_sha256}"
            )
        progressive_optimizer_contract = {
            "grad_accumulation": 8,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "optimizer_memory_mode": "default",
            "warmup_steps": 500,
            "max_grad_norm": 1.0,
            "seed": 42,
            "disable_target_gradient_checkpointing": False,
            "disable_compressor_gradient_checkpointing": False,
        }
        progressive_mismatches = {
            name: (getattr(args, name), expected)
            for name, expected in progressive_optimizer_contract.items()
            if getattr(args, name) != expected
        }
        if progressive_mismatches:
            raise ValueError(
                "progressive recursion must preserve the step-1000 optimizer/RNG "
                f"contract: {progressive_mismatches}"
            )
    if args.resume_transition is not None:
        if not recursive_event_caps:
            raise ValueError("resume-transition requires recursive-event-caps")
        if not args.resume_at_recursive_epoch:
            raise ValueError(
                "the Qwen3.8 step-1000 transition requires resume-at-recursive-epoch"
            )
    elif recursive_event_caps and args.resume_at_recursive_epoch:
        raise ValueError(
            "resume-at-recursive-epoch is only valid on the initial pinned transition; "
            "resume new progressive checkpoints without it"
        )
    if (
        not recursive_event_caps
        and args.expected_progressive_trainer_sha256 is not None
    ):
        raise ValueError(
            "expected-progressive-trainer-sha256 is only valid with "
            "recursive-event-caps"
        )
    if args.verify_parameters_finite_after_step:
        if (
            not recursive_event_caps
            and args.data_contract != QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
        ):
            raise ValueError(
                "verify-parameters-finite-after-step is reserved for progressive "
                "recursive or Qwen3.8 vibe action-KL canaries"
            )
        if recursive_event_caps and args.max_steps not in {1001, 1002}:
            raise ValueError(
                "verify-parameters-finite-after-step requires a pinned one-update "
                "progressive canary max-steps value"
            )
        # The frozen formal WM4 run is bounded by its positive epoch count and
        # intentionally uses max_steps=-1 to consume all ten epochs.  Canary
        # manifests independently require max_steps in {1, 2} when their
        # manifest contract is loaded, so accepting -1 here does not weaken
        # the canary boundary.
    if args.resume_from is not None and args.init_compressor is not None:
        raise ValueError("init-compressor and resume-from are mutually exclusive")
    if not math.isfinite(args.kl_loss_weight) or args.kl_loss_weight <= 0:
        raise ValueError("kl-loss-weight must be finite and positive")
    if (
        args.data_contract != MINDFORGE_FRONTIER_CE_DATA_CONTRACT
        and args.objective != "kl"
    ):
        raise ValueError(
            "ce and ce_kl objectives require the pinned MindForge data contract"
        )
    if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        if args.target_family != "qwen38":
            raise ValueError(
                "qwen38_vibe_wm4_action_kl_v1 requires target-family=qwen38"
            )
        if args.curriculum_mode != "recursive":
            raise ValueError(
                "qwen38_vibe_wm4_action_kl_v1 requires curriculum-mode=recursive"
            )
        if args.epochs != QWEN38_VIBE_WM4_ACTION_KL_CONTRACT["epochs"]:
            raise ValueError("Qwen3.8 vibe action-KL requires exactly 10 epochs")
        if (
            args.vibe_working_memory_rounds
            != QWEN38_VIBE_WM4_ACTION_KL_CONTRACT["working_memory_rounds"]
        ):
            raise ValueError("Qwen3.8 vibe action-KL requires exactly four WM rounds")
        if (args.init_compressor is None) == (args.resume_from is None):
            raise ValueError(
                "Qwen3.8 vibe action-KL requires exactly one of init-compressor "
                "or resume-from"
            )
        if args.data.suffix.lower() != ".json":
            raise ValueError(
                "Qwen3.8 vibe action-KL --data must be a frozen JSON manifest"
            )
        if args.expected_data_sha256 is None:
            raise ValueError(
                "Qwen3.8 vibe action-KL requires expected-data-sha256 for the "
                "frozen manifest raw file bytes"
            )
        if args.expected_data_manifest_canonical_sha256 is None:
            raise ValueError(
                "Qwen3.8 vibe action-KL requires "
                "expected-data-manifest-canonical-sha256 for the frozen manifest"
            )
        expected_vibe_trainer_sha256 = args.expected_vibe_trainer_sha256
        if (
            not isinstance(expected_vibe_trainer_sha256, str)
            or len(expected_vibe_trainer_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_vibe_trainer_sha256
            )
        ):
            raise ValueError(
                "Qwen3.8 vibe action-KL requires a lowercase 64-hex "
                "expected-vibe-trainer-sha256"
            )
        observed_vibe_trainer_sha256 = _sha256(Path(__file__).resolve())
        if observed_vibe_trainer_sha256 != expected_vibe_trainer_sha256:
            raise ValueError(
                "Qwen3.8 vibe trainer SHA256 differs from its launch pin: "
                f"expected={expected_vibe_trainer_sha256}, "
                f"observed={observed_vibe_trainer_sha256}"
            )
        if args.teacher_execution_mode != "monolithic_only":
            raise ValueError(
                "Qwen3.8 vibe action-KL requires "
                "teacher-execution-mode=monolithic_only"
            )
        vibe_geometry = {
            "working_memory_rounds": args.vibe_working_memory_rounds,
            "epochs": args.epochs,
            "target_tokens": args.kl_tokens,
            "max_target_sequence_tokens": args.max_target_sequence_tokens,
            "max_teacher_sequence_tokens": args.max_teacher_sequence_tokens,
            "teacher_prefill_chunk_tokens": args.teacher_prefill_chunk_tokens,
            "lm_head_chunk_size": args.lm_head_chunk_size,
            "max_history_tokens": args.max_history_tokens,
            "compression_ratio": COMPRESSION_RATIO,
            "eviction_block_tokens": EVICTION_BLOCK_TOKENS,
        }
        if vibe_geometry != QWEN38_VIBE_WM4_ACTION_KL_CONTRACT:
            raise ValueError(
                "Qwen3.8 vibe action-KL geometry differs from its pinned contract: "
                f"{vibe_geometry}"
            )
    elif args.expected_vibe_trainer_sha256 is not None:
        raise ValueError(
            "expected-vibe-trainer-sha256 is reserved for the Qwen3.8 vibe "
            "action-KL contract"
        )
    if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
        observed_runtime = {
            "torch_public_version": torch.__version__.split("+", 1)[0],
            "transformers_version": importlib.metadata.version("transformers"),
        }
        if observed_runtime != MINDFORGE_RUNTIME_ABI:
            raise RuntimeError(
                "MindForge frontier pilot runtime ABI mismatch: "
                f"expected={MINDFORGE_RUNTIME_ABI}, observed={observed_runtime}"
            )
        if args.target_family != "qwen38":
            raise ValueError(
                "MindForge frontier action CE requires target-family=qwen38"
            )
        if args.curriculum_mode != "recursive":
            raise ValueError("MindForge frontier CE requires curriculum-mode=recursive")
        if args.init_compressor is None:
            raise ValueError(
                "MindForge frontier CE requires a pretrained init-compressor"
            )
        if args.mindforge_event_manifest is None:
            raise ValueError("MindForge frontier CE requires mindforge-event-manifest")
        for name in (
            "expected_mindforge_manifest_sha256",
            "expected_mindforge_manifest_contract_sha256",
        ):
            value = getattr(args, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a pinned lowercase SHA256")
        if args.data.suffix.lower() != ".jsonl":
            raise ValueError("MindForge frontier CE requires a JSONL source")
        if (
            args.mindforge_history_window_tokens <= 0
            or args.mindforge_history_window_tokens % EVICTION_BLOCK_TOKENS
        ):
            raise ValueError(
                "mindforge-history-window-tokens must be a positive multiple of 1024"
            )
        if args.mindforge_history_window_tokens > args.max_history_tokens:
            raise ValueError(
                "mindforge-history-window-tokens exceeds max-history-tokens"
            )
        if args.mindforge_max_action_tokens <= 0:
            raise ValueError("mindforge-max-action-tokens must be positive")
        expected_pilot_contract = MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT
        if args.mindforge_pilot_profile == MINDFORGE_FULL_VIEW_H4096_PROFILE:
            if args.objective != "ce":
                raise ValueError("MindForge 4K full-view pilot requires objective=ce")
            if args.optimizer_memory_mode != "low_peak":
                raise ValueError(
                    "MindForge 4K full-view pilot requires optimizer-memory-mode="
                    "low_peak"
                )
            if (
                args.expected_mindforge_manifest_sha256
                != MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256
                or args.expected_mindforge_manifest_contract_sha256
                != MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256
            ):
                raise ValueError(
                    "MindForge 4K full-view pilot requires its exact frozen "
                    "manifest pins"
                )
            expected_pilot_contract = MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT_H4096
        pilot_values = {
            "history_window_tokens": args.mindforge_history_window_tokens,
            "max_action_tokens": args.mindforge_max_action_tokens,
            "max_target_sequence_tokens": args.max_target_sequence_tokens,
            "max_teacher_sequence_tokens": args.max_teacher_sequence_tokens,
            "teacher_prefill_chunk_tokens": args.teacher_prefill_chunk_tokens,
            "lm_head_chunk_size": args.lm_head_chunk_size,
            "compression_ratio": COMPRESSION_RATIO,
            "eviction_block_tokens": EVICTION_BLOCK_TOKENS,
            "final_working_memory_tokens": RAW_WORKING_MEMORY_TOKENS,
        }
        if pilot_values != expected_pilot_contract:
            raise ValueError(
                "MindForge frontier action pilot geometry differs from its pinned "
                f"{args.mindforge_pilot_profile} contract: {pilot_values}"
            )
        if args.teacher_execution_mode != "monolithic_only":
            raise ValueError(
                "MindForge frontier action pilot requires monolithic_only teacher "
                "arguments (the CE-only arm still executes no raw teacher)"
            )
    elif any(
        value is not None
        for value in (
            args.mindforge_event_manifest,
            args.expected_mindforge_manifest_sha256,
            args.expected_mindforge_manifest_contract_sha256,
        )
    ):
        raise ValueError(
            "MindForge manifest pins are reserved for the MindForge data contract"
        )
    elif args.mindforge_pilot_profile != MINDFORGE_PRIMARY_H8192_PROFILE:
        raise ValueError(
            "non-default MindForge pilot profiles are reserved for the MindForge "
            "data contract"
        )
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max-steps must be -1 or positive")
    for name in (
        "grad_accumulation",
        "kl_tokens",
        "lm_head_chunk_size",
        "max_target_sequence_tokens",
        "max_teacher_sequence_tokens",
        "teacher_prefill_chunk_tokens",
        "max_history_tokens",
        "log_interval",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("optimizer hyperparameters are invalid")
    if not math.isfinite(args.cached_teacher_max_output_kl) or (
        args.cached_teacher_max_output_kl < 0
    ):
        raise ValueError("cached-teacher-max-output-kl must be finite and non-negative")
    if not math.isfinite(args.cached_teacher_min_top1_agreement) or not (
        0 <= args.cached_teacher_min_top1_agreement <= 1
    ):
        raise ValueError("cached-teacher-min-top1-agreement must be within [0, 1]")
    if args.max_teacher_sequence_tokens > 262144:
        raise ValueError("Qwen3.5-family target context cannot exceed 262144 tokens")
    if args.teacher_prefill_chunk_tokens > args.max_target_sequence_tokens:
        raise ValueError(
            "teacher-prefill-chunk-tokens cannot exceed the single-forward cap"
        )
    if args.save_interval < 0:
        raise ValueError("save-interval must be non-negative")
    if args.resume_at_recursive_epoch and args.resume_from is None:
        raise ValueError("resume-at-recursive-epoch requires resume-from")
    if args.resume_at_recursive_epoch and args.curriculum_mode != "auto":
        raise ValueError("resume-at-recursive-epoch requires curriculum-mode=auto")
    if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
        if args.target_family != "qwen36_moe":
            raise ValueError(
                "qwen36_agentic_tool_batches_v1 requires target-family=qwen36_moe"
            )
        if args.data.suffix.lower() != ".parquet":
            raise ValueError("qwen36_agentic_tool_batches_v1 requires a parquet input")
        if not args.source_dataset_id or not args.source_dataset_revision:
            raise ValueError(
                "agentic parquet training requires source-dataset-id and an "
                "immutable source-dataset-revision"
            )
        if len(args.source_dataset_revision) != 40 or any(
            character not in "0123456789abcdef"
            for character in args.source_dataset_revision
        ):
            raise ValueError(
                "source-dataset-revision must be a full lowercase commit SHA"
            )
        expected_audit_counts = (
            args.expected_agentic_total_rows,
            args.expected_agentic_valid_rows,
            args.expected_agentic_rejected_rows,
        )
        if args.agentic_data_audit is None or any(
            value is None for value in expected_audit_counts
        ):
            raise ValueError(
                "agentic parquet training requires a merged data audit and every "
                "expected agentic row count"
            )
        if any(int(value) < 0 for value in expected_audit_counts):
            raise ValueError("expected agentic row counts must be non-negative")
        if (
            args.expected_agentic_valid_rows + args.expected_agentic_rejected_rows
            != args.expected_agentic_total_rows
        ):
            raise ValueError("expected valid+rejected rows must equal total rows")
        expected_reject_categories = _parse_expected_reject_categories(
            args.expected_agentic_reject_category
        )
        if (
            not expected_reject_categories
            or sum(expected_reject_categories.values())
            != args.expected_agentic_rejected_rows
        ):
            raise ValueError(
                "expected reject categories must exactly account for rejected rows"
            )
        for name in (
            "expected_agentic_reject_rows_sha256",
            "expected_agentic_audit_sha256",
            "expected_agentic_audit_contract_sha256",
            "expected_agentic_12288_cap_stats_sha256",
        ):
            value = getattr(args, name)
            if (
                value is None
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a pinned lowercase SHA256")
    elif any(
        value is not None
        for value in (
            args.source_dataset_id,
            args.source_dataset_revision,
            args.agentic_data_audit,
            args.expected_agentic_total_rows,
            args.expected_agentic_valid_rows,
            args.expected_agentic_rejected_rows,
            *args.expected_agentic_reject_category,
            args.expected_agentic_reject_rows_sha256,
            args.expected_agentic_audit_sha256,
            args.expected_agentic_audit_contract_sha256,
            args.expected_agentic_12288_cap_stats_sha256,
        )
    ):
        raise ValueError(
            "source/audit provenance flags are reserved for the agentic parquet contract"
        )
    parity_selector = (args.parity_line_index, args.parity_event_index)
    uses_raw_teacher = _objective_uses_raw_teacher(args.objective)
    if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT and (
        args.verify_cached_teacher
        or args.cached_teacher_parity_artifact
        or args.required_parity_chunk_tokens
        or parity_selector[0] is not None
        or parity_selector[1] is not None
    ):
        raise ValueError(
            "MindForge pilot does not accept legacy cached-teacher parity selectors"
        )
    if not uses_raw_teacher and (
        args.verify_cached_teacher
        or args.cached_teacher_parity_artifact
        or args.required_parity_chunk_tokens
        or parity_selector[0] is not None
        or parity_selector[1] is not None
    ):
        raise ValueError(
            "objective=ce cannot select or validate a raw-teacher parity path"
        )
    if (parity_selector[0] is None) != (parity_selector[1] is None):
        raise ValueError(
            "parity-line-index and parity-event-index must be set together"
        )
    if parity_selector[0] is not None:
        if parity_selector[0] < 0 or parity_selector[1] < 0:
            raise ValueError("fixed-probe line/event indices must be non-negative")
        if args.teacher_execution_mode == "cached" and not args.verify_cached_teacher:
            raise ValueError(
                "a cached-teacher fixed probe requires verify-cached-teacher"
            )
        if (
            args.curriculum_mode != "recursive"
            or args.epochs != 1
            or args.max_steps != 1
            or args.grad_accumulation != 1
            or args.resume_from is not None
            or not args.skip_final_checkpoint
        ):
            raise ValueError(
                "fixed-probe selectors require recursive, epochs=1, max-steps=1, "
                "grad-accumulation=1, no resume, and skip-final-checkpoint"
            )
        expected_parity = (
            args.parity_expected_example_id,
            args.parity_expected_teacher_tokens,
            args.parity_expected_student_tokens,
            args.parity_expected_cached_chunks,
        )
        if any(value is None for value in expected_parity):
            raise ValueError("fixed-probe selectors require every expected value")
        expected_row_sha256 = args.fixed_probe_expected_row_sha256
        if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
            if (
                expected_row_sha256 is None
                or len(expected_row_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_row_sha256
                )
            ):
                raise ValueError(
                    "an agentic parquet fixed probe requires a canonical lowercase "
                    "fixed-probe-expected-row-sha256"
                )
        elif expected_row_sha256 is not None:
            raise ValueError(
                "fixed-probe-expected-row-sha256 is only valid for agentic parquet"
            )
        if any(int(value) <= 0 for value in expected_parity[1:3]):
            raise ValueError("fixed-probe expected sequence widths must be positive")
        expected_cached_chunks = int(expected_parity[3])
        if expected_cached_chunks < 0:
            raise ValueError("fixed-probe expected cached chunks cannot be negative")
        if args.teacher_execution_mode == "monolithic_only" and expected_cached_chunks:
            raise ValueError(
                "a monolithic-only fixed probe must expect zero cached chunks"
            )
        if args.teacher_execution_mode == "cached" and expected_cached_chunks <= 0:
            raise ValueError("a cached-teacher fixed probe must expect cached chunks")
    if args.verify_cached_teacher and parity_selector[0] is None:
        raise ValueError("verify-cached-teacher requires an explicit parity selector")
    if bool(args.cached_teacher_parity_artifact) != bool(
        args.required_parity_chunk_tokens
    ):
        raise ValueError(
            "cached-teacher-parity-artifact and required-parity-chunk-tokens "
            "must be supplied together"
        )
    if args.verify_cached_teacher and args.cached_teacher_parity_artifact:
        raise ValueError("a parity smoke cannot consume formal parity artifacts")
    if any(value <= 0 for value in args.required_parity_chunk_tokens):
        raise ValueError("required parity chunk sizes must be positive")
    if uses_raw_teacher and args.teacher_execution_mode == "monolithic_only":
        if args.verify_cached_teacher:
            raise ValueError(
                "monolithic-only teacher execution cannot verify cached teacher"
            )
        if args.cached_teacher_parity_artifact or args.required_parity_chunk_tokens:
            raise ValueError(
                "monolithic-only teacher execution cannot consume cached-teacher "
                "parity evidence"
            )
        if args.teacher_prefill_chunk_tokens != args.max_teacher_sequence_tokens:
            raise ValueError(
                "monolithic-only teacher execution requires the forward cap to "
                "equal the teacher sequence hard cap"
            )
    if (
        uses_raw_teacher
        and args.target_family == "qwen38"
        and not args.verify_cached_teacher
    ):
        if args.teacher_execution_mode == "cached":
            if len(args.cached_teacher_parity_artifact) != 2 or sorted(
                args.required_parity_chunk_tokens
            ) != [512, 8192]:
                raise ValueError(
                    "formal Qwen3.8 cached training requires the canonical 512- "
                    "and 8192-token cached-teacher parity evidence"
                )
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
                production_contract = {
                    name: value
                    for name, value in QWEN38_VIBE_WM4_ACTION_KL_CONTRACT.items()
                    if name
                    in {
                        "teacher_prefill_chunk_tokens",
                        "target_tokens",
                        "lm_head_chunk_size",
                        "max_target_sequence_tokens",
                        "max_teacher_sequence_tokens",
                        "max_history_tokens",
                    }
                }
                production_contract["kl_tokens"] = production_contract.pop(
                    "target_tokens"
                )
                contract_name = "WM4 action-KL cached production parity contract"
            else:
                production_contract = {
                    "teacher_prefill_chunk_tokens": 8192,
                    "kl_tokens": 256,
                    "lm_head_chunk_size": 16,
                    "max_target_sequence_tokens": 8192,
                    "max_teacher_sequence_tokens": 262144,
                    "max_history_tokens": 262144,
                }
                contract_name = "cached production parity contract"
        else:
            production_contract = QWEN38_MONOLITHIC_12K_PRODUCTION_CONTRACT
            contract_name = "monolithic-only 12288-token production contract"
        mismatches = {
            name: (getattr(args, name), expected)
            for name, expected in production_contract.items()
            if int(getattr(args, name)) != expected
        }
        if mismatches:
            raise ValueError(
                "formal Qwen3.8 training arguments do not match the "
                f"{contract_name}: {mismatches}"
            )
        if args.teacher_execution_mode == "cached" and (
            args.cached_teacher_max_output_kl != 1e-4
            or args.cached_teacher_min_top1_agreement != 1.0
        ):
            raise ValueError(
                "formal Qwen3.8 training requires cached-teacher KL <= 1e-4 "
                "and exact top-1 agreement"
            )
    if uses_raw_teacher and args.target_family == "qwen36_moe":
        if args.teacher_execution_mode != "monolithic_only":
            raise ValueError(
                "Qwen3.6-35B-A3B training currently requires the verified "
                "monolithic-only teacher path"
            )
        mismatches = {
            name: (getattr(args, name), expected)
            for name, expected in QWEN36_MOE_MONOLITHIC_12K_PRODUCTION_CONTRACT.items()
            if int(getattr(args, name)) != expected
        }
        if mismatches:
            raise ValueError(
                "Qwen3.6-35B-A3B training arguments do not match the "
                f"monolithic-only 12288-token contract: {mismatches}"
            )
    if not args.target_model.is_dir() or not args.data.is_file():
        raise FileNotFoundError("target model or training data is missing")
    if args.init_compressor is not None and not args.init_compressor.is_file():
        raise FileNotFoundError("init-compressor weights are missing")
    if (
        args.mindforge_event_manifest is not None
        and not args.mindforge_event_manifest.is_file()
    ):
        raise FileNotFoundError("MindForge event manifest is missing")
    if (
        args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT
        and not args.agentic_data_audit.is_file()
    ):
        raise FileNotFoundError("agentic merged CPU audit is missing")
    manifest_family = _target_family_from_manifest(args.target_model)
    if args.target_family != manifest_family:
        raise ValueError(
            "target-family does not match the target snapshot manifest: "
            f"argument={args.target_family!r}, manifest={manifest_family!r}"
        )
    if args.expected_data_sha256 is not None and (
        len(args.expected_data_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_data_sha256
        )
    ):
        raise ValueError("expected-data-sha256 must be a lowercase SHA256 digest")
    if args.expected_data_manifest_canonical_sha256 is not None and (
        len(args.expected_data_manifest_canonical_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_data_manifest_canonical_sha256
        )
    ):
        raise ValueError(
            "expected-data-manifest-canonical-sha256 must be a lowercase SHA256 digest"
        )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_RUN_CONFIG_ENVELOPE_KEYS = {
    "run_contract_sha256",
    "output_dir",
    "resume_from",
    "init_compressor",
    "resume_cursor_override",
    "resume_transition_invocation",
    "wm4_resume_binding",
    "wm4_resume_source_checkpoint_identity",
}


def _contract_from_run_config(run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in run_config.items()
        if key not in _RUN_CONFIG_ENVELOPE_KEYS
    }


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _validate_run_config_self_hash(
    run_config: dict[str, Any],
    *,
    description: str,
) -> str:
    recorded = run_config.get("run_contract_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(f"{description} has no valid run-contract SHA256")
    observed = _canonical_json_sha256(_contract_from_run_config(run_config))
    if observed != recorded:
        raise ValueError(
            f"{description} run-contract self-hash mismatch: "
            f"recorded={recorded}, observed={observed}"
        )
    return recorded


def _identity_file_sha256(identity: Any, group: str) -> dict[str, str]:
    if not isinstance(identity, dict) or not isinstance(identity.get(group), dict):
        raise ValueError(f"implementation identity has no valid {group}")
    projected: dict[str, str] = {}
    for name, receipt in identity[group].items():
        if (
            not isinstance(name, str)
            or not isinstance(receipt, dict)
            or not isinstance(receipt.get("sha256"), str)
        ):
            raise ValueError(f"implementation identity has malformed {group}")
        projected[name] = receipt["sha256"]
    return projected


def _expected_qwen38_progressive_target_loader_receipt(
    target_content_identity: dict[str, Any],
) -> dict[str, Any]:
    loader_audit = QWEN38_PROGRESSIVE_SEMANTIC_AUDIT[
        "authorized_target_loader_refactor"
    ]
    frozen_target = loader_audit["frozen_target"]
    if (
        target_content_identity.get("target_content_sha256")
        != frozen_target["target_content_sha256"]
    ):
        raise ValueError("progressive target content differs from loader audit")
    unsigned = {
        "schema": "qwen38_progressive_target_loader_runtime_v1",
        "classification": loader_audit["classification"],
        "target_content_sha256": frozen_target["target_content_sha256"],
        "target_config_sha256": frozen_target["config_sha256"],
        "model_type": frozen_target["model_type"],
        "architecture": frozen_target["architecture"],
        "model_class": (
            "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM"
        ),
        "destination_loader_sha256": loader_audit["destination_loader_sha256"],
        "callsite": {
            "local_files_only": True,
            "dtype": "torch.bfloat16",
            "attn_implementation": "sdpa",
            "tp_plan": "auto",
        },
        "semantic_audit_sha256": _canonical_json_sha256(loader_audit),
    }
    return {**unsigned, "receipt_sha256": _canonical_json_sha256(unsigned)}


def _qwen38_progressive_target_loader_runtime_receipt(
    model: Qwen35TextTarget,
    *,
    target_content_identity: dict[str, Any],
) -> dict[str, Any]:
    receipt = _expected_qwen38_progressive_target_loader_receipt(
        target_content_identity
    )
    observed = {
        "model_type": str(model.config.model_type),
        "architecture": model.__class__.__name__,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    }
    expected = {key: receipt[key] for key in observed}
    if observed != expected:
        raise ValueError(
            "progressive target loader runtime dispatch changed: "
            f"expected={expected}, observed={observed}"
        )
    return receipt


def _progressive_implementation_transition_receipt(
    *,
    source_identity: dict[str, Any],
    destination_identity: dict[str, Any],
    expected_trainer_sha256: str,
) -> dict[str, Any]:
    """Bind the only authorized old-to-new implementation closure.

    The Qwen3.8 compressor and KL objective paths are AST-identical to the
    frozen legacy snapshots.  Their enclosing files changed only to add an
    unrelated Qwen3.6-MoE geometry and activation-checkpointed CE, respectively.
    The trainer itself necessarily changes to implement the curriculum and is
    therefore bound externally by its exact launch-time SHA256.
    """

    expected = QWEN38_STEP1000_SOURCE_CONTRACT
    source_local = _identity_file_sha256(source_identity, "local_files")
    if source_local != expected["implementation_local_file_sha256"]:
        raise ValueError(
            "progressive transition source implementation closure is not pinned"
        )
    source_runtime = {
        field: source_identity.get(field) for field in QWEN38_STEP1000_SOURCE_RUNTIME
    }
    if source_runtime != QWEN38_STEP1000_SOURCE_RUNTIME:
        raise ValueError("progressive transition source runtime closure is not pinned")

    destination_local = _identity_file_sha256(destination_identity, "local_files")
    expected_destination_local = {
        **QWEN38_PROGRESSIVE_DESTINATION_LOCAL_FILE_SHA256,
        "trainer": expected_trainer_sha256,
    }
    if destination_local != expected_destination_local:
        raise ValueError(
            "progressive transition destination implementation closure changed: "
            f"expected={expected_destination_local}, observed={destination_local}"
        )

    for field, expected_runtime_value in QWEN38_STEP1000_SOURCE_RUNTIME.items():
        if destination_identity.get(field) != expected_runtime_value:
            raise ValueError(
                f"progressive transition changes runtime implementation field {field}"
            )

    source_dependencies = _identity_file_sha256(source_identity, "dependency_files")
    if source_dependencies != QWEN38_STEP1000_SOURCE_DEPENDENCY_FILE_SHA256:
        raise ValueError(
            "progressive transition source dependency closure is not pinned"
        )
    destination_dependencies = _identity_file_sha256(
        destination_identity, "dependency_files"
    )
    expected_destination_dependencies = {
        **source_dependencies,
        **QWEN38_PROGRESSIVE_ADDED_DEPENDENCY_FILE_SHA256,
    }
    if destination_dependencies != expected_destination_dependencies:
        raise ValueError(
            "progressive transition dependency closure changed: "
            f"expected={expected_destination_dependencies}, "
            f"observed={destination_dependencies}"
        )

    return {
        "schema": "qwen38_progressive_implementation_transition_v1",
        "source_identity_sha256": _canonical_json_sha256(source_identity),
        "source_local_file_sha256": source_local,
        "destination_local_file_sha256": destination_local,
        "unchanged_runtime": {
            field: source_identity[field] for field in QWEN38_STEP1000_SOURCE_RUNTIME
        },
        "source_dependency_file_sha256": source_dependencies,
        "added_dependency_file_sha256": dict(
            QWEN38_PROGRESSIVE_ADDED_DEPENDENCY_FILE_SHA256
        ),
        "semantic_audit": QWEN38_PROGRESSIVE_SEMANTIC_AUDIT,
    }


def _validate_progressive_implementation_transition_receipt(
    receipt: Any,
    *,
    destination_identity: dict[str, Any],
    expected_trainer_sha256: str,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ValueError("progressive lineage has no implementation transition")
    expected_destination_local = {
        **QWEN38_PROGRESSIVE_DESTINATION_LOCAL_FILE_SHA256,
        "trainer": expected_trainer_sha256,
    }
    required = {
        "schema": "qwen38_progressive_implementation_transition_v1",
        "source_identity_sha256": (
            "a01ee72b1484d4361a84cec84e1409a000b8cd6ac8dc6e82b35fb9222ccfbbbb"
        ),
        "source_local_file_sha256": QWEN38_STEP1000_SOURCE_CONTRACT[
            "implementation_local_file_sha256"
        ],
        "destination_local_file_sha256": expected_destination_local,
        "added_dependency_file_sha256": (
            QWEN38_PROGRESSIVE_ADDED_DEPENDENCY_FILE_SHA256
        ),
        "semantic_audit": QWEN38_PROGRESSIVE_SEMANTIC_AUDIT,
        "unchanged_runtime": QWEN38_STEP1000_SOURCE_RUNTIME,
        "source_dependency_file_sha256": (
            QWEN38_STEP1000_SOURCE_DEPENDENCY_FILE_SHA256
        ),
    }
    if set(receipt) != set(required) or receipt != required:
        raise ValueError("progressive implementation transition receipt changed")
    if _identity_file_sha256(destination_identity, "local_files") != (
        expected_destination_local
    ):
        raise ValueError("progressive destination local implementation changed")
    expected_dependencies = {
        **QWEN38_STEP1000_SOURCE_DEPENDENCY_FILE_SHA256,
        **QWEN38_PROGRESSIVE_ADDED_DEPENDENCY_FILE_SHA256,
    }
    if (
        _identity_file_sha256(destination_identity, "dependency_files")
        != expected_dependencies
    ):
        raise ValueError("progressive destination dependency closure changed")
    for field, value in QWEN38_STEP1000_SOURCE_RUNTIME.items():
        if destination_identity.get(field) != value:
            raise ValueError(f"progressive destination runtime field changed: {field}")
    return receipt


def _validate_qwen38_step1000_transition_source(
    checkpoint: Path,
    *,
    expected: dict[str, Any] = QWEN38_STEP1000_SOURCE_CONTRACT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one explicitly authorized legacy checkpoint fork."""

    resolved = checkpoint.expanduser().resolve()
    trainer_path = resolved / "trainer_state.json"
    run_config_path = resolved.parent / "run_config.json"
    trainer = _read_json_object(trainer_path, description="source trainer state")
    source_run_config = _read_json_object(
        run_config_path,
        description="source run config",
    )
    source_run_hash = _validate_run_config_self_hash(
        source_run_config,
        description="source run config",
    )
    if source_run_hash != expected["run_contract_sha256"]:
        raise ValueError("resume transition source run-contract SHA256 is not pinned")

    expected_trainer = {
        "format_version": expected["format_version"],
        "schema": expected["schema"],
        "global_step": expected["global_step"],
        "next_epoch": expected["next_epoch"],
        "next_line_index": expected["next_line_index"],
        "run_contract_sha256": expected["run_contract_sha256"],
        "rank_files": [
            f"rank-{rank:05d}.pt" for rank in range(int(expected["world_size"]))
        ],
    }
    observed_trainer = {key: trainer.get(key) for key in expected_trainer}
    if observed_trainer != expected_trainer:
        raise ValueError(
            "resume transition source trainer contract mismatch: "
            f"expected={expected_trainer}, observed={observed_trainer}"
        )
    if _sha256(trainer_path) != expected["trainer_state_file_sha256"]:
        raise ValueError("resume transition source trainer-state SHA256 changed")
    for rank_file in expected_trainer["rank_files"]:
        rank_path = resolved / rank_file
        if not rank_path.is_file():
            raise FileNotFoundError(
                f"resume transition source rank payload is missing: {rank_file}"
            )
        if _sha256(rank_path) != expected["rank_file_sha256"][rank_file]:
            raise ValueError(
                f"resume transition source rank payload SHA256 changed: {rank_file}"
            )

    source_data_stat = source_run_config.get("data_stat")
    source_contract_fields = {
        "format_version": source_run_config.get("format_version"),
        "schema": source_run_config.get("schema"),
        "target_family": source_run_config.get("target_family"),
        "data_sha256": source_run_config.get("data_sha256"),
        "data_size": (
            source_data_stat.get("size") if isinstance(source_data_stat, dict) else None
        ),
    }
    expected_source_fields = {
        "format_version": expected["format_version"],
        "schema": expected["schema"],
        "target_family": "qwen38",
        "data_sha256": expected["data_sha256"],
        "data_size": expected["data_size"],
    }
    if source_contract_fields != expected_source_fields:
        raise ValueError(
            "resume transition source data/model contract mismatch: "
            f"expected={expected_source_fields}, observed={source_contract_fields}"
        )
    if _sha256(run_config_path) != expected["run_config_file_sha256"]:
        raise ValueError("resume transition source run-config file SHA256 changed")
    source_implementation_identity = source_run_config.get("implementation_identity")
    if not isinstance(source_implementation_identity, dict):
        raise ValueError("resume transition source implementation identity is missing")
    if (
        _identity_file_sha256(source_implementation_identity, "local_files")
        != expected["implementation_local_file_sha256"]
    ):
        raise ValueError("resume transition source local implementation changed")
    source_implementation_identity_sha256 = _canonical_json_sha256(
        source_implementation_identity
    )
    if (
        source_implementation_identity_sha256
        != expected["implementation_identity_sha256"]
    ):
        raise ValueError("resume transition source implementation identity changed")

    lineage = {
        "schema": QWEN38_PROGRESSIVE_LINEAGE_SCHEMA,
        "transition": QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION,
        "source_checkpoint": str(resolved),
        "source_run_config": str(run_config_path.resolve()),
        "source_run_contract_sha256": source_run_hash,
        "source_checkpoint_format_version": expected["format_version"],
        "source_checkpoint_schema": expected["schema"],
        "source_world_size": expected["world_size"],
        "source_global_step": expected["global_step"],
        "source_cursor": {
            "next_epoch": expected["next_epoch"],
            "next_line_index": expected["next_line_index"],
        },
        "source_data": {
            "sha256": expected["data_sha256"],
            "size": expected["data_size"],
        },
        "source_artifact_sha256": {
            "run_config.json": expected["run_config_file_sha256"],
            "trainer_state.json": expected["trainer_state_file_sha256"],
            **expected["rank_file_sha256"],
        },
        "source_implementation_identity_sha256": (
            source_implementation_identity_sha256
        ),
        "source_implementation_local_file_sha256": expected[
            "implementation_local_file_sha256"
        ],
        "restored_state": [
            "model",
            "optimizer",
            "rank_cpu_rng",
            "rank_cuda_rng",
            "global_step",
        ],
        "cursor_override": {
            "from": [expected["next_epoch"], expected["next_line_index"]],
            "to": [1, 0],
        },
        "allowed_contract_deltas": [
            "implementation_identity_exact_transition_receipt",
            "curriculum",
            "epochs_and_declared_parser_surface",
            "explicit_versioned_contract_receipts",
        ],
    }
    return lineage, source_run_config


def _validate_progressive_lineage(lineage: Any) -> dict[str, Any]:
    if not isinstance(lineage, dict):
        raise ValueError("progressive checkpoint is missing resume lineage")
    expected = QWEN38_STEP1000_SOURCE_CONTRACT
    required = {
        "schema": QWEN38_PROGRESSIVE_LINEAGE_SCHEMA,
        "transition": QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION,
        "source_run_contract_sha256": expected["run_contract_sha256"],
        "source_checkpoint_format_version": expected["format_version"],
        "source_checkpoint_schema": expected["schema"],
        "source_world_size": expected["world_size"],
        "source_global_step": expected["global_step"],
        "source_cursor": {
            "next_epoch": expected["next_epoch"],
            "next_line_index": expected["next_line_index"],
        },
        "source_data": {
            "sha256": expected["data_sha256"],
            "size": expected["data_size"],
        },
        "source_artifact_sha256": {
            "run_config.json": expected["run_config_file_sha256"],
            "trainer_state.json": expected["trainer_state_file_sha256"],
            **expected["rank_file_sha256"],
        },
        "source_implementation_identity_sha256": (
            "a01ee72b1484d4361a84cec84e1409a000b8cd6ac8dc6e82b35fb9222ccfbbbb"
        ),
        "source_implementation_local_file_sha256": expected[
            "implementation_local_file_sha256"
        ],
        "restored_state": [
            "model",
            "optimizer",
            "rank_cpu_rng",
            "rank_cuda_rng",
            "global_step",
        ],
        "cursor_override": {
            "from": [expected["next_epoch"], expected["next_line_index"]],
            "to": [1, 0],
        },
        "allowed_contract_deltas": [
            "implementation_identity_exact_transition_receipt",
            "curriculum",
            "epochs_and_declared_parser_surface",
            "explicit_versioned_contract_receipts",
        ],
    }
    observed = {key: lineage.get(key) for key in required}
    if observed != required:
        raise ValueError(
            "progressive checkpoint lineage does not match the pinned transition"
        )
    for name in ("source_checkpoint", "source_run_config"):
        if not isinstance(lineage.get(name), str) or not lineage[name]:
            raise ValueError(f"progressive checkpoint lineage is missing {name}")
    expected_keys = set(required) | {"source_checkpoint", "source_run_config"}
    if "implementation_transition" in lineage:
        expected_keys.add("implementation_transition")
    if set(lineage) != expected_keys:
        raise ValueError("progressive checkpoint lineage keys changed")
    return lineage


def _progressive_resume_context(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    caps = _parse_recursive_event_caps(args.recursive_event_caps)
    if not caps:
        return None
    if args.resume_from is None:
        raise ValueError("progressive resume context requires a checkpoint")
    checkpoint = args.resume_from.expanduser().resolve()
    if args.resume_transition == QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION:
        lineage, source_run_config = _validate_qwen38_step1000_transition_source(
            checkpoint
        )
        return {
            "lineage": lineage,
            "transition_source_run_config": source_run_config,
        }

    parent_run_config_path = checkpoint.parent / "run_config.json"
    parent_run_config = _read_json_object(
        parent_run_config_path,
        description="progressive parent run config",
    )
    parent_run_contract_sha256 = _validate_run_config_self_hash(
        parent_run_config,
        description="progressive parent run config",
    )
    lineage = _validate_progressive_lineage(parent_run_config.get("resume_lineage"))
    return {
        "lineage": lineage,
        "transition_source_run_config": None,
        "resume_checkpoint_run_contract_sha256": parent_run_contract_sha256,
    }


def _validate_qwen38_progressive_destination(
    source: dict[str, Any],
    destination: dict[str, Any],
) -> None:
    """Allow only the closed-world, predeclared fork from pinned step 1000."""

    source_contract = _contract_from_run_config(source)
    source_args = source_contract.get("args")
    destination_args = destination.get("args")
    if not isinstance(source_args, dict) or not isinstance(destination_args, dict):
        raise ValueError("progressive source/destination args are malformed")

    added_top_level = {
        "agentic_data_audit",
        "compressor_architecture_upstream_receipt",
        "compressor_initialization",
        "data_contract",
        "gradient_accumulation_contract",
        "resume_lineage",
        "target_loader_runtime_receipt",
        "trajectory_supervision",
    }
    expected_destination_keys = set(source_contract) | added_top_level
    if set(destination) != expected_destination_keys:
        raise ValueError(
            "progressive destination top-level contract keys changed: "
            f"expected={sorted(expected_destination_keys)}, "
            f"observed={sorted(destination)}"
        )

    expected_added_values = {
        "agentic_data_audit": None,
        "compressor_architecture_upstream_receipt": None,
        "data_contract": {
            "name": LEGACY_TOKEN_CLOCK_DATA_CONTRACT,
            "source_format": "jsonl_conversations",
        },
        "gradient_accumulation_contract": (
            "mean_over_events_within_each_example_then_mean_over_actual_examples; "
            "short_epoch_tail_is_flushed_before_curriculum_transition; "
            "checkpoint_cursor_points_after_the_flushed_rows"
        ),
        "trajectory_supervision": (
            "offline regenerated SFT trajectory supplies token-clock contexts only; "
            "the frozen target distribution is computed online at every event"
        ),
    }
    for key, expected_value in expected_added_values.items():
        if destination.get(key) != expected_value:
            raise ValueError(
                "progressive destination changes an explicit added contract field: "
                f"{key}"
            )

    expected_loader_receipt = _expected_qwen38_progressive_target_loader_receipt(
        source_contract.get("target_content_identity")
    )
    if destination.get("target_loader_runtime_receipt") != expected_loader_receipt:
        raise ValueError("progressive target loader runtime receipt changed")

    lineage = _validate_progressive_lineage(destination.get("resume_lineage"))
    expected_initialization = {
        "source": "resume_lineage",
        "source_global_step": lineage["source_global_step"],
        "restored_state": lineage["restored_state"],
    }
    if destination.get("compressor_initialization") != expected_initialization:
        raise ValueError("progressive destination compressor lineage changed")

    expected_curriculum = {
        "schema": "progressive_recursive_event_cap_v1",
        "epoch_1": "source_checkpoint_one_shot_prefix_skipped_by_explicit_transition",
        "epoch_2_to_11": {
            "maximum_chronological_events": [2, 3, 4, 5, 6, 7, 8, 9, 10, "all"],
            "cap_application_order": (
                "teacher_and_student_sequence_caps_then_causal_prefix_cap"
            ),
            "recurrence": (
                "previous_compressor_memory_plus_next_1024_block_with_"
                "stop_gradient_between_state_updates"
            ),
            "loss_weighting": (
                "mean_over_actual_selected_events_then_mean_over_examples"
            ),
        },
    }
    if destination.get("curriculum") != expected_curriculum:
        raise ValueError("progressive destination curriculum changed")

    destination_identity = destination.get("implementation_identity")
    if not isinstance(destination_identity, dict):
        raise ValueError("progressive destination implementation identity is malformed")
    destination_local = _identity_file_sha256(destination_identity, "local_files")
    expected_trainer_sha256 = destination_local.get("trainer")
    if not isinstance(expected_trainer_sha256, str):
        raise ValueError("progressive destination trainer identity is malformed")
    _validate_progressive_implementation_transition_receipt(
        lineage.get("implementation_transition"),
        destination_identity=destination_identity,
        expected_trainer_sha256=expected_trainer_sha256,
    )

    added_args = {
        "agentic_data_audit": None,
        "data_contract": LEGACY_TOKEN_CLOCK_DATA_CONTRACT,
        "expected_agentic_12288_cap_stats_sha256": None,
        "expected_agentic_audit_contract_sha256": None,
        "expected_agentic_audit_sha256": None,
        "expected_agentic_reject_category": [],
        "expected_agentic_reject_rows_sha256": None,
        "expected_agentic_rejected_rows": None,
        "expected_agentic_total_rows": None,
        "expected_agentic_valid_rows": None,
        "expected_progressive_trainer_sha256": expected_trainer_sha256,
        "fixed_probe_expected_row_sha256": None,
        "mindforge_pilot_profile": MINDFORGE_PRIMARY_H8192_PROFILE,
        "optimizer_memory_mode": "default",
        "source_dataset_id": None,
        "source_dataset_revision": None,
    }
    expected_destination_args = json.loads(json.dumps(source_args))
    expected_destination_args["epochs"] = 11
    for key, value in added_args.items():
        if key in expected_destination_args:
            raise ValueError(f"progressive added arg already exists in source: {key}")
        expected_destination_args[key] = value
    if destination_args != expected_destination_args:
        changed_args = {
            key: (expected_destination_args.get(key), destination_args.get(key))
            for key in set(expected_destination_args) | set(destination_args)
            if expected_destination_args.get(key) != destination_args.get(key)
            or (key in expected_destination_args) != (key in destination_args)
        }
        raise ValueError(
            "progressive destination changes a non-authorized args field: "
            f"{changed_args}"
        )

    source_projection = json.loads(json.dumps(source_contract))
    destination_projection = json.loads(json.dumps(destination))
    for key in (
        "implementation_identity",
        "curriculum",
        "args",
    ):
        source_projection.pop(key)
        destination_projection.pop(key)
    for key in added_top_level:
        destination_projection.pop(key)
    if source_projection != destination_projection:
        changed = {
            key: (source_projection.get(key), destination_projection.get(key))
            for key in set(source_projection) | set(destination_projection)
            if source_projection.get(key) != destination_projection.get(key)
            or (key in source_projection) != (key in destination_projection)
        }
        raise ValueError(
            "progressive destination changes a non-authorized source invariant: "
            f"{changed}"
        )


def _portable_compressor_init_receipt(path: Path) -> dict[str, Any]:
    """Bind a portable deployment export to its exact training provenance."""

    artifact = path.expanduser().resolve()
    if not artifact.is_file() or artifact.name != "compressor.pt":
        raise ValueError("init-compressor must point to a portable compressor.pt")
    manifest_path = artifact.parent / "manifest.json"
    config_path = artifact.parent / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError(
            "init-compressor requires adjacent manifest.json and config.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("init-compressor provenance is not valid JSON") from error
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise ValueError("init-compressor provenance must contain JSON objects")
    from context_compression.mindforge_ce_manifest import checkpoint_receipt

    checkpoint_identity = checkpoint_receipt(artifact.parent)

    artifact_sha256 = _sha256(artifact)
    config_file_sha256 = _sha256(config_path)
    config_sha256 = _canonical_json_sha256(config)
    expected_export_schema = "qwen38_27b_r2eact_token_clock_serving_v1"
    expected_manifest_abi = {
        "schema": "context_compression_ddp_v1",
        "world_size": 2,
        "compressor_file": artifact.name,
        "config_file": config_path.name,
        "rank_state_files": [],
        "frozen_target_saved": False,
        "export_schema": expected_export_schema,
    }
    if (
        any(
            manifest.get(name) != expected
            for name, expected in expected_manifest_abi.items()
        )
        or manifest.get("compressor_sha256") != artifact_sha256
        or manifest.get("config_sha256") != config_sha256
    ):
        raise ValueError("init-compressor export ABI/hashes do not match its artifacts")
    if config.get("schema") != expected_export_schema:
        raise ValueError("init-compressor config schema differs from export_schema")
    if config.get("target_family") != "qwen38":
        raise ValueError("init-compressor config target_family must be qwen38")
    source = config.get("deployment_export")
    if not isinstance(source, dict):
        raise ValueError("init-compressor config has no deployment_export provenance")
    source_step = source.get("source_global_step")
    source_run_contract = source.get("source_run_contract_sha256")
    source_stage = source.get("source_curriculum_stage")
    if (
        isinstance(source_step, bool)
        or not isinstance(source_step, int)
        or source_step <= 0
        or not isinstance(source_run_contract, str)
        or len(source_run_contract) != 64
        or any(character not in "0123456789abcdef" for character in source_run_contract)
    ):
        raise ValueError("init-compressor source step/run contract is invalid")
    is_baseline = source_stage == "epoch_1_one_shot_only" and source_step == 1_000
    is_mindforge_ce = (
        source_stage == "mindforge_frontier_ce_recursive_all_epochs"
        and source.get("source_objective") == "ce"
        and checkpoint_identity.get("source_objective") == "ce"
        and isinstance(checkpoint_identity.get("mindforge_ce_lineage"), dict)
        and checkpoint_identity["mindforge_ce_lineage"].get(
            "source_run_contract_sha256"
        )
        == source_run_contract
        and checkpoint_identity["mindforge_ce_lineage"].get("source_global_step")
        == source_step
    )
    if not (is_baseline or is_mindforge_ce):
        raise ValueError("init-compressor source stage/objective lineage is invalid")
    compressor_config = config.get("compressor")
    if not isinstance(compressor_config, dict):
        raise ValueError("init-compressor config has no compressor architecture")
    source_target_content_identity = checkpoint_identity[
        "source_target_content_identity"
    ]
    receipt = {
        "mode": "strict_portable_weights_only",
        "artifact": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": artifact_sha256,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "config": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "config_file_sha256": config_file_sha256,
        "export_schema": manifest.get("export_schema"),
        "source_checkpoint": source.get("source_checkpoint"),
        "source_global_step": source_step,
        "source_curriculum_stage": source_stage,
        "source_run_contract_sha256": source_run_contract,
        "source_target_content_identity": source_target_content_identity,
        "source_target_content_identity_sha256": checkpoint_identity[
            "source_target_content_identity_sha256"
        ],
        "checkpoint_receipt_sha256": checkpoint_identity["receipt_sha256"],
        "compressor_config": compressor_config,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "rng_state_loaded": False,
        "global_step_loaded": False,
    }
    if is_mindforge_ce:
        receipt.update(
            {
                "source_objective": "ce",
                "mindforge_ce_lineage": checkpoint_identity["mindforge_ce_lineage"],
            }
        )
    return receipt


def _load_portable_compressor_weights_only(
    module: torch.nn.Module,
    path: Path,
    *,
    receipt: dict[str, Any],
) -> None:
    """Strictly initialize a bare compressor without checkpoint trainer state."""

    artifact = path.expanduser().resolve()
    if str(artifact) != receipt.get("artifact") or _sha256(artifact) != receipt.get(
        "artifact_sha256"
    ):
        raise ValueError("init-compressor changed after its receipt was created")
    export_config = getattr(module, "export_config", None)
    if callable(export_config) and export_config() != receipt.get("compressor_config"):
        raise ValueError("init-compressor architecture differs from this compressor")
    state = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise TypeError("portable compressor.pt must be a tensor state-dict mapping")
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict init-compressor load returned incompatible keys")


def _validate_vibe_step1000_init_receipt(
    receipt: dict[str, Any],
    *,
    target_content_identity: dict[str, Any],
) -> dict[str, Any]:
    """Admit only the deployed step-1000 weights across the new ABI boundary."""

    expected = {
        "mode": "strict_portable_weights_only",
        "source_global_step": 1_000,
        "source_curriculum_stage": "epoch_1_one_shot_only",
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "rng_state_loaded": False,
        "global_step_loaded": False,
    }
    mismatches = {
        name: (receipt.get(name), value)
        for name, value in expected.items()
        if receipt.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Qwen3.8 vibe action-KL initialization is not the portable step-1000 "
            f"weights-only boundary: {mismatches}"
        )
    source_target_identity = receipt.get("source_target_content_identity")
    if source_target_identity != target_content_identity or receipt.get(
        "source_target_content_identity_sha256"
    ) != _canonical_json_sha256(target_content_identity):
        raise ValueError(
            "Qwen3.8 vibe init-compressor was trained against a different frozen "
            "target identity"
        )
    return {
        **receipt,
        "new_optimizer_created": True,
        "new_warmup_schedule_origin_global_step": 0,
        "new_rng_stream_created": True,
        "new_global_step": 0,
        "data_contract_transition": QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT,
    }


def _vibe_resume_initialization_receipt(
    checkpoint: Path,
    *,
    target_content_identity: dict[str, Any],
) -> dict[str, Any]:
    """Recover the immutable weights-only origin without reloading those weights."""

    run_config_path = checkpoint.expanduser().resolve().parent / "run_config.json"
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Qwen3.8 vibe resume requires its adjacent valid run_config.json"
        ) from error
    if not isinstance(run_config, dict):
        raise ValueError("Qwen3.8 vibe resume run config must be a JSON object")
    data_contract = run_config.get("data_contract")
    if (
        not isinstance(data_contract, dict)
        or data_contract.get("name") != QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
    ):
        raise ValueError("Qwen3.8 vibe resume source has a different data contract")
    receipt = run_config.get("compressor_initialization")
    if not isinstance(receipt, dict):
        raise ValueError("Qwen3.8 vibe resume source has no portable origin receipt")
    return _validate_vibe_step1000_init_receipt(
        receipt,
        target_content_identity=target_content_identity,
    )


def _validate_vibe_manifest_training_contract(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_file_sha256: str,
    args: argparse.Namespace,
    adapter_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], ...]]]:
    """Bind the frozen accepted snapshot, tokenizer, caps, and descriptors."""

    from context_compression.qwen38_coding_trajectory_data import (
        CACHED_PREFILL_CAP,
        MANIFEST_SCHEMA,
        MANIFEST_VERSION,
        PARALLEL_NORMALIZATION,
        QWEN38_GENERATION_BOUNDARY_IDS,
        REPORT_SEQUENCE_CAPS,
        TEACHER_EXECUTION_MODE,
        TRAJECTORY_WEIGHTING,
    )

    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("version") != MANIFEST_VERSION
        or manifest.get("dataset_kind") != "accepted"
        or manifest.get("accepted_only") is not True
    ):
        raise ValueError("Qwen3.8 vibe manifest is not the accepted-only v1 snapshot")
    if manifest.get("row_order") != (
        "ordered_files_then_complete_jsonl_lines_first_id_wins"
    ):
        raise ValueError("Qwen3.8 vibe manifest row order is not deterministic")
    adapter_contract = manifest.get("adapter_contract")
    expected_adapter_contract = {
        "working_memory_rounds": args.vibe_working_memory_rounds,
        "parallel_tool_normalization": PARALLEL_NORMALIZATION,
        "target_token_cap": args.kl_tokens,
        "eviction_chunk_tokens": EVICTION_BLOCK_TOKENS,
        "compression_ratio": COMPRESSION_RATIO,
        "training_epochs": args.epochs,
        "trajectory_weighting": TRAJECTORY_WEIGHTING,
        "epoch_selector": "progressive_monotone_depth_quantiles",
        "generation_boundary_input_ids": list(QWEN38_GENERATION_BOUNDARY_IDS),
        "student_sequence_cap": args.max_target_sequence_tokens,
        "raw_teacher_sequence_cap": args.max_teacher_sequence_tokens,
        "cached_prefill_cap": args.teacher_prefill_chunk_tokens,
        "teacher_execution_mode": TEACHER_EXECUTION_MODE,
        "round_boundary": ("one_assistant_plus_one_canonical_tool_observation"),
        "event_state": "final_serving_state_for_each_target",
    }
    if adapter_contract != expected_adapter_contract:
        raise ValueError(
            "Qwen3.8 vibe manifest adapter contract differs from trainer v1"
        )
    if CACHED_PREFILL_CAP != args.teacher_prefill_chunk_tokens:
        raise RuntimeError("Qwen3.8 vibe adapter cached-prefill constant drifted")

    tokenizer_receipt = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer_receipt, dict)
        or Path(str(tokenizer_receipt.get("path", ""))).resolve()
        != args.target_model.resolve()
    ):
        raise ValueError("Qwen3.8 vibe manifest tokenizer path differs from target")
    tokenizer_files = tokenizer_receipt.get("files")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise ValueError("Qwen3.8 vibe manifest tokenizer receipt is empty")
    expected_tokenizer_files = []
    for item in tokenizer_files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Qwen3.8 vibe tokenizer file receipt is invalid")
        path = args.target_model.resolve() / item["name"]
        if not path.is_file():
            raise FileNotFoundError(f"Qwen3.8 vibe tokenizer file is missing: {path}")
        expected_tokenizer_files.append(
            {
                "name": item["name"],
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    expected_tokenizer = {
        "path": str(args.target_model.resolve()),
        "files": expected_tokenizer_files,
    }
    expected_tokenizer["receipt_sha256"] = _canonical_json_sha256(expected_tokenizer)
    if tokenizer_receipt != expected_tokenizer:
        raise ValueError("Qwen3.8 vibe manifest tokenizer/template receipt is stale")

    descriptors = manifest.get("action_descriptors")
    counts = manifest.get("counts")
    if not isinstance(descriptors, list) or not isinstance(counts, dict):
        raise ValueError("Qwen3.8 vibe manifest descriptors/counts are missing")
    if manifest.get("action_descriptor_count") != len(descriptors):
        raise ValueError("Qwen3.8 vibe manifest action descriptor count changed")
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    descriptor_ids: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("Qwen3.8 vibe action descriptor must be an object")
        unsigned = dict(descriptor)
        descriptor_sha256 = unsigned.pop("descriptor_sha256", None)
        trajectory_order = unsigned.get("trajectory_order")
        if (
            not isinstance(trajectory_order, int)
            or isinstance(trajectory_order, bool)
            or descriptor_sha256 != _canonical_json_sha256(unsigned)
        ):
            raise ValueError("Qwen3.8 vibe action descriptor hash/order is invalid")
        descriptor_id = descriptor.get("descriptor_id")
        trajectory_id = descriptor.get("trajectory_id")
        if (
            not isinstance(descriptor_id, str)
            or descriptor_id in descriptor_ids
            or not isinstance(trajectory_id, str)
            or not trajectory_id
            or descriptor.get("is_final_serving_state") is not True
            or int(descriptor.get("completed_rounds", -1))
            != int(descriptor.get("evicted_rounds", -1))
            + args.vibe_working_memory_rounds
            or not 0 < int(descriptor.get("target_tokens", 0)) <= args.kl_tokens
        ):
            raise ValueError("Qwen3.8 vibe action descriptor semantics are invalid")
        descriptor_ids.add(descriptor_id)
        by_trajectory.setdefault(trajectory_id, []).append(dict(descriptor))
    eligible_trajectories = int(counts.get("eligible_trajectories", -1))
    if eligible_trajectories != len(by_trajectory) or eligible_trajectories <= 0:
        raise ValueError("Qwen3.8 vibe eligible trajectory count is inconsistent")
    if "trajectory_count" in manifest and int(manifest["trajectory_count"]) != (
        eligible_trajectories
    ):
        raise ValueError("Qwen3.8 vibe trajectory_count alias is inconsistent")

    final_student_usable = sum(
        int(item["student_sequence_tokens"]) <= args.max_target_sequence_tokens
        for item in descriptors
    )
    recurrent_source_usable = sum(
        int(item["max_recurrent_source_sequence_tokens"])
        <= args.max_target_sequence_tokens
        for item in descriptors
    )
    student_usable = sum(
        int(item["student_sequence_tokens"]) <= args.max_target_sequence_tokens
        and int(item["max_recurrent_source_sequence_tokens"])
        <= args.max_target_sequence_tokens
        for item in descriptors
    )
    teacher_usable = sum(
        int(item["teacher_sequence_tokens"]) <= args.max_teacher_sequence_tokens
        for item in descriptors
    )
    cap_safe_trajectory_ids = {
        str(item["trajectory_id"])
        for item in descriptors
        if int(item["student_sequence_tokens"]) <= args.max_target_sequence_tokens
        and int(item["max_recurrent_source_sequence_tokens"])
        <= args.max_target_sequence_tokens
        and int(item["teacher_sequence_tokens"]) <= args.max_teacher_sequence_tokens
    }
    cap_coverage = manifest.get("cap_coverage")
    locked_coverage = (
        cap_coverage.get("locked_geometry") if isinstance(cap_coverage, dict) else None
    )
    expected_locked_coverage = {
        "student_cap": args.max_target_sequence_tokens,
        "raw_teacher_cap": args.max_teacher_sequence_tokens,
        "cached_prefill_cap": args.teacher_prefill_chunk_tokens,
        "teacher_execution_mode": args.teacher_execution_mode,
        "cap_safe_trajectories": len(cap_safe_trajectory_ids),
        "trajectories_without_cap_safe_action": len(by_trajectory)
        - len(cap_safe_trajectory_ids),
        "student_usable": student_usable,
        "student_skipped": len(descriptors) - student_usable,
        "final_student_usable": final_student_usable,
        "final_student_skipped": len(descriptors) - final_student_usable,
        "recurrent_source_usable": recurrent_source_usable,
        "recurrent_source_skipped": len(descriptors) - recurrent_source_usable,
        "raw_teacher_usable": teacher_usable,
        "cached_prefill_usable": sum(
            int(item["cached_prefill_tokens"]) <= args.teacher_prefill_chunk_tokens
            for item in descriptors
        ),
    }
    expected_report_matrix = {
        str(cap): {
            "teacher": sum(
                int(item["teacher_sequence_tokens"]) <= cap for item in descriptors
            ),
            "student": sum(
                int(item["student_sequence_tokens"]) <= cap for item in descriptors
            ),
            "both": sum(
                int(item["teacher_sequence_tokens"]) <= cap
                and int(item["student_sequence_tokens"]) <= cap
                for item in descriptors
            ),
        }
        for cap in REPORT_SEQUENCE_CAPS
    }
    if (
        cap_coverage
        != {
            "report_matrix": expected_report_matrix,
            "locked_geometry": expected_locked_coverage,
        }
        or student_usable <= 0
    ):
        raise ValueError("Qwen3.8 vibe manifest cap coverage is invalid")
    manifest_kind = manifest.get("manifest_kind")
    canary_schedule_selection = manifest.get("canary_schedule_selection")
    if manifest_kind == "canary":
        if (
            not isinstance(canary_schedule_selection, dict)
            or int(canary_schedule_selection.get("seed", -1)) != args.seed
            or canary_schedule_selection.get("selector")
            != "progressive_monotone_depth_quantiles"
            or canary_schedule_selection.get("row_schedule") != [0, 0, 1]
            or not isinstance(canary_schedule_selection.get("epoch_schedule"), list)
            or len(canary_schedule_selection["epoch_schedule"]) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < args.epochs
                for value in canary_schedule_selection["epoch_schedule"]
            )
            or not isinstance(
                canary_schedule_selection.get("descriptor_schedule"), list
            )
            or len(canary_schedule_selection["descriptor_schedule"]) != 3
        ):
            raise ValueError("Qwen3.8 vibe canary schedule/seed differs from trainer")
        if (
            args.grad_accumulation != 1
            or args.max_steps not in {1, 2}
            or not args.verify_parameters_finite_after_step
        ):
            raise ValueError(
                "Qwen3.8 vibe canary manifest requires grad-accumulation=1, "
                "max-steps=1|2, and finite-parameter verification"
            )
    elif manifest_kind != "formal" or canary_schedule_selection is not None:
        raise ValueError("Qwen3.8 vibe manifest kind/canary receipt is invalid")
    frozen_files = manifest.get("files")
    if not isinstance(frozen_files, list) or not frozen_files:
        raise ValueError("Qwen3.8 vibe manifest has no frozen source files")
    receipt = {
        "name": QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT,
        "manifest": str(manifest_path.resolve()),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": manifest["canonical_sha256"],
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "manifest_kind": manifest_kind,
        "accepted_only": True,
        "parallel_tool_normalization": PARALLEL_NORMALIZATION,
        "final_serving_state_only": True,
        "working_memory_rounds": args.vibe_working_memory_rounds,
        "target_token_cap": args.kl_tokens,
        "student_sequence_cap": args.max_target_sequence_tokens,
        "raw_teacher_sequence_cap": args.max_teacher_sequence_tokens,
        "teacher_cached_prefill_chunk_tokens": args.teacher_prefill_chunk_tokens,
        "teacher_execution_mode": args.teacher_execution_mode,
        "trajectory_weighting": TRAJECTORY_WEIGHTING,
        "epoch_selector": "progressive_monotone_depth_quantiles",
        "generation_boundary_input_ids": list(QWEN38_GENERATION_BOUNDARY_IDS),
        "eligible_trajectories": eligible_trajectories,
        "action_descriptor_count": len(descriptors),
        "cap_coverage": manifest["cap_coverage"],
        "source_complete_rows": manifest.get("source_complete_rows"),
        "source_complete_bytes": manifest.get("source_complete_bytes"),
        "unique_ids_order_sha256": manifest.get("unique_ids_order_sha256"),
        "decontamination": manifest.get("decontamination"),
        "tokenizer": tokenizer_receipt,
        "adapter_source_sha256": adapter_source_sha256,
        "canary_probe_selection": manifest.get("canary_probe_selection"),
        "canary_schedule_selection": canary_schedule_selection,
    }
    return receipt, {
        trajectory_id: tuple(items) for trajectory_id, items in by_trajectory.items()
    }


def _validate_agentic_data_audit(
    args: argparse.Namespace,
    *,
    data_sha256: str,
) -> dict[str, Any]:
    """Bind a full CPU normalization audit to this exact data/template pair."""

    artifact_path = args.agentic_data_audit.expanduser().resolve()
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("agentic data audit is not valid JSON") from error
    if not isinstance(artifact, dict):
        raise ValueError("agentic data audit must be a JSON object")
    artifact_sha256 = _sha256(artifact_path)
    if artifact_sha256 != args.expected_agentic_audit_sha256:
        raise ValueError("agentic data audit artifact SHA256 mismatch")
    if (
        artifact.get("format_version") != QWEN36_AGENTIC_AUDIT_FORMAT_VERSION
        or artifact.get("schema") != QWEN36_AGENTIC_AUDIT_SCHEMA
    ):
        raise ValueError("agentic data audit schema/version mismatch")
    recorded_contract_sha256 = artifact.get("audit_contract_sha256")
    unsigned = dict(artifact)
    unsigned.pop("audit_contract_sha256", None)
    if recorded_contract_sha256 != _canonical_json_sha256(unsigned):
        raise ValueError("agentic data audit contract SHA256 mismatch")
    if recorded_contract_sha256 != args.expected_agentic_audit_contract_sha256:
        raise ValueError("agentic data audit is not the pinned audit contract")
    if Path(str(artifact.get("data", ""))).resolve() != args.data.resolve():
        raise ValueError("agentic data audit points at a different parquet")
    if artifact.get("data_sha256") != data_sha256:
        raise ValueError("agentic data audit parquet SHA256 mismatch")
    if (
        Path(str(artifact.get("tokenizer", ""))).resolve()
        != args.target_model.resolve()
    ):
        raise ValueError("agentic data audit points at a different tokenizer")
    template_path = args.target_model.resolve() / "chat_template.jinja"
    if not template_path.is_file():
        template_path = args.target_model.resolve() / "tokenizer_config.json"
    if artifact.get("chat_template_sha256") != _sha256(template_path):
        raise ValueError("agentic data audit chat-template SHA256 mismatch")
    adapter_path = Path(__file__).resolve().parent / "qwen36_agentic_data.py"
    if artifact.get("adapter_source_sha256") != _sha256(adapter_path):
        raise ValueError("agentic data audit adapter source SHA256 mismatch")
    audit_source_path = Path(__file__).resolve().parent / "audit_qwen36_agentic_data.py"
    if artifact.get("audit_source_sha256") != _sha256(audit_source_path):
        raise ValueError("agentic data audit implementation SHA256 mismatch")
    if artifact.get("target_cap") != args.kl_tokens:
        raise ValueError("agentic data audit KL target cap mismatch")
    if artifact.get("block_tokens") != EVICTION_BLOCK_TOKENS:
        raise ValueError("agentic data audit eviction block width mismatch")
    if artifact.get("compression_ratio") != COMPRESSION_RATIO:
        raise ValueError("agentic data audit compression ratio mismatch")
    audit_config = artifact.get("audit_config")
    if not isinstance(audit_config, dict) or (
        audit_config.get("target_cap") != args.kl_tokens
        or audit_config.get("block_tokens") != EVICTION_BLOCK_TOKENS
        or audit_config.get("compression_ratio") != COMPRESSION_RATIO
        or 12_288 not in audit_config.get("sequence_caps", [])
    ):
        raise ValueError("agentic data audit geometry/config mismatch")

    expected = {
        "rows": int(args.expected_agentic_total_rows),
        "valid": int(args.expected_agentic_valid_rows),
        "rejected": int(args.expected_agentic_rejected_rows),
    }
    counters = artifact.get("counters")
    if not isinstance(counters, dict) or any(
        counters.get(name) != value for name, value in expected.items()
    ):
        raise ValueError(
            f"agentic data audit row counts differ: expected {expected}, got {counters}"
        )
    if artifact.get("range") != [0, expected["rows"]]:
        raise ValueError("agentic data audit does not cover the complete parquet")
    expected_reject_categories = _parse_expected_reject_categories(
        args.expected_agentic_reject_category
    )
    if artifact.get("reject_categories") != expected_reject_categories:
        raise ValueError(
            "agentic data audit reject policy changed: expected only "
            f"{expected_reject_categories}"
        )
    reject_rows = artifact.get("reject_rows")
    if not isinstance(reject_rows, list) or len(reject_rows) != expected["rejected"]:
        raise ValueError("agentic data audit reject-row receipt count mismatch")
    if artifact.get("reject_rows_sha256") != _canonical_json_sha256(reject_rows):
        raise ValueError("agentic data audit reject-row receipt SHA256 mismatch")
    if artifact["reject_rows_sha256"] != args.expected_agentic_reject_rows_sha256:
        raise ValueError("agentic data audit reject-row set is not pinned")
    cap_stats = artifact.get("caps", {}).get("12288")
    if not isinstance(cap_stats, dict):
        raise ValueError("agentic data audit has no 12288 cap statistics")
    cap_stats_sha256 = _canonical_json_sha256(cap_stats)
    observed_cap_hashes = artifact.get("cap_stats_sha256")
    if not isinstance(observed_cap_hashes, dict) or observed_cap_hashes != {
        str(cap): _canonical_json_sha256(stats)
        for cap, stats in artifact.get("caps", {}).items()
    }:
        raise ValueError("agentic data audit cap-stat hashes are invalid")
    if observed_cap_hashes.get("12288") != cap_stats_sha256:
        raise ValueError("agentic data audit 12288 cap-stat hash is invalid")
    if cap_stats_sha256 != args.expected_agentic_12288_cap_stats_sha256:
        raise ValueError("agentic data audit 12288 cap statistics are not pinned")
    sensitivity = artifact.get("target_cap_sensitivity")
    if not isinstance(sensitivity, dict) or artifact.get(
        "target_cap_sensitivity_sha256"
    ) != _canonical_json_sha256(sensitivity):
        raise ValueError("agentic data audit target-cap sensitivity hash is invalid")
    return {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "audit_contract_sha256": recorded_contract_sha256,
        "schema": QWEN36_AGENTIC_AUDIT_SCHEMA,
        "format_version": QWEN36_AGENTIC_AUDIT_FORMAT_VERSION,
        "range": artifact["range"],
        "counters": expected,
        "reject_categories": expected_reject_categories,
        "reject_rows_sha256": artifact["reject_rows_sha256"],
        "target_cap": artifact["target_cap"],
        "adapter_source_sha256": artifact["adapter_source_sha256"],
        "audit_source_sha256": artifact["audit_source_sha256"],
        "block_tokens": artifact["block_tokens"],
        "compression_ratio": artifact["compression_ratio"],
        "cap_12288_stats": cap_stats,
        "cap_12288_stats_sha256": cap_stats_sha256,
        "target_cap_sensitivity_sha256": artifact["target_cap_sensitivity_sha256"],
        "selected_target_cap_stats": sensitivity.get(str(args.kl_tokens)),
    }


def _validate_mindforge_manifest_training_contract(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    args: argparse.Namespace,
    target_content_identity: dict[str, Any],
    adapter_source_sha256: str,
    tokenizer: Any,
    init_compressor_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Bind the generic v1 manifest to this exact tokenizer/training policy."""

    expected_manifest_schema = MINDFORGE_DECISION_MANIFEST_SCHEMA
    if args.mindforge_pilot_profile == MINDFORGE_FULL_VIEW_H4096_PROFILE:
        expected_manifest_schema = MINDFORGE_DECISION_MANIFEST_VIEW_SCHEMA
    if (
        manifest.get("schema") != expected_manifest_schema
        or manifest.get("format_version") != MINDFORGE_DECISION_MANIFEST_FORMAT_VERSION
    ):
        raise ValueError("MindForge decision manifest schema/version mismatch")
    manifest_artifact_sha256 = _sha256(manifest_path)
    if manifest_artifact_sha256 != args.expected_mindforge_manifest_sha256:
        raise ValueError("MindForge decision manifest artifact SHA256 mismatch")
    if args.mindforge_pilot_profile == MINDFORGE_FULL_VIEW_H4096_PROFILE and (
        manifest_artifact_sha256 != MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256
        or manifest.get("contract_sha256") != MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256
    ):
        raise ValueError("MindForge 4K full-view manifest lineage mismatch")
    source_target_identity = init_compressor_receipt.get(
        "source_target_content_identity"
    )
    if (
        not isinstance(source_target_identity, dict)
        or source_target_identity != target_content_identity
        or init_compressor_receipt.get("source_target_content_identity_sha256")
        != _canonical_json_sha256(target_content_identity)
    ):
        raise ValueError(
            "init-compressor was trained against a different frozen target identity"
        )
    from context_compression.mindforge_ce_manifest import tokenizer_receipt

    observed_tokenizer_receipt = manifest.get("tokenizer")
    expected_tokenizer = tokenizer_receipt(tokenizer, args.target_model)
    if observed_tokenizer_receipt != expected_tokenizer:
        raise ValueError(
            "MindForge decision manifest tokenizer receipt differs from this target"
        )
    expected_checkpoint_fields = {
        "resolved_path": str(args.init_compressor.resolve().parent),
        "manifest_sha256": init_compressor_receipt["manifest_sha256"],
        "config_sha256": init_compressor_receipt["config_sha256"],
        "config_file_sha256": init_compressor_receipt["config_file_sha256"],
        "compressor_sha256": init_compressor_receipt["artifact_sha256"],
        "compressor_size_bytes": init_compressor_receipt["artifact_size_bytes"],
        "export_schema": init_compressor_receipt["export_schema"],
        "source_global_step": init_compressor_receipt["source_global_step"],
        "source_curriculum_stage": init_compressor_receipt["source_curriculum_stage"],
        "source_target_content_identity": init_compressor_receipt[
            "source_target_content_identity"
        ],
        "source_target_content_identity_sha256": init_compressor_receipt[
            "source_target_content_identity_sha256"
        ],
    }
    expected_checkpoint = {
        **expected_checkpoint_fields,
        "receipt_sha256": _canonical_json_sha256(expected_checkpoint_fields),
    }
    if manifest.get("checkpoint") != expected_checkpoint:
        raise ValueError(
            "MindForge decision manifest checkpoint differs from init-compressor"
        )
    expected_adapter = {
        "contract": MINDFORGE_CONTRACT,
        "implementation_sha256": adapter_source_sha256,
    }
    if manifest.get("adapter") != expected_adapter:
        raise ValueError("MindForge decision manifest adapter receipt differs")
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("MindForge decision manifest policy fields differ")
    if isinstance(policy.get("selection_seed"), bool) or not isinstance(
        policy.get("selection_seed"), int
    ):
        raise ValueError("MindForge manifest selection_seed must be a plain int")
    expected_policy = {
        "history_window_tokens": args.mindforge_history_window_tokens,
        "max_action_tokens": args.mindforge_max_action_tokens,
        "max_student_sequence_tokens": args.max_target_sequence_tokens,
        "eviction_block_tokens": EVICTION_BLOCK_TOKENS,
        "final_working_memory_tokens": RAW_WORKING_MEMORY_TOKENS,
        "compression_ratio": COMPRESSION_RATIO,
        "history_before_window_excluded": True,
        "initial_memory": None,
        "long_trajectory_position_used_for_action_selection_only": True,
        "actions_per_train_trajectory": (MINDFORGE_SELECTED_ACTIONS_PER_TRAJECTORY),
        "action_truncation": "forbidden",
        "action_weighting": "unit_trajectory_then_uniform_selected_action",
        "compression_cutpoint_weighting": "uniform_1_over_K_within_action",
    }
    observed_policy = {name: policy.get(name) for name in expected_policy}
    if observed_policy != expected_policy:
        raise ValueError(
            "MindForge decision manifest policy differs from trainer arguments"
        )

    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("MindForge decision manifest has no rows")
    train_rows = [
        row for row in rows if isinstance(row, dict) and row.get("split") == "train"
    ]
    if not train_rows:
        raise ValueError("MindForge decision manifest has no train rows")
    expected_counts = {
        "train_trajectories": MINDFORGE_PRIMARY_TRAIN_TRAJECTORIES,
        "train_actions": (
            MINDFORGE_PRIMARY_TRAIN_TRAJECTORIES
            * MINDFORGE_SELECTED_ACTIONS_PER_TRAJECTORY
        ),
        "validation_trajectories": MINDFORGE_PRIMARY_VALIDATION_TRAJECTORIES,
        "validation_actions": (
            MINDFORGE_PRIMARY_VALIDATION_TRAJECTORIES
            * MINDFORGE_PRIMARY_VALIDATION_ACTIONS_PER_TRAJECTORY
        ),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("MindForge production manifest counts are not 512x4+16x2")
    selected_event_instance_ids: set[str] = set()
    selected_event_ids: set[str] = set()
    event_count = 0
    for order, row in enumerate(train_rows):
        if row.get("training_order") != order:
            raise ValueError("MindForge training_order must be contiguous and exact")
        events = row.get("events")
        if not isinstance(events, list) or len(events) != (
            MINDFORGE_SELECTED_ACTIONS_PER_TRAJECTORY
        ):
            raise ValueError("MindForge pilot requires four selected actions per row")
        cutpoints = [
            event.get("cutpoint_index") for event in events if isinstance(event, dict)
        ]
        if len(cutpoints) != len(events) or cutpoints != sorted(set(cutpoints)):
            raise ValueError("MindForge selected cutpoints must be unique and ordered")
        expected_action_weight = 1.0 / float(len(events))
        row_event_ids: set[str] = set()
        for event in events:
            event_id = event.get("event_id")
            instance_id = event.get("selected_event_instance_id")
            if not isinstance(event_id, str) or event_id in row_event_ids:
                raise ValueError("MindForge manifest event IDs must be unique per row")
            if event_id in selected_event_ids:
                raise ValueError("MindForge manifest event IDs must be globally unique")
            if (
                not isinstance(instance_id, str)
                or instance_id in selected_event_instance_ids
            ):
                raise ValueError(
                    "MindForge selected event instance IDs must be globally unique"
                )
            row_event_ids.add(event_id)
            selected_event_ids.add(event_id)
            selected_event_instance_ids.add(instance_id)
            weight = event.get("selected_event_weight")
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isclose(
                    float(weight),
                    expected_action_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("MindForge selected action weights must sum per row")
            target_tokens = event.get("target_tokens")
            if (
                isinstance(target_tokens, bool)
                or not isinstance(target_tokens, int)
                or target_tokens <= 0
                or target_tokens > args.mindforge_max_action_tokens
            ):
                raise ValueError("MindForge manifest contains an over-cap full action")
            event_count += 1
    contract_sha256 = manifest.get("contract_sha256")
    unsigned = dict(manifest)
    unsigned.pop("contract_sha256", None)
    if contract_sha256 != _canonical_json_sha256(unsigned):
        raise ValueError("MindForge decision manifest contract SHA256 mismatch")
    if contract_sha256 != args.expected_mindforge_manifest_contract_sha256:
        raise ValueError("MindForge decision manifest is not the CLI-pinned contract")
    return {
        "path": str(manifest_path.resolve()),
        "artifact_sha256": manifest_artifact_sha256,
        "schema": expected_manifest_schema,
        "format_version": MINDFORGE_DECISION_MANIFEST_FORMAT_VERSION,
        "contract_sha256": contract_sha256,
        "source": manifest["source"],
        "tokenizer": observed_tokenizer_receipt,
        "checkpoint": manifest["checkpoint"],
        "adapter": manifest["adapter"],
        "policy": policy,
        "adapter_contract": MINDFORGE_CONTRACT,
        "adapter_source_sha256": adapter_source_sha256,
        "train_trajectories": len(train_rows),
        "selected_actions": event_count,
        "selected_actions_per_trajectory": MINDFORGE_SELECTED_ACTIONS_PER_TRAJECTORY,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required package metadata is missing: {name}") from error


def _module_origin(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"required module origin is missing: {name}")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"required module origin is not a file: {path}")
    return path


def _implementation_identity(
    data_contract: str = LEGACY_TOKEN_CLOCK_DATA_CONTRACT,
) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    local_files = {
        "trainer": Path(__file__).resolve(),
        "data": source_root / "qwen36_recursive_data.py",
        "compressor": source_root / "grepseek_modeling.py",
        "loss": source_root / "losses.py",
        "parity_validator": source_root / "qwen36_teacher_parity.py",
        "target_loader": source_root / "qwen35_text_target.py",
    }
    if data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
        local_files["agentic_data"] = source_root / "qwen36_agentic_data.py"
        local_files["agentic_auditor"] = source_root / "audit_qwen36_agentic_data.py"
    elif data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
        local_files["mindforge_data"] = source_root / "mindforge_glm52_data.py"
        local_files["mindforge_manifest"] = source_root / "mindforge_ce_manifest.py"
    elif data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        local_files["vibe_trajectory_data"] = (
            source_root / "qwen38_coding_trajectory_data.py"
        )
        # WM4 portable-export provenance imports these at runtime, before the
        # fresh optimizer/model training path is entered.  Keep those direct
        # local imports in the same implementation identity as the trainer.
        local_files["context_compression_package"] = source_root / "__init__.py"
        local_files["mindforge_manifest"] = source_root / "mindforge_ce_manifest.py"
        local_files["specforge_decontamination"] = (
            source_root.parent / "specforge" / "data" / "decontamination.py"
        )
    elif data_contract != LEGACY_TOKEN_CLOCK_DATA_CONTRACT:
        raise ValueError(f"unsupported data contract: {data_contract}")
    dependency_files = {
        "torch_init": _module_origin("torch"),
        "torch_c_extension": _module_origin("torch._C"),
        "torch_attention_bias": _module_origin("torch.nn.attention.bias"),
        "transformers_qwen3_5": Path(qwen3_5_modeling.__file__).resolve(),
        "transformers_qwen3_5_moe": Path(qwen3_5_moe_modeling.__file__).resolve(),
        "transformers_cache_utils": _module_origin("transformers.cache_utils"),
        "transformers_masking_utils": _module_origin("transformers.masking_utils"),
        "transformers_sdpa_attention": _module_origin(
            "transformers.integrations.sdpa_attention"
        ),
        "causal_conv1d_interface": _module_origin(
            "causal_conv1d.causal_conv1d_interface"
        ),
        "causal_conv1d_cuda": _module_origin("causal_conv1d_cuda"),
        "fla_gated_delta_chunk": _module_origin("fla.ops.gated_delta_rule.chunk"),
        "fla_gated_delta_chunk_fwd": _module_origin(
            "fla.ops.gated_delta_rule.chunk_fwd"
        ),
        "fla_gated_delta_fused_recurrent": _module_origin(
            "fla.ops.gated_delta_rule.fused_recurrent"
        ),
        "fla_common_chunk_delta_h": _module_origin("fla.ops.common.chunk_delta_h"),
    }
    for path in (*local_files.values(), *dependency_files.values()):
        if not path.is_file():
            raise RuntimeError(f"implementation dependency is missing: {path}")
    return {
        "python": sys.version.split()[0],
        "qwen35_cached_full_attention_patch": QWEN35_CACHED_FULL_ATTENTION_PATCH,
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "causal-conv1d": _package_version("causal-conv1d"),
            "flash-linear-attention": _package_version("flash-linear-attention"),
            "triton": _package_version("triton"),
        },
        "torch_build": {
            "git_version": torch.version.git_version,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": list(torch.cuda.nccl.version()),
            "device_name": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
        },
        "local_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(local_files.items())
        },
        "dependency_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(dependency_files.items())
        },
    }


def _cached_teacher_implementation_projection(
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Project historical parity onto the byte-exact cached-runtime pieces."""

    local_files = identity.get("local_files")
    if not isinstance(local_files, dict):
        raise ValueError("cached-teacher implementation identity has no local files")
    required_local = ("compressor", "loss", "parity_validator", "target_loader")
    if any(name not in local_files for name in required_local):
        raise ValueError("cached-teacher implementation identity is incomplete")
    return {
        name: identity.get(name)
        for name in (
            "python",
            "qwen35_cached_full_attention_patch",
            "packages",
            "torch_build",
            "dependency_files",
        )
    } | {"local_files": {name: local_files[name] for name in required_local}}


def _validate_cross_dataset_cached_teacher_parity_artifacts(
    artifacts: Sequence[Path],
    *,
    required_chunk_tokens: Sequence[int],
    target_model: Path,
    target_manifest_sha256: str,
    target_content_identity: dict[str, Any],
    implementation_identity: dict[str, Any],
    maximum_output_distribution_kl: float,
    minimum_top1_agreement: float,
) -> list[dict[str, Any]]:
    """Validate old probes as implementation-only, never as new-data evidence."""

    source_headers: list[dict[str, Any]] = []
    for path in artifacts:
        resolved = path.expanduser().resolve()
        try:
            artifact = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "cached-teacher parity artifact is not valid JSON"
            ) from error
        if not isinstance(artifact, dict):
            raise ValueError("cached-teacher parity artifact must be a JSON object")
        source_headers.append(
            {
                "data": artifact.get("data"),
                "data_sha256": artifact.get("data_sha256"),
                "implementation_identity": artifact.get("implementation_identity"),
            }
        )
    if not source_headers or any(item != source_headers[0] for item in source_headers):
        raise ValueError("cached-teacher parity artifacts have different source scopes")
    source = source_headers[0]
    if (
        not isinstance(source["data"], str)
        or not isinstance(source["data_sha256"], str)
        or not isinstance(source["implementation_identity"], dict)
    ):
        raise ValueError("cached-teacher parity source scope is incomplete")
    historical_projection = _cached_teacher_implementation_projection(
        source["implementation_identity"]
    )
    current_projection = _cached_teacher_implementation_projection(
        implementation_identity
    )
    if historical_projection != current_projection:
        raise ValueError(
            "cached-teacher implementation-only evidence differs from the current "
            "runtime projection"
        )
    from context_compression.qwen36_teacher_parity import (
        validate_cached_teacher_parity_artifacts,
    )

    receipts = validate_cached_teacher_parity_artifacts(
        artifacts,
        required_chunk_tokens=required_chunk_tokens,
        data_path=Path(source["data"]),
        data_sha256=source["data_sha256"],
        target_model=target_model,
        target_manifest_sha256=target_manifest_sha256,
        target_content_identity=target_content_identity,
        implementation_identity=source["implementation_identity"],
        maximum_output_distribution_kl=maximum_output_distribution_kl,
        minimum_top1_agreement=minimum_top1_agreement,
    )
    return [
        {
            **receipt,
            "parity_evidence_scope": "implementation_only_cross_dataset",
            "source_data": source["data"],
            "source_data_sha256": source["data_sha256"],
            "current_data_covered": False,
            "excluded_identity_fields": [
                "local_files.trainer",
                "local_files.data",
                "contract_specific_adapter_files",
            ],
            "implementation_projection_sha256": _canonical_json_sha256(
                current_projection
            ),
        }
        for receipt in receipts
    ]


def _target_content_identity(root: Path) -> dict[str, Any]:
    """Hash every target weight plus the model/tokenizer metadata used here."""

    resolved = root.expanduser().resolve()
    weight_paths = sorted(
        {
            candidate
            for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
            for candidate in resolved.glob(pattern)
            if candidate.is_file()
        }
    )
    if not weight_paths:
        raise FileNotFoundError(f"target checkpoint contains no weights: {resolved}")
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "text_snapshot_manifest.json",
    )
    metadata_paths = [resolved / name for name in metadata_names]
    missing = [path.name for path in metadata_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "target checkpoint metadata is incomplete: " + ", ".join(missing)
        )
    weights = [{"name": path.name, "sha256": _sha256(path)} for path in weight_paths]
    metadata = [{"name": path.name, "sha256": _sha256(path)} for path in metadata_paths]

    def canonical_sha(value: Any) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    identity = {
        "resolved_path": str(resolved),
        "weights": weights,
        "metadata": metadata,
        "weight_content_sha256": canonical_sha(weights),
        "metadata_content_sha256": canonical_sha(metadata),
    }
    return {**identity, "target_content_sha256": canonical_sha(identity)}


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return {key: convert(value) for key, value in vars(args).items()}


def _run_contract_sha256(run_contract: dict[str, Any]) -> str:
    return _canonical_json_sha256(run_contract)


def _wm4_resume_semantic_projection(run_contract: dict[str, Any]) -> dict[str, Any]:
    """Return the closed semantic projection used only for WM4 checkpoint reload.

    Generic checkpoints retain their historical full run-contract equality
    rule. The WM4 A/B/C transition is special only because the pre-load
    compressor object reports a target-derived output RMS before the saved
    persistent buffer replaces it. No model, data, initialization,
    optimization, curriculum, selector, or implementation property may
    change with that transition.
    """

    if not isinstance(run_contract, dict):
        raise ValueError("WM4 resume run contract must be an object")
    if run_contract.get("data_contract", {}).get("name") != (
        QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
    ):
        raise ValueError("WM4 resume projection requires the WM4 data contract")
    if "compressor_initialization" not in run_contract:
        raise ValueError("WM4 resume run contract has no initialization receipt")
    compressor = run_contract.get("compressor")
    if not isinstance(compressor, dict) or set(compressor) != (
        WM4_RESUME_COMPRESSOR_ARCHITECTURE_KEYS
    ):
        raise ValueError("WM4 resume compressor config has unexpected keys")
    architecture = {
        key: value
        for key, value in compressor.items()
        if key != "output_embedding_rms"
    }
    return {
        "schema": WM4_RESUME_BINDING_SCHEMA,
        "allowed_run_contract_delta_keys": sorted(
            WM4_RESUME_ALLOWED_RUN_CONTRACT_DELTA_KEYS
        ),
        "semantic_run_contract": {
            key: value
            for key, value in run_contract.items()
            if key not in WM4_RESUME_ALLOWED_RUN_CONTRACT_DELTA_KEYS
        }
        | {"compressor": architecture},
    }


def _wm4_resume_operational_controls(args: argparse.Namespace) -> dict[str, Any]:
    """Make the finite WM4 resume-control allowlist explicit and auditable."""

    controls = {
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "resume_from": (
            str(args.resume_from.expanduser().resolve())
            if args.resume_from is not None
            else None
        ),
        "init_compressor": (
            str(args.init_compressor.expanduser().resolve())
            if args.init_compressor is not None
            else None
        ),
        "max_steps": args.max_steps,
        "save_interval": args.save_interval,
        "skip_final_checkpoint": args.skip_final_checkpoint,
        "log_interval": args.log_interval,
        "verify_parameters_finite_after_step": (
            args.verify_parameters_finite_after_step
        ),
    }
    if set(controls) != WM4_RESUME_OPERATIONAL_CONTROL_KEYS:
        raise RuntimeError("WM4 resume operational-control closure changed")
    return controls


def _validate_wm4_resume_operational_controls(
    controls: Any,
    *,
    description: str,
) -> dict[str, Any]:
    if not isinstance(controls, dict) or set(controls) != (
        WM4_RESUME_OPERATIONAL_CONTROL_KEYS
    ):
        raise ValueError(f"{description} WM4 operational-control keys differ")
    for name in ("output_dir",):
        value = controls.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{description} WM4 {name} is not an absolute path")
    for name in ("resume_from", "init_compressor"):
        value = controls.get(name)
        if value is not None and (
            not isinstance(value, str) or not Path(value).is_absolute()
        ):
            raise ValueError(f"{description} WM4 {name} is invalid")
    if (controls["resume_from"] is None) == (controls["init_compressor"] is None):
        raise ValueError(
            f"{description} WM4 controls require exactly one initialization source"
        )
    for name in ("max_steps", "save_interval", "log_interval"):
        value = controls.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{description} WM4 {name} is not an integer")
    if controls["max_steps"] == 0 or controls["max_steps"] < -1:
        raise ValueError(f"{description} WM4 max_steps is invalid")
    if controls["save_interval"] < 0 or controls["log_interval"] <= 0:
        raise ValueError(f"{description} WM4 save/log control is invalid")
    for name in ("skip_final_checkpoint", "verify_parameters_finite_after_step"):
        if not isinstance(controls.get(name), bool):
            raise ValueError(f"{description} WM4 {name} is not boolean")
    return dict(controls)


def _wm4_resume_binding(
    run_contract: dict[str, Any],
    *,
    operational_controls: dict[str, Any],
) -> dict[str, Any]:
    """Bind a WM4 checkpoint to semantic identity and closed runtime controls."""

    controls = _validate_wm4_resume_operational_controls(
        operational_controls, description="current"
    )
    projection = _wm4_resume_semantic_projection(run_contract)
    return {
        "schema": WM4_RESUME_BINDING_SCHEMA,
        "semantic_contract_sha256": _canonical_json_sha256(projection),
        "operational_controls": controls,
        "operational_controls_sha256": _canonical_json_sha256(controls),
    }


def _validate_wm4_resume_binding(
    checkpoint: Path,
    *,
    source_run_contract_sha256: str,
    source_binding: Any,
    current_run_contract: dict[str, Any],
    current_binding: dict[str, Any],
) -> None:
    """Validate the one exact WM4 transition before model/optimizer loading."""

    if not isinstance(source_binding, dict) or set(source_binding) != {
        "schema",
        "semantic_contract_sha256",
        "operational_controls",
        "operational_controls_sha256",
    }:
        raise ValueError("WM4 checkpoint has no closed resume binding")
    if source_binding.get("schema") != WM4_RESUME_BINDING_SCHEMA:
        raise ValueError("WM4 checkpoint resume binding schema differs")
    source_semantic = source_binding.get("semantic_contract_sha256")
    current_semantic = current_binding.get("semantic_contract_sha256")
    if (
        not isinstance(source_semantic, str)
        or not isinstance(current_semantic, str)
        or source_semantic != current_semantic
    ):
        raise ValueError("WM4 checkpoint semantic contract does not match this invocation")
    if current_semantic != _canonical_json_sha256(
        _wm4_resume_semantic_projection(current_run_contract)
    ):
        raise ValueError("current WM4 semantic contract self-hash mismatch")
    source_controls = _validate_wm4_resume_operational_controls(
        source_binding.get("operational_controls"), description="checkpoint source"
    )
    if source_binding.get("operational_controls_sha256") != _canonical_json_sha256(
        source_controls
    ):
        raise ValueError("WM4 checkpoint operational-control self-hash mismatch")
    current_controls = _validate_wm4_resume_operational_controls(
        current_binding.get("operational_controls"), description="current"
    )
    if current_binding.get("schema") != WM4_RESUME_BINDING_SCHEMA or (
        current_binding.get("operational_controls_sha256")
        != _canonical_json_sha256(current_controls)
    ):
        raise ValueError("current WM4 resume binding is malformed")
    if current_controls["resume_from"] != str(checkpoint.resolve()):
        raise ValueError("WM4 resume control does not bind the loaded checkpoint")
    if current_controls["init_compressor"] is not None:
        raise ValueError("WM4 resumed invocation must not carry portable init weights")
    if current_controls["output_dir"] == source_controls["output_dir"]:
        raise ValueError("WM4 resumed output must be fresh and distinct")
    if checkpoint.parent.resolve() != Path(source_controls["output_dir"]).resolve():
        raise ValueError("WM4 checkpoint does not belong to its source output")
    source_config_path = checkpoint.parent / "run_config.json"
    source_config = _read_json_object(
        source_config_path, description="WM4 checkpoint source run config"
    )
    if _validate_run_config_self_hash(
        source_config, description="WM4 checkpoint source run config"
    ) != source_run_contract_sha256:
        raise ValueError("WM4 checkpoint source run-contract differs from its config")
    source_contract = _contract_from_run_config(source_config)
    if source_config.get("wm4_resume_binding") != source_binding:
        raise ValueError("WM4 checkpoint source binding differs from its config")
    if source_binding.get("semantic_contract_sha256") != _canonical_json_sha256(
        _wm4_resume_semantic_projection(source_contract)
    ):
        raise ValueError("WM4 checkpoint source semantic self-hash mismatch")
    changed_keys = {
        key
        for key in set(source_contract).union(current_run_contract)
        if source_contract.get(key) != current_run_contract.get(key)
    }
    if changed_keys - WM4_RESUME_ALLOWED_RUN_CONTRACT_DELTA_KEYS:
        raise ValueError(
            "WM4 checkpoint changed training-contract keys outside the exact "
            f"output-RMS transition: {sorted(changed_keys)}"
        )


def _checkpoint_configs() -> tuple[ShardedStateDictConfig, ShardedOptimStateDictConfig]:
    return (
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def _checkpoint_cuda_cache_barrier() -> None:
    """Collectively release inactive CUDA allocator blocks before save work.

    ``ShardedOptimStateDictConfig(offload_to_cpu=True)`` only controls the
    returned state.  FSDP1 still all-gathers one full flat optimizer tensor on
    the compute device before re-sharding it and moving the result to CPU.  A
    long training step leaves large, *inactive* activation blocks reserved by
    the CUDA caching allocator; retaining those blocks alongside FSDP's
    transient all-gather needlessly raises the device-level checkpoint peak.

    Synchronizing first makes the allocator/free boundary explicit.  The
    surrounding barriers keep both ranks on the same checkpoint phase.  This
    does not change live tensors, gradients, optimizer values, RNG, or the
    sharded checkpoint representation.
    """

    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    dist.barrier()


def _save_checkpoint(
    output_dir: Path,
    *,
    compressor: FSDP,
    optimizer: torch.optim.Optimizer,
    rank: int,
    global_step: int,
    next_epoch: int,
    next_line_index: int,
    run_contract_sha256: str,
    checkpoint_schema: str,
    wm4_resume_binding: dict[str, Any] | None = None,
    wm4_low_peak_moment_residency: "WM4LowPeakAdamMomentResidency | None" = None,
) -> Path:
    destination = output_dir / f"checkpoint-{global_step:08d}"
    staging = output_dir / f".{destination.name}.staging"
    if rank == 0:
        if destination.exists() or staging.exists():
            raise FileExistsError(
                f"checkpoint destination already exists: {destination}"
            )
        staging.mkdir(parents=False)
    # Reclaim cached activation blocks before FSDP creates its transient full
    # flat optimizer-state gather.  The first barrier also publishes staging.
    _checkpoint_cuda_cache_barrier()
    try:
        if wm4_low_peak_moment_residency is not None:
            # Do *not* restore the two complete AdamW moments here.  Pinned
            # PyTorch's FSDP sharded optimizer exporter accepts CPU local
            # state and CPU-offloads its exported shard.  Its internal
            # conversion can still materialize GPU buffers for both states,
            # so v5 makes B-save telemetry a hard peak gate.  Keeping the
            # moments off-device still avoids retaining them during the prior
            # model-state export/cache-cleanup phase; it does not claim to
            # eliminate the exporter's own transient conversion peak.
            # Verify the invariant before crossing the exporter boundary.
            wm4_low_peak_moment_residency.assert_cpu_resident_at_forward_entry(
                optimizer, record=False
            )
        state_config, optim_config = _checkpoint_configs()
        with FSDP.state_dict_type(
            compressor,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
            optim_config,
        ):
            model_state = compressor.state_dict()
            # Model shards are already CPU-offloaded at this point, so any
            # temporary model-state CUDA blocks may be returned before optimizer
            # conversion starts.
            _checkpoint_cuda_cache_barrier()
            optimizer_state = FSDP.optim_state_dict(compressor, optimizer)
            # The returned optimizer shards are CPU-offloaded too.  Do not retain
            # FSDP's full-flat all-gather buffer while torch.save writes to disk.
            _checkpoint_cuda_cache_barrier()
    finally:
        if wm4_low_peak_moment_residency is not None:
            # This is idempotent if FSDP only used temporary copies.  It also
            # converges mixed state should an exporter error escape midway.
            wm4_low_peak_moment_residency.offload_after_optimizer_use(
                optimizer, reason="after_save"
            )
            _checkpoint_cuda_cache_barrier()
    rank_payload = {
        "format_version": FORMAT_VERSION,
        "schema": checkpoint_schema,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "global_step": global_step,
        "run_contract_sha256": run_contract_sha256,
        "model": model_state,
        "optimizer": optimizer_state,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        **(
            {"wm4_resume_binding": wm4_resume_binding}
            if wm4_resume_binding is not None
            else {}
        ),
    }
    rank_file = staging / f"rank-{rank:05d}.pt"
    torch.save(rank_payload, rank_file)
    dist.barrier()
    if rank == 0:
        trainer_state = {
            "format_version": FORMAT_VERSION,
            "schema": checkpoint_schema,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_line_index": next_line_index,
            "run_contract_sha256": run_contract_sha256,
            "rank_files": [
                f"rank-{index:05d}.pt" for index in range(dist.get_world_size())
            ],
            **(
                {"wm4_resume_binding": wm4_resume_binding}
                if wm4_resume_binding is not None
                else {}
            ),
        }
        _atomic_json(staging / "trainer_state.json", trainer_state)
        os.replace(staging, destination)
    dist.barrier()
    return destination


def _wm4_resume_checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Return the exact B checkpoint bytes that a WM4 C load is allowed to use."""

    resolved = checkpoint.expanduser().resolve()
    names = ("trainer_state.json", "rank-00000.pt", "rank-00001.pt")
    files: list[dict[str, Any]] = []
    for name in names:
        path = resolved / name
        if not path.is_file():
            raise FileNotFoundError(f"WM4 resume checkpoint is missing {name}")
        status = path.stat()
        files.append(
            {
                "name": name,
                "sha256": _sha256(path),
                "size_bytes": status.st_size,
            }
        )
    return {"checkpoint": str(resolved), "files": files}


def _verify_wm4_resume_checkpoint_identity(
    checkpoint: Path, expected: Any
) -> dict[str, Any]:
    if not isinstance(expected, dict) or set(expected) != {"checkpoint", "files"}:
        raise ValueError("WM4 resume checkpoint identity is malformed")
    observed = _wm4_resume_checkpoint_identity(checkpoint)
    if observed != expected:
        raise ValueError("WM4 resume checkpoint bytes changed during load")
    return observed


def _load_checkpoint(
    checkpoint: Path,
    *,
    compressor: FSDP,
    optimizer: torch.optim.Optimizer,
    rank: int,
    run_contract_sha256: str,
    checkpoint_schema: str,
    wm4_resume_binding: dict[str, Any] | None = None,
    current_run_contract: dict[str, Any] | None = None,
    wm4_resume_source_checkpoint_identity: dict[str, Any] | None = None,
    wm4_low_peak_moment_residency: "WM4LowPeakAdamMomentResidency | None" = None,
) -> tuple[int, int, int]:
    if wm4_resume_binding is not None:
        if wm4_resume_source_checkpoint_identity is None:
            raise ValueError("WM4 resume has no exact source checkpoint identity")
        _verify_wm4_resume_checkpoint_identity(
            checkpoint, wm4_resume_source_checkpoint_identity
        )
    trainer_path = checkpoint / "trainer_state.json"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"missing checkpoint trainer state: {trainer_path}")
    trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
    if (trainer.get("format_version"), trainer.get("schema")) != (
        FORMAT_VERSION,
        checkpoint_schema,
    ):
        raise ValueError("checkpoint training schema mismatch")
    if wm4_resume_binding is None:
        if trainer.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError("checkpoint run contract does not match this invocation")
    elif current_run_contract is None:
        raise ValueError("WM4 resume requires the current exact run contract")
    else:
        _validate_wm4_resume_binding(
            checkpoint,
            source_run_contract_sha256=trainer.get("run_contract_sha256"),
            source_binding=trainer.get("wm4_resume_binding"),
            current_run_contract=current_run_contract,
            current_binding=wm4_resume_binding,
        )
    rank_payload = torch.load(
        checkpoint / f"rank-{rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        rank_payload.get("rank") != rank
        or rank_payload.get("world_size") != dist.get_world_size()
    ):
        raise ValueError("checkpoint rank/world contract mismatch")
    if rank_payload.get("global_step") != trainer.get("global_step"):
        raise ValueError("checkpoint rank payload step/contract mismatch")
    if wm4_resume_binding is None:
        if rank_payload.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError("checkpoint rank payload step/contract mismatch")
    elif (
        rank_payload.get("run_contract_sha256") != trainer.get("run_contract_sha256")
        or rank_payload.get("wm4_resume_binding")
        != trainer.get("wm4_resume_binding")
    ):
        raise ValueError("WM4 checkpoint rank payload binding differs from trainer state")
    state_config, optim_config = _checkpoint_configs()
    try:
        with FSDP.state_dict_type(
            compressor,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
            optim_config,
        ):
            compressor.load_state_dict(rank_payload["model"])
            optimizer_state = FSDP.optim_state_dict_to_load(
                compressor,
                optimizer,
                rank_payload["optimizer"],
            )
            optimizer.load_state_dict(optimizer_state)
            # FSDP's conversion result can retain a second CPU copy of every
            # local AdamW tensor.  Once it has been installed, release that
            # transient before the residency finally-block moves any partial
            # GPU state back to CPU.  This is intentionally after a successful
            # load only: the finally path still handles conversion/load errors.
            del optimizer_state
            rank_payload.pop("optimizer", None)
    finally:
        if wm4_low_peak_moment_residency is not None:
            # A conversion/load failure can install only part of a rank's
            # moments.  The cleanup accepts a CPU/device mixture and returns
            # every installed exp_avg/exp_avg_sq tensor to CPU before the
            # exception reaches the stage monitor.
            wm4_low_peak_moment_residency.offload_after_optimizer_use(
                optimizer, reason="after_resume", tolerate_partial=True
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    torch.set_rng_state(rank_payload["rng_cpu"])
    torch.cuda.set_rng_state(rank_payload["rng_cuda"])
    if wm4_resume_binding is not None:
        _verify_wm4_resume_checkpoint_identity(
            checkpoint, wm4_resume_source_checkpoint_identity
        )
    return (
        int(trainer["global_step"]),
        int(trainer["next_epoch"]),
        int(trainer["next_line_index"]),
    )


def _resolve_resume_cursor(
    loaded_epoch: int,
    loaded_line_index: int,
    *,
    resume_at_recursive_epoch: bool,
) -> tuple[int, int]:
    """Apply the explicit early curriculum transition without altering training state."""

    if not resume_at_recursive_epoch:
        return loaded_epoch, loaded_line_index
    if loaded_epoch != 0:
        raise ValueError(
            "resume-at-recursive-epoch only accepts a checkpoint from epoch 1"
        )
    return 1, 0


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(value, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else value


def _causal_conv_initial_state(
    conv_state: torch.Tensor,
    *,
    kernel_size: int,
) -> torch.Tensor:
    """Return the numerical cache tail in causal-conv's required layout.

    ``causal_conv1d`` accepts ``[batch, channels, kernel_size - 1]`` but its
    CUDA kernel additionally requires the channels dimension to have stride
    one.  A plain tail slice has the right values and the wrong layout.  The
    transpose/contiguous/transpose sequence preserves the values while making
    channels the contiguous dimension.
    """

    if conv_state.ndim != 3:
        raise ValueError("convolution cache must have shape [batch, channels, width]")
    if kernel_size <= 1 or conv_state.shape[-1] < kernel_size - 1:
        raise ValueError("convolution cache is too short for the configured kernel")
    initial_state = (
        conv_state[..., -(kernel_size - 1) :]
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
    )
    if initial_state.stride(1) != 1:
        raise RuntimeError("failed to construct channel-contiguous convolution state")
    return initial_state


def _chunkable_gated_delta_net_forward(
    self: Any,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Qwen3.5 GatedDeltaNet forward with exact multi-token cache carry.

    Transformers 5.3 only consumes cached convolution/recurrent state when the
    next call contains exactly one token.  For chunked prefill, its stock path
    silently resets both states.  The public CUDA kernels already support
    initial states, so this local method preserves the native calculation while
    carrying state across an arbitrary multi-token chunk boundary.
    """

    apply_padding_mask = getattr(
        self,
        "_specforge_apply_mask_to_padding_states",
        qwen3_5_modeling.apply_mask_to_padding_states,
    )
    hidden_states = apply_padding_mask(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    has_previous_state = bool(
        cache_params is not None
        and cache_params.has_previous_state
        and cache_position is not None
    )
    single_token_update = has_previous_state and seq_len == 1
    conv_state = None
    recurrent_state = None
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    raw_mixed_qkv = mixed_qkv
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if single_token_update:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        initial_conv_state = None
        if has_previous_state:
            if conv_state is None:
                raise RuntimeError("linear-attention convolution cache is missing")
            initial_conv_state = _causal_conv_initial_state(
                conv_state,
                kernel_size=self.conv_kernel_size,
            )
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                initial_states=initial_conv_state,
                activation=self.activation,
            )
        else:
            if initial_conv_state is None:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
            else:
                convolution_input = torch.cat((initial_conv_state, mixed_qkv), dim=-1)
                mixed_qkv = F.conv1d(
                    convolution_input,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    groups=self.conv1d.groups,
                )
                mixed_qkv = F.silu(mixed_qkv)
        if cache_params is not None:
            if conv_state is None:
                next_conv_state = F.pad(
                    raw_mixed_qkv,
                    (self.conv_kernel_size - raw_mixed_qkv.shape[-1], 0),
                )
            else:
                next_conv_state = torch.cat((conv_state, raw_mixed_qkv), dim=-1)[
                    ..., -self.conv_kernel_size :
                ]
            cache_params.conv_states[self.layer_idx] = next_conv_state

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        repeat = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    if single_token_update:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state if has_previous_state else None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def _install_chunkable_qwen35_linear_cache(model: Qwen35TextTarget) -> int:
    modeling_module = qwen35_modeling_module(model)
    patched = 0
    for layer in model.model.layers:
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        linear_attn._specforge_apply_mask_to_padding_states = (
            modeling_module.apply_mask_to_padding_states
        )
        linear_attn.forward = types.MethodType(
            _chunkable_gated_delta_net_forward, linear_attn
        )
        patched += 1
    if patched != int(model.config.layer_types.count("linear_attention")):
        raise RuntimeError("failed to patch every Qwen3.5 linear-attention layer")
    return patched


_QWEN35_CACHED_SDPA_IMPLEMENTATION = "specforge_qwen35_cached_lower_right_sdpa_v2"
_ENABLE_QWEN35_LOWER_RIGHT_SDPA = "_specforge_enable_cached_teacher_lower_right_sdpa"
_INSTALLED_QWEN35_LOWER_RIGHT_SDPA = (
    "_specforge_installed_cached_teacher_lower_right_sdpa"
)


def _qwen35_lower_right_gqa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float,
    scale: float | None,
) -> torch.Tensor:
    """Run non-square cached causal attention without expanding GQA KV heads.

    PyTorch's ``CausalBias`` dispatch represents the lower-right causal mask
    needed by a multi-token KV-cache continuation.  Unlike Transformers 5.3's
    generic SDPA adapter, it can retain native grouped-query attention instead
    of materializing a boolean mask and repeating every KV head.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("cached Qwen3.5 SDPA tensors must be rank four")
    if key.shape != value.shape:
        raise ValueError("cached Qwen3.5 SDPA key/value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("cached Qwen3.5 SDPA batch/head dimensions are invalid")
    query_tokens = int(query.shape[-2])
    key_tokens = int(key.shape[-2])
    if query_tokens <= 0 or key_tokens <= query_tokens:
        raise ValueError("lower-right SDPA requires a non-empty cached prefix")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("Qwen3.5 query heads must be divisible by KV heads")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("SDPA dropout probability is outside [0, 1)")
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_lower_right(query_tokens, key_tokens),
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        enable_gqa=True,
    )


def _qwen35_cached_attention_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Dispatch only controlled non-square cache attention to lower-right GQA."""

    enabled = bool(getattr(module.config, _ENABLE_QWEN35_LOWER_RIGHT_SDPA, False))
    if enabled and query.shape[-2] < key.shape[-2] and attention_mask is None:
        if module.training:
            raise RuntimeError(
                "cached lower-right teacher attention requires eval mode"
            )
        if query.shape[0] != 1:
            raise RuntimeError(
                "cached lower-right teacher attention requires batch size one"
            )
        output = _qwen35_lower_right_gqa_sdpa(
            query,
            key,
            value,
            dropout_p=dropout,
            scale=scaling,
        )
        return output.transpose(1, 2).contiguous(), None
    return transformers_sdpa_attention.sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def _qwen35_cached_mask_interface(**kwargs: Any) -> torch.Tensor | None:
    """Avoid a dense mask only for a verified unpadded cached continuation."""

    config = kwargs.get("config")
    cache_position = kwargs.get("cache_position")
    key_tokens = kwargs.get("kv_length")
    attention_mask = kwargs.get("attention_mask")
    if config is None or not isinstance(cache_position, torch.Tensor):
        raise RuntimeError("cached Qwen3.5 mask interface received invalid inputs")
    query_tokens = int(cache_position.numel())
    if isinstance(key_tokens, bool) or not isinstance(key_tokens, int):
        raise RuntimeError("cached Qwen3.5 mask interface has invalid KV width")
    enabled = bool(getattr(config, _ENABLE_QWEN35_LOWER_RIGHT_SDPA, False))
    if enabled and query_tokens < key_tokens and attention_mask is None:
        if int(kwargs.get("batch_size", 0)) != 1:
            raise RuntimeError(
                "cached Qwen3.5 lower-right mask requires batch size one"
            )
        if int(kwargs.get("kv_offset", 0)) != 0:
            raise RuntimeError(
                "cached Qwen3.5 lower-right mask requires zero KV offset"
            )
        past_tokens = key_tokens - query_tokens
        expected_positions = torch.arange(
            past_tokens,
            key_tokens,
            device=cache_position.device,
            dtype=cache_position.dtype,
        )
        if not torch.equal(cache_position, expected_positions):
            raise RuntimeError(
                "cached Qwen3.5 cache_position is not contiguous with its KV prefix"
            )
        # The paired attention interface constructs CausalBias directly.  It
        # is critical that no [batch, 1, query, KV] tensor is allocated here.
        return None
    return masking_utils.sdpa_mask(**kwargs)


def _register_qwen35_cached_sdpa_interfaces(
    modeling_module: types.ModuleType = qwen3_5_modeling,
) -> None:
    """Register a unique key without replacing Transformers' stock SDPA."""

    attention_mapping = modeling_module.ALL_ATTENTION_FUNCTIONS._global_mapping
    existing_attention = attention_mapping.get(_QWEN35_CACHED_SDPA_IMPLEMENTATION)
    if existing_attention not in (None, _qwen35_cached_attention_interface):
        raise RuntimeError("cached Qwen3.5 attention registry key collision")
    if existing_attention is None:
        modeling_module.ALL_ATTENTION_FUNCTIONS.register(
            _QWEN35_CACHED_SDPA_IMPLEMENTATION,
            _qwen35_cached_attention_interface,
        )

    mask_mapping = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    existing_mask = mask_mapping.get(_QWEN35_CACHED_SDPA_IMPLEMENTATION)
    if existing_mask not in (None, _qwen35_cached_mask_interface):
        raise RuntimeError("cached Qwen3.5 mask registry key collision")
    if existing_mask is None:
        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS.register(
            _QWEN35_CACHED_SDPA_IMPLEMENTATION,
            _qwen35_cached_mask_interface,
        )


def _install_qwen35_cached_lower_right_sdpa(model: Qwen35TextTarget) -> int:
    """Bind the model config to the private registered SDPA implementation."""

    config = model.config
    if bool(getattr(config, _INSTALLED_QWEN35_LOWER_RIGHT_SDPA, False)):
        raise RuntimeError("Qwen3.5 cached SDPA patch was installed twice")
    if config._attn_implementation != "sdpa":
        raise RuntimeError("Qwen3.5 cached lower-right patch requires stock SDPA")
    _register_qwen35_cached_sdpa_interfaces(qwen35_modeling_module(model))
    attentions = [
        layer.self_attn
        for layer in model.model.layers
        if getattr(layer, "self_attn", None) is not None
    ]
    patched = len(attentions)
    if patched != int(model.config.layer_types.count("full_attention")):
        raise RuntimeError("failed to patch every Qwen3.5 full-attention layer")
    setattr(config, _ENABLE_QWEN35_LOWER_RIGHT_SDPA, False)
    setattr(config, _INSTALLED_QWEN35_LOWER_RIGHT_SDPA, True)
    config._attn_implementation = _QWEN35_CACHED_SDPA_IMPLEMENTATION
    if any(
        attention.config._attn_implementation != _QWEN35_CACHED_SDPA_IMPLEMENTATION
        for attention in attentions
    ):
        raise RuntimeError("Qwen3.5 attention modules do not share the patched config")
    return patched


@contextlib.contextmanager
def _enable_qwen35_cached_lower_right_sdpa(model: Qwen35TextTarget) -> Iterator[None]:
    """Enable the local SDPA branch for one unpadded cached-teacher pass."""

    config = model.config
    if (
        not bool(getattr(config, _INSTALLED_QWEN35_LOWER_RIGHT_SDPA, False))
        or config._attn_implementation != _QWEN35_CACHED_SDPA_IMPLEMENTATION
    ):
        raise RuntimeError("Qwen3.5 cached SDPA patch is incomplete")
    if bool(getattr(config, _ENABLE_QWEN35_LOWER_RIGHT_SDPA, False)):
        raise RuntimeError("Qwen3.5 cached SDPA patch is already enabled")
    setattr(config, _ENABLE_QWEN35_LOWER_RIGHT_SDPA, True)
    try:
        yield
    finally:
        setattr(config, _ENABLE_QWEN35_LOWER_RIGHT_SDPA, False)


def _target_layer_ids_for_model_type(model_type: str) -> tuple[int, ...]:
    if model_type == "qwen3_5_text":
        return DEFAULT_QWEN36_27B_TARGET_LAYER_IDS
    if model_type == "qwen3_5_moe_text":
        return DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS
    raise ValueError(f"unsupported frozen target model_type: {model_type!r}")


def _build_compressor_for_target_family(
    target_family: str,
    *,
    output_embedding_rms: float,
    gradient_checkpointing: bool,
) -> GrepSeekDFlashCompressor:
    if target_family == "qwen36_moe":
        return GrepSeekDFlashCompressor.canonical_qwen36_35b_a3b(
            output_embedding_rms=output_embedding_rms,
            gradient_checkpointing=gradient_checkpointing,
        )
    if target_family in {"qwen36", "qwen38"}:
        return GrepSeekDFlashCompressor.canonical_qwen36_27b(
            output_embedding_rms=output_embedding_rms,
            gradient_checkpointing=gradient_checkpointing,
        )
    raise ValueError(f"unsupported compressor target family: {target_family!r}")


def _compressor_architecture_source(target_family: str) -> str:
    if target_family == "qwen36_moe":
        return (
            "z-lab/Qwen3.6-35B-A3B-DFlash@"
            "0ce151d016ba8f0f0454d160a81b52ad4588c924 geometry, random "
            "initialization; no public draft weights loaded"
        )
    if target_family in {"qwen36", "qwen38"}:
        return (
            "z-lab/Qwen3.6-27B-DFlash geometry, random initialization; "
            "no public draft weights loaded"
        )
    raise ValueError(f"unsupported compressor target family: {target_family!r}")


def _compressor_architecture_provenance(
    target_family: str,
) -> dict[str, Any] | None:
    """Return the immutable public geometry receipt used by the A3B draft."""

    if target_family != "qwen36_moe":
        if target_family in {"qwen36", "qwen38"}:
            return None
        raise ValueError(f"unsupported compressor target family: {target_family!r}")
    receipt = json.loads(json.dumps(QWEN36_35B_A3B_DFLASH_UPSTREAM_RECEIPT))
    receipt_sha256 = _canonical_json_sha256(receipt)
    if receipt_sha256 != QWEN36_35B_A3B_DFLASH_UPSTREAM_RECEIPT_SHA256:
        raise RuntimeError("pinned A3B DFlash upstream geometry receipt changed")
    return {
        **receipt,
        "receipt_sha256": receipt_sha256,
        "weights_loaded": False,
        "initialization": "fresh_random_matching_geometry",
    }


class FrozenTP2Target:
    def __init__(
        self,
        model: Qwen35TextTarget,
        *,
        gradient_checkpointing: bool,
        cached_teacher_enabled: bool = True,
    ) -> None:
        self.model = model.requires_grad_(False)
        self.target_layer_ids = _target_layer_ids_for_model_type(
            str(model.config.model_type)
        )
        if not self.target_layer_ids or max(self.target_layer_ids) >= len(
            model.model.layers
        ):
            raise ValueError(
                "selected target layers are incompatible with the target: "
                f"layers={self.target_layer_ids!r}, "
                f"decoder_layers={len(model.model.layers)}"
            )
        self.cached_teacher_enabled = bool(cached_teacher_enabled)
        if self.cached_teacher_enabled:
            self.chunkable_linear_cache_layers = _install_chunkable_qwen35_linear_cache(
                self.model
            )
            self.cached_lower_right_sdpa_layers = (
                _install_qwen35_cached_lower_right_sdpa(self.model)
            )
        else:
            self.chunkable_linear_cache_layers = 0
            self.cached_lower_right_sdpa_layers = 0
        self.model.config.use_cache = False
        if gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
            self.model.train()
        else:
            self.model.eval()
        self.gradient_checkpointing = bool(gradient_checkpointing)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.model.lm_head

    def embed_ids(self, ids: Sequence[int]) -> torch.Tensor:
        tensor = torch.tensor(ids, device=self.device, dtype=torch.long)[None]
        return _full_tensor(self.model.get_input_embeddings()(tensor))

    def hidden(self, embeddings: torch.Tensor, *, grad: bool) -> torch.Tensor:
        mask = torch.ones(
            embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
        )
        mode = contextlib.nullcontext() if grad else torch.no_grad()
        with mode:
            output = self.model.model(
                inputs_embeds=embeddings,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
        return _full_tensor(output.last_hidden_state)

    def cached_teacher_prediction_hidden(
        self,
        prefix_ids: Sequence[int],
        target_ids: Sequence[int],
        *,
        chunk_tokens: int,
    ) -> torch.Tensor:
        """Return target-aligned teacher rows using exact cached causal prefill.

        The returned width equals ``len(target_ids)``. Row zero is the hidden
        state of the final prefix token (which predicts target token zero), and
        later rows are the hidden states of all target tokens except the last.
        Every target forward receives at most ``chunk_tokens`` new positions;
        full-attention KV and linear-attention recurrent states remain in the
        native Qwen3.5 hybrid cache across chunks.
        """

        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if not self.cached_teacher_enabled:
            raise RuntimeError("cached teacher execution is disabled for this target")
        if not prefix_ids or not target_ids:
            raise ValueError("cached teacher prefix and target must be non-empty")
        # Match the canonical monolithic teacher/student forward shape exactly.
        # The final target token's hidden row is unused by the shifted KL, but
        # retaining it avoids shape-dependent SDPA kernel drift in the earlier
        # causal rows.
        cached_teacher_input_ids = (*prefix_ids, *target_ids)
        first_prediction_position = len(prefix_ids) - 1
        prediction_stop = first_prediction_position + len(target_ids)
        past_key_values = None
        prediction_rows: list[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), _enable_qwen35_cached_lower_right_sdpa(self.model):
                chunk_ranges = _cached_teacher_chunk_ranges(
                    len(cached_teacher_input_ids),
                    chunk_tokens=chunk_tokens,
                    # The prediction span begins at the final prefix token, one
                    # position before the target suffix itself.
                    protected_suffix_tokens=len(target_ids) + 1,
                )
                for start, stop in chunk_ranges:
                    embeddings = self.embed_ids(cached_teacher_input_ids[start:stop])
                    output = self.model.model(
                        inputs_embeds=embeddings,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = output.past_key_values
                    hidden = _full_tensor(output.last_hidden_state)
                    overlap_start = max(start, first_prediction_position)
                    overlap_stop = min(stop, prediction_stop)
                    if overlap_start < overlap_stop:
                        prediction_rows.append(
                            hidden[
                                :,
                                overlap_start - start : overlap_stop - start,
                                :,
                            ]
                        )
        finally:
            self.model.train(was_training)
        if not prediction_rows:
            raise RuntimeError("cached teacher produced no prediction rows")
        result = torch.cat(prediction_rows, dim=1).detach()
        if result.shape[1] != len(target_ids):
            raise RuntimeError(
                "cached teacher prediction width mismatch: "
                f"expected {len(target_ids)}, got {result.shape[1]}"
            )
        return result

    def source_features(
        self,
        context_embeddings: torch.Tensor,
        *,
        source_start: int,
        source_stop: int,
    ) -> torch.Tensor:
        if not 0 <= source_start < source_stop <= context_embeddings.shape[1]:
            raise ValueError("source span is outside the contextual input")
        captures: dict[int, torch.Tensor] = {}
        hooks = []
        for layer_id in self.target_layer_ids:

            def capture(_module, _inputs, output, *, layer_id=layer_id):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[layer_id] = _full_tensor(
                    hidden[:, source_start:source_stop]
                ).detach()

            hooks.append(
                self.model.model.layers[layer_id].register_forward_hook(capture)
            )
        try:
            self.hidden(context_embeddings, grad=False)
        finally:
            for hook in hooks:
                hook.remove()
        if tuple(captures) != self.target_layer_ids:
            raise RuntimeError(
                f"selected target captures are incomplete: {tuple(captures)}"
            )
        return torch.cat(
            [captures[layer_id] for layer_id in self.target_layer_ids],
            dim=-1,
        )


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield line_index, row


def _canonical_row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _iter_parquet(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream the strict upstream ``tools/messages`` parquet in physical order."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for agentic parquet training"
        ) from error
    parquet = pq.ParquetFile(str(path))
    columns = tuple(parquet.schema_arrow.names)
    if columns != ("tools", "messages"):
        raise ValueError(
            "agentic parquet must contain exactly ordered tools/messages columns, "
            f"got {columns}"
        )
    row_index = 0
    for batch in parquet.iter_batches(
        batch_size=128,
        columns=["tools", "messages"],
        use_threads=False,
    ):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise ValueError("agentic parquet yielded a non-object row")
            yield row_index, row
            row_index += 1


def _iter_training_rows(
    path: Path,
    data_contract: str,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if data_contract == LEGACY_TOKEN_CLOCK_DATA_CONTRACT:
        yield from _iter_jsonl(path)
        return
    if data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
        yield from _iter_parquet(path)
        return
    raise ValueError(f"unsupported data contract: {data_contract}")


def _stage1_event(
    events: tuple[TokenClockEvent | AgenticCompressionEvent, ...],
    *,
    example_id: str,
    epoch: int,
    seed: int,
) -> TokenClockEvent | AgenticCompressionEvent:
    payload = f"{seed}:{epoch}:{example_id}".encode()
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(events)
    return events[index]


def _events_within_sequence_cap(
    events: Sequence[TokenClockEvent],
    *,
    prefix_tokens: int,
    max_sequence_tokens: int,
) -> tuple[TokenClockEvent, ...]:
    """Keep the chronological prefix whose raw teacher inputs fit the cap."""

    if prefix_tokens < 0 or max_sequence_tokens <= 0:
        raise ValueError("sequence-cap inputs are invalid")
    kept = tuple(
        event
        for event in events
        if prefix_tokens
        + event.working_memory_stop
        + (event.target_stop - event.target_start)
        <= max_sequence_tokens
    )
    if kept != tuple(events[: len(kept)]):
        raise ValueError("legal token-clock events must form a chronological prefix")
    return kept


def _recursive_events_within_caps(
    events: Sequence[TokenClockEvent],
    *,
    prefix_tokens: int,
    max_student_sequence_tokens: int,
    max_teacher_sequence_tokens: int,
) -> tuple[TokenClockEvent, ...]:
    """Keep recursive events whose student and raw teacher fit their caps."""

    kept: list[TokenClockEvent] = []
    slot_counts = recurrent_slot_counts(len(events))
    for event, memory_tokens in zip(events, slot_counts, strict=True):
        target_tokens = event.target_stop - event.target_start
        teacher_tokens = prefix_tokens + event.working_memory_stop + target_tokens
        student_tokens = (
            prefix_tokens + memory_tokens + RAW_WORKING_MEMORY_TOKENS + target_tokens
        )
        if (
            teacher_tokens > max_teacher_sequence_tokens
            or student_tokens > max_student_sequence_tokens
        ):
            break
        kept.append(event)
    return tuple(kept)


def _agentic_recurrent_slot_counts(
    events: Sequence[AgenticCompressionEvent],
) -> tuple[int, ...]:
    counts: list[int] = []
    previous_slots = 0
    previous_stop = 0
    for expected_index, event in enumerate(events):
        if event.event_index != expected_index or event.block_start != previous_stop:
            raise ValueError("agentic events must be one contiguous causal prefix")
        block_tokens = event.block_stop - event.block_start
        if block_tokens <= 0:
            raise ValueError("agentic compression blocks must be non-empty")
        previous_slots = agentic_recurrent_slot_count(
            previous_slots,
            block_tokens,
        )
        counts.append(previous_slots)
        previous_stop = event.block_stop
    return tuple(counts)


def _agentic_events_within_caps(
    events: Sequence[AgenticCompressionEvent],
    *,
    prefix_tokens: int,
    max_student_sequence_tokens: int,
    max_teacher_sequence_tokens: int,
    stage_one: bool,
) -> tuple[AgenticCompressionEvent, ...]:
    """Apply caps using variable chunk widths and adapter recurrence geometry."""

    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    if max_student_sequence_tokens <= 0 or max_teacher_sequence_tokens <= 0:
        raise ValueError("agentic sequence caps must be positive")
    recurrent_slots = _agentic_recurrent_slot_counts(events)
    kept: list[AgenticCompressionEvent] = []
    for event, memory_tokens in zip(events, recurrent_slots, strict=True):
        block_tokens = event.block_stop - event.block_start
        target_tokens = event.target_stop - event.target_start
        working_tokens = event.working_stop - event.working_start
        if target_tokens <= 0 or working_tokens < 0:
            raise ValueError("agentic event target/working ranges are invalid")
        teacher_tokens = prefix_tokens + event.working_stop + target_tokens
        if stage_one:
            memory_tokens = slot_count_for_physical_tokens(block_tokens)
            student_tokens = (
                prefix_tokens
                + event.block_start
                + memory_tokens
                + working_tokens
                + target_tokens
            )
        else:
            student_tokens = (
                prefix_tokens + memory_tokens + working_tokens + target_tokens
            )
        fits = (
            teacher_tokens <= max_teacher_sequence_tokens
            and student_tokens <= max_student_sequence_tokens
        )
        if fits:
            kept.append(event)
        elif not stage_one:
            # A later recurrent event cannot be reached without this state update.
            break
    return tuple(kept)


def _resolve_parity_descriptor(
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any] | None:
    if args.parity_line_index is None:
        return None
    if (
        getattr(args, "data_contract", LEGACY_TOKEN_CLOCK_DATA_CONTRACT)
        == QWEN36_AGENTIC_DATA_CONTRACT
    ):
        return _resolve_agentic_probe_descriptor(args, tokenizer)
    teacher_execution_mode = getattr(args, "teacher_execution_mode", "cached")
    selected_row: Any = None
    selected_line_text: str | None = None
    with args.data.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index != args.parity_line_index:
                continue
            try:
                selected_row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("selected fixed-probe line is invalid JSON") from error
            selected_line_text = line
            break
    if selected_row is None:
        raise ValueError("selected fixed-probe line does not exist")
    if not isinstance(selected_row, dict):
        raise ValueError("selected fixed-probe line must contain a JSON object")
    example = build_qwen36_token_clock_example(selected_row, tokenizer)
    if example.example_id != args.parity_expected_example_id:
        raise ValueError(
            "selected fixed-probe example id mismatch: "
            f"expected {args.parity_expected_example_id!r}, got {example.example_id!r}"
        )
    candidate_events = token_clock_events(
        len(example.history_input_ids), target_tokens=args.kl_tokens
    )
    events = _recursive_events_within_caps(
        candidate_events,
        prefix_tokens=len(example.raw_prefix_input_ids),
        max_student_sequence_tokens=args.max_target_sequence_tokens,
        max_teacher_sequence_tokens=args.max_teacher_sequence_tokens,
    )
    endpoint = args.parity_event_index
    if endpoint >= len(events):
        raise ValueError(
            "selected fixed-probe event is absent or excluded by sequence caps"
        )
    selected_events = events[: endpoint + 1]
    if tuple(event.event_index for event in selected_events) != tuple(
        range(endpoint + 1)
    ):
        raise ValueError("selected fixed-probe events are not a complete causal prefix")
    event = selected_events[-1]
    target_tokens = event.target_stop - event.target_start
    teacher_tokens = (
        len(example.raw_prefix_input_ids) + event.working_memory_stop + target_tokens
    )
    memory_tokens = recurrent_slot_counts(len(candidate_events))[endpoint]
    student_tokens = (
        len(example.raw_prefix_input_ids)
        + memory_tokens
        + RAW_WORKING_MEMORY_TOKENS
        + target_tokens
    )
    cached_teacher_chunk_ranges = (
        _cached_teacher_chunk_ranges(
            teacher_tokens,
            chunk_tokens=args.teacher_prefill_chunk_tokens,
            protected_suffix_tokens=target_tokens + 1,
        )
        if teacher_execution_mode == "cached"
        else ()
    )
    cached_chunks = len(cached_teacher_chunk_ranges)
    expected = {
        "teacher_sequence_tokens": args.parity_expected_teacher_tokens,
        "student_sequence_tokens": args.parity_expected_student_tokens,
        "teacher_cached_chunks": args.parity_expected_cached_chunks,
    }
    actual = {
        "teacher_sequence_tokens": teacher_tokens,
        "student_sequence_tokens": student_tokens,
        "teacher_cached_chunks": cached_chunks,
    }
    if actual != expected:
        raise ValueError(
            "selected fixed-probe width contract mismatch: "
            f"expected {expected}, got {actual}"
        )
    if teacher_execution_mode == "cached" and cached_chunks < 2:
        raise ValueError("selected parity event does not cross a cache boundary")
    teacher_prefix_ids = (
        *example.raw_prefix_input_ids,
        *example.history_input_ids[: event.working_memory_stop],
    )
    target_ids = example.history_input_ids[event.target_start : event.target_stop]

    def ids_sha256(ids: Sequence[int]) -> str:
        payload = json.dumps(
            list(ids), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    if selected_line_text is None:
        raise RuntimeError("selected fixed-probe raw line was not retained")
    full_teacher_input_ids_sha256 = ids_sha256((*teacher_prefix_ids, *target_ids))
    return {
        "line_index": args.parity_line_index,
        "example_id": example.example_id,
        "endpoint_event_index": endpoint,
        "selected_event_indices": list(range(endpoint + 1)),
        "raw_line_sha256": hashlib.sha256(
            selected_line_text.encode("utf-8")
        ).hexdigest(),
        "teacher_prefix_ids_sha256": ids_sha256(teacher_prefix_ids),
        "target_ids_sha256": ids_sha256(target_ids),
        # Logical causal-LM forcing excludes the final target token.  The
        # cached implementation deliberately includes that unused token so
        # its full input shape matches the monolithic parity reference.
        "logical_teacher_forcing_ids_sha256": ids_sha256(
            (*teacher_prefix_ids, *target_ids[:-1])
        ),
        "cached_teacher_input_ids_sha256": (
            full_teacher_input_ids_sha256
            if teacher_execution_mode == "cached"
            else None
        ),
        "cached_teacher_chunk_ranges": [
            [start, stop] for start, stop in cached_teacher_chunk_ranges
        ],
        **actual,
        **(
            {
                "teacher_execution_mode": "monolithic_only",
                "monolithic_teacher_input_ids_sha256": (full_teacher_input_ids_sha256),
            }
            if teacher_execution_mode == "monolithic_only"
            else {}
        ),
    }


def _resolve_agentic_probe_descriptor(
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any]:
    selected_row: dict[str, Any] | None = None
    for row_index, row in _iter_parquet(args.data):
        if row_index == args.parity_line_index:
            selected_row = row
            break
    if selected_row is None:
        raise ValueError("selected fixed-probe parquet row does not exist")
    canonical_row_sha256 = _canonical_row_sha256(selected_row)
    if canonical_row_sha256 != args.fixed_probe_expected_row_sha256:
        raise ValueError(
            "selected fixed-probe canonical row SHA256 mismatch: "
            f"expected {args.fixed_probe_expected_row_sha256}, "
            f"got {canonical_row_sha256}"
        )
    example = render_qwen36_agentic_example(
        selected_row,
        tokenizer,
        target_cap=args.kl_tokens,
    )
    if example.example_id != args.parity_expected_example_id:
        raise ValueError(
            "selected fixed-probe example id mismatch: "
            f"expected {args.parity_expected_example_id!r}, got {example.example_id!r}"
        )
    candidate_events = example.events
    events = _agentic_events_within_caps(
        candidate_events,
        prefix_tokens=len(example.raw_prefix_input_ids),
        max_student_sequence_tokens=args.max_target_sequence_tokens,
        max_teacher_sequence_tokens=args.max_teacher_sequence_tokens,
        stage_one=False,
    )
    endpoint = args.parity_event_index
    if endpoint >= len(events):
        raise ValueError(
            "selected fixed-probe event is absent or excluded by sequence caps"
        )
    selected_events = events[: endpoint + 1]
    if tuple(event.event_index for event in selected_events) != tuple(
        range(endpoint + 1)
    ):
        raise ValueError("selected fixed-probe events are not a complete causal prefix")
    event = selected_events[-1]
    target_tokens = event.target_stop - event.target_start
    teacher_tokens = (
        len(example.raw_prefix_input_ids) + event.working_stop + target_tokens
    )
    memory_tokens = _agentic_recurrent_slot_counts(candidate_events)[endpoint]
    student_tokens = (
        len(example.raw_prefix_input_ids)
        + memory_tokens
        + (event.working_stop - event.working_start)
        + target_tokens
    )
    teacher_execution_mode = getattr(args, "teacher_execution_mode", "cached")
    cached_teacher_chunk_ranges = (
        _cached_teacher_chunk_ranges(
            teacher_tokens,
            chunk_tokens=args.teacher_prefill_chunk_tokens,
            protected_suffix_tokens=target_tokens + 1,
        )
        if teacher_execution_mode == "cached"
        else ()
    )
    actual = {
        "teacher_sequence_tokens": teacher_tokens,
        "student_sequence_tokens": student_tokens,
        "teacher_cached_chunks": len(cached_teacher_chunk_ranges),
    }
    expected = {
        "teacher_sequence_tokens": args.parity_expected_teacher_tokens,
        "student_sequence_tokens": args.parity_expected_student_tokens,
        "teacher_cached_chunks": args.parity_expected_cached_chunks,
    }
    if actual != expected:
        raise ValueError(
            "selected fixed-probe width contract mismatch: "
            f"expected {expected}, got {actual}"
        )
    if teacher_execution_mode == "cached" and len(cached_teacher_chunk_ranges) < 2:
        raise ValueError("selected parity event does not cross a cache boundary")
    teacher_prefix_ids = (
        *example.raw_prefix_input_ids,
        *example.history_input_ids[: event.working_stop],
    )
    target_ids = example.history_input_ids[event.target_start : event.target_stop]

    def ids_sha256(ids: Sequence[int]) -> str:
        payload = json.dumps(
            list(ids), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    full_teacher_input_ids_sha256 = ids_sha256((*teacher_prefix_ids, *target_ids))
    return {
        "data_contract": QWEN36_AGENTIC_DATA_CONTRACT,
        "source_record_kind": "parquet_physical_row",
        "line_index": args.parity_line_index,
        "canonical_row_sha256": canonical_row_sha256,
        "example_id": example.example_id,
        "prompt_group_id": example.prompt_group_id,
        "tool_batch_count": len(example.tool_batches),
        "endpoint_event_index": endpoint,
        "endpoint_is_serving_state": event.is_serving_state,
        "selected_event_indices": list(range(endpoint + 1)),
        "teacher_prefix_ids_sha256": ids_sha256(teacher_prefix_ids),
        "target_ids_sha256": ids_sha256(target_ids),
        "logical_teacher_forcing_ids_sha256": ids_sha256(
            (*teacher_prefix_ids, *target_ids[:-1])
        ),
        "cached_teacher_input_ids_sha256": (
            full_teacher_input_ids_sha256
            if teacher_execution_mode == "cached"
            else None
        ),
        "cached_teacher_chunk_ranges": [
            [start, stop] for start, stop in cached_teacher_chunk_ranges
        ],
        **actual,
        **(
            {
                "teacher_execution_mode": "monolithic_only",
                "monolithic_teacher_input_ids_sha256": (full_teacher_input_ids_sha256),
            }
            if teacher_execution_mode == "monolithic_only"
            else {}
        ),
    }


def _prefix_and_target(
    target: FrozenTP2Target,
    prefix_ids: Sequence[int],
    target_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = target.embed_ids(prefix_ids)
    continuation = target.embed_ids(target_ids)
    return torch.cat((prefix, continuation), dim=1), continuation


@dataclass(frozen=True)
class MindForgeCompressionPlan:
    """Fixed suffix replay plus every directly supervised legal state."""

    history_window_tokens: int
    recurrent_updates: int
    first_supervised_update: int
    state_replay_updates: int
    supervised_updates: int


class MindForgeObjectiveOutput(NamedTuple):
    loss: torch.Tensor
    ce_loss: torch.Tensor | None
    kl_loss: torch.Tensor | None
    token_accuracy: torch.Tensor | None
    top1_agreement: torch.Tensor | None
    token_count: torch.Tensor


def _detach_mindforge_objective_output(
    output: MindForgeObjectiveOutput,
) -> MindForgeObjectiveOutput:
    """Retain scalar telemetry without keeping completed autograd graphs alive."""

    def detached(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach()

    return MindForgeObjectiveOutput(
        loss=output.loss.detach(),
        ce_loss=detached(output.ce_loss),
        kl_loss=detached(output.kl_loss),
        token_accuracy=detached(output.token_accuracy),
        top1_agreement=detached(output.top1_agreement),
        token_count=output.token_count.detach(),
    )


@dataclass(frozen=True)
class MindForgePreparedAction:
    event_id: str
    prefix_ids: tuple[int, ...]
    history_window_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    raw_history_tokens: int
    rendered: MindForgeRenderedEvent
    plan: MindForgeCompressionPlan


def _mindforge_selected_action_weights(
    events: Sequence[MindForgeDecisionEvent],
) -> tuple[float, ...]:
    """Give every selected action equal mass and every trajectory unit mass."""

    if not events:
        raise ValueError("MindForge trajectory has no selected actions")
    trajectory_sha = events[0].source_trajectory_sha256
    event_ids = [event.event_id for event in events]
    if any(event.source_trajectory_sha256 != trajectory_sha for event in events):
        raise ValueError("selected MindForge actions cross trajectory boundaries")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("selected MindForge actions contain duplicates")
    weight = 1.0 / float(len(events))
    weights = tuple(weight for _ in events)
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("MindForge selected action weights do not sum to one")
    return weights


def _mindforge_compression_plan(
    *,
    prefix_tokens: int,
    history_tokens: int,
    target_tokens: int,
    max_student_sequence_tokens: int,
    history_window_tokens: int,
) -> MindForgeCompressionPlan:
    """Plan all legal action-supervised states within one fixed raw suffix."""

    for name, value in (
        ("prefix_tokens", prefix_tokens),
        ("history_tokens", history_tokens),
        ("target_tokens", target_tokens),
        ("max_student_sequence_tokens", max_student_sequence_tokens),
        ("history_window_tokens", history_window_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a plain int")
    if prefix_tokens <= 0 or history_tokens < 0 or target_tokens <= 0:
        raise ValueError("MindForge prefix/target must be non-empty")
    if max_student_sequence_tokens <= 0:
        raise ValueError("max_student_sequence_tokens must be positive")
    if history_window_tokens <= 0 or history_window_tokens % EVICTION_BLOCK_TOKENS:
        raise ValueError("history_window_tokens must be a positive 1024 multiple")
    if history_tokens < history_window_tokens:
        raise ValueError("MindForge action has less history than the fixed window")

    maximum_updates = history_window_tokens // EVICTION_BLOCK_TOKENS - 1
    if maximum_updates <= 0:
        raise ValueError(
            "history_window_tokens must retain one complete 1024-token working block"
        )
    slot_counts = recurrent_slot_counts(maximum_updates)
    first_supervised_update = None
    for update_count, memory_tokens in enumerate(slot_counts, start=1):
        working_tokens = history_window_tokens - update_count * EVICTION_BLOCK_TOKENS
        student_tokens = prefix_tokens + memory_tokens + working_tokens + target_tokens
        if student_tokens <= max_student_sequence_tokens:
            first_supervised_update = update_count
            break
    if first_supervised_update is None:
        raise ValueError(
            "complete MindForge action exceeds the student cap after every legal "
            "1024-token recurrent update"
        )
    return MindForgeCompressionPlan(
        history_window_tokens=history_window_tokens,
        recurrent_updates=maximum_updates,
        first_supervised_update=first_supervised_update,
        state_replay_updates=first_supervised_update - 1,
        supervised_updates=maximum_updates - first_supervised_update + 1,
    )


def _mindforge_supervised_cutpoint_weights(
    plan: MindForgeCompressionPlan,
) -> tuple[float, ...]:
    if plan.supervised_updates <= 0:
        raise ValueError("MindForge plan has no supervised compression states")
    weight = 1.0 / float(plan.supervised_updates)
    result = tuple(weight for _ in range(plan.supervised_updates))
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("MindForge compression cutpoint weights do not sum to one")
    return result


def _mindforge_stable_prefix_ids(
    event: MindForgeDecisionEvent,
    rendered: MindForgeRenderedEvent,
    tokenizer: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split native prompt into persistent system/user prefix and transcript."""

    if len(event.context_messages) < 2 or [
        message.get("role") for message in event.context_messages[:2]
    ] != ["system", "user"]:
        raise ValueError("MindForge context must begin with system+user")
    prefix_value = tokenizer.apply_chat_template(
        list(event.context_messages[:2]),
        tools=list(event.tools),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=QWEN38_ENABLE_THINKING,
        preserve_thinking=QWEN38_PRESERVE_THINKING,
        return_dict=False,
    )
    if isinstance(prefix_value, dict):
        prefix_value = prefix_value.get("input_ids")
    if hasattr(prefix_value, "tolist"):
        prefix_value = prefix_value.tolist()
    if (
        not isinstance(prefix_value, Sequence)
        or isinstance(prefix_value, (str, bytes))
        or not prefix_value
    ):
        raise ValueError("MindForge persistent prefix did not render to token ids")
    if prefix_value and isinstance(prefix_value[0], Sequence):
        if len(prefix_value) != 1:
            raise ValueError("MindForge persistent prefix unexpectedly batched")
        prefix_value = prefix_value[0]
    prefix_ids = tuple(int(token) for token in prefix_value)
    if any(token < 0 for token in prefix_ids):
        raise ValueError("MindForge persistent prefix contains invalid token ids")

    prompt_ids = tuple(rendered.input_ids[: rendered.prompt_tokens])
    target_ids = tuple(rendered.input_ids[rendered.target_start : rendered.target_stop])
    if (
        rendered.target_start != rendered.prompt_tokens
        or rendered.target_stop != len(rendered.input_ids)
        or rendered.labels[: rendered.target_start] != (-100,) * rendered.target_start
        or tuple(rendered.labels[rendered.target_start :]) != target_ids
    ):
        raise ValueError("MindForge rendered event violates the prompt/action CE mask")
    if prompt_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("MindForge native prompt is not stable after system+user")
    history_ids = prompt_ids[len(prefix_ids) :]
    if not history_ids or not target_ids:
        raise ValueError("MindForge prompt history/action is empty")
    return prefix_ids, history_ids, target_ids


def _prepare_mindforge_action(
    event: MindForgeDecisionEvent,
    tokenizer: Any,
    *,
    history_window_tokens: int,
    max_action_tokens: int,
    max_student_sequence_tokens: int,
) -> MindForgePreparedAction:
    """Reconstruct one manifest-selected action without any token truncation."""

    rendered = render_mindforge_decision_event(event, tokenizer)
    prefix_ids, history_ids, target_ids = _mindforge_stable_prefix_ids(
        event, rendered, tokenizer
    )
    if len(target_ids) > max_action_tokens:
        raise ValueError(
            "complete MindForge action exceeds mindforge-max-action-tokens; "
            "the action was not truncated"
        )
    plan = _mindforge_compression_plan(
        prefix_tokens=len(prefix_ids),
        history_tokens=len(history_ids),
        target_tokens=len(target_ids),
        max_student_sequence_tokens=max_student_sequence_tokens,
        history_window_tokens=history_window_tokens,
    )
    window = tuple(history_ids[-history_window_tokens:])
    if len(window) != history_window_tokens:
        raise RuntimeError("MindForge history suffix differs from its fixed contract")
    return MindForgePreparedAction(
        event_id=event.event_id,
        prefix_ids=prefix_ids,
        history_window_ids=window,
        target_ids=target_ids,
        raw_history_tokens=len(history_ids),
        rendered=rendered,
        plan=plan,
    )


def _mindforge_recurrent_update(
    *,
    target: FrozenTP2Target,
    compressor: FSDP,
    prefix_ids: Sequence[int],
    block_ids: Sequence[int],
    previous_memory: torch.Tensor | None,
    grad: bool,
) -> torch.Tensor:
    """Apply one update with a detached recurrent input and explicit grad mode."""

    if len(block_ids) != EVICTION_BLOCK_TOKENS:
        raise ValueError("MindForge recurrent update requires one full 1024 block")
    with torch.no_grad():
        prefix_embeddings = target.embed_ids(prefix_ids)
        block_embeddings = target.embed_ids(block_ids)
        source_memory = (
            block_embeddings
            if previous_memory is None
            else torch.cat((previous_memory.detach(), block_embeddings), dim=1)
        )
        context_embeddings = torch.cat((prefix_embeddings, source_memory), dim=1)
        source_start = len(prefix_ids)
        source_features = target.source_features(
            context_embeddings,
            source_start=source_start,
            source_stop=source_start + int(source_memory.shape[1]),
        )
    source_lengths = torch.tensor(
        [int(source_memory.shape[1])],
        device=source_features.device,
        dtype=torch.long,
    )
    source_mask = torch.ones(
        source_features.shape[:2],
        device=source_features.device,
        dtype=torch.bool,
    )
    update_context = contextlib.nullcontext() if grad else torch.no_grad()
    with update_context:
        compressed = compressor(
            source_features,
            source_mask,
            source_token_lengths=source_lengths,
        )
        memory = compressed.embeddings[compressed.attention_mask][None]
    expected_slots = math.ceil(int(source_memory.shape[1]) / COMPRESSION_RATIO)
    if memory.shape != (1, expected_slots, target.hidden_size):
        raise RuntimeError(f"unexpected MindForge memory shape: {memory.shape}")
    if bool(memory.requires_grad) != bool(grad):
        raise RuntimeError("MindForge recurrent update violated explicit grad mode")
    return memory


def _mindforge_action_objective(
    *,
    target: FrozenTP2Target,
    prefix_ids: Sequence[int],
    memory: torch.Tensor,
    working_ids: Sequence[int],
    target_ids: Sequence[int],
    objective: str,
    kl_loss_weight: float,
    lm_head_chunk_size: int,
    teacher_hidden: torch.Tensor | None,
) -> MindForgeObjectiveOutput:
    """Score one full action; prompt/tool observations are absent from labels."""

    if objective not in OBJECTIVES:
        raise ValueError("unsupported MindForge event objective")
    student_prefix_embeddings = torch.cat(
        (
            target.embed_ids(prefix_ids),
            memory,
            target.embed_ids(working_ids),
        ),
        dim=1,
    )
    student_embeddings = torch.cat(
        (student_prefix_embeddings, target.embed_ids(target_ids)), dim=1
    )
    student_hidden = target.hidden(student_embeddings, grad=True)
    target_tensor = torch.tensor(target_ids, device=target.device, dtype=torch.long)[
        None
    ]
    ce = None
    if _objective_uses_ce(objective):
        ce = frozen_lm_head_chunked_token_ce(
            student_hidden,
            target_tensor,
            target.lm_head,
            prefix_width=int(student_prefix_embeddings.shape[1]),
            chunk_size=lm_head_chunk_size,
        )
    kl = None
    if _objective_uses_raw_teacher(objective):
        if teacher_hidden is None:
            raise ValueError(
                "KL objective requires one action-shared raw teacher state"
            )
        kl = frozen_lm_head_chunked_forward_kl(
            student_hidden,
            teacher_hidden,
            target_tensor,
            target.lm_head,
            student_prefix_width=int(student_prefix_embeddings.shape[1]),
            teacher_prefix_width=1,
            chunk_size=lm_head_chunk_size,
        )
        if ce is not None and int(kl.token_count.item()) != int(ce.token_count.item()):
            raise RuntimeError("MindForge CE and KL token masks differ")
    elif teacher_hidden is not None:
        raise ValueError("ce-only action must not receive raw teacher state")
    if objective == "kl":
        if kl is None:
            raise RuntimeError("MindForge KL objective did not produce KL")
        loss = kl.loss
    elif objective == "ce":
        if ce is None:
            raise RuntimeError("MindForge CE objective did not produce CE")
        loss = ce.loss
    else:
        if ce is None or kl is None:
            raise RuntimeError("MindForge CE+KL objective is incomplete")
        loss = ce.loss + float(kl_loss_weight) * kl.loss
    token_count = ce.token_count if ce is not None else kl.token_count
    if token_count is None:
        raise RuntimeError("MindForge objective produced no token count")
    return MindForgeObjectiveOutput(
        loss=loss,
        ce_loss=ce.loss if ce is not None else None,
        kl_loss=kl.loss if kl is not None else None,
        token_accuracy=ce.token_accuracy if ce is not None else None,
        top1_agreement=kl.top1_agreement if kl is not None else None,
        token_count=token_count,
    )


def _vibe_expected_folds(example: Any) -> tuple[tuple[int, ...], ...]:
    """Rebuild serving's left-to-right per-round chunk schedule exactly."""

    rounds = tuple(tuple(round_ids) for round_ids in example.eviction_round_input_ids)
    if not rounds or any(not round_ids for round_ids in rounds):
        raise ValueError("Qwen3.8 vibe action requires non-empty evicted rounds")
    if len(example.eviction_folds) != len(rounds):
        raise ValueError("Qwen3.8 vibe eviction fold/round counts differ")
    chunks: list[tuple[int, ...]] = []
    for round_index, (round_ids, fold) in enumerate(
        zip(rounds, example.eviction_folds, strict=True)
    ):
        expected_chunks = tuple(
            tuple(round_ids[start : start + EVICTION_BLOCK_TOKENS])
            for start in range(0, len(round_ids), EVICTION_BLOCK_TOKENS)
        )
        observed_chunks = tuple(tuple(chunk.input_ids) for chunk in fold.chunks)
        if (
            int(fold.round_index) != round_index
            or tuple(fold.input_ids) != round_ids
            or observed_chunks != expected_chunks
            or any(
                int(chunk.round_index) != round_index
                or int(chunk.chunk_index) != chunk_index
                for chunk_index, chunk in enumerate(fold.chunks)
            )
        ):
            raise ValueError(
                "Qwen3.8 vibe eviction folds changed, crossed a round boundary, or "
                "truncated a complete round"
            )
        chunks.extend(observed_chunks)
    return tuple(chunks)


def _validate_vibe_serving_action_geometry(
    example: Any,
    *,
    working_memory_rounds: int,
    max_student_sequence_tokens: int,
    max_teacher_sequence_tokens: int,
    target_cap: int,
) -> dict[str, Any]:
    """Fail closed on the exact system+user+soft+WM4 -> assistant ABI."""

    from context_compression.qwen38_coding_trajectory_data import (
        QWEN38_GENERATION_BOUNDARY_IDS,
    )

    if not bool(example.is_final_serving_state):
        raise ValueError("Qwen3.8 vibe training admits only final serving states")
    prefix_ids = tuple(example.raw_prefix_input_ids)
    boundary_ids = tuple(example.generation_boundary_input_ids)
    target_ids = tuple(example.target_input_ids)
    target_mask = tuple(example.target_loss_mask)
    working_rounds = tuple(
        tuple(round_ids) for round_ids in example.working_memory_round_input_ids
    )
    working_ids = tuple(example.working_memory_input_ids)
    evicted_rounds = tuple(
        tuple(round_ids) for round_ids in example.eviction_round_input_ids
    )
    raw_teacher_history_ids = tuple(example.raw_teacher_history_input_ids)
    if not prefix_ids or not boundary_ids or not target_ids:
        raise ValueError(
            "Qwen3.8 vibe prefix, generation boundary, and target are required"
        )
    if boundary_ids != QWEN38_GENERATION_BOUNDARY_IDS:
        raise ValueError("Qwen3.8 vibe generation boundary differs from exact G5")
    if len(working_rounds) != working_memory_rounds:
        raise ValueError("Qwen3.8 vibe action does not retain exactly four raw rounds")
    if working_ids != tuple(token for row in working_rounds for token in row):
        raise ValueError("Qwen3.8 vibe working-memory rounds were cropped or reordered")
    if raw_teacher_history_ids != tuple(
        token for row in (*evicted_rounds, *working_rounds) for token in row
    ):
        raise ValueError("Qwen3.8 vibe raw teacher history is not every complete round")
    if (
        int(example.completed_rounds) != len(evicted_rounds) + len(working_rounds)
        or int(example.evicted_rounds) != len(evicted_rounds)
        or len(evicted_rounds) <= 0
    ):
        raise ValueError("Qwen3.8 vibe complete/evicted round counts are inconsistent")
    if len(target_ids) > target_cap:
        raise ValueError("Qwen3.8 vibe assistant target escaped the 256-token cap")
    if len(target_mask) != len(target_ids) or not all(
        bool(value) for value in target_mask
    ):
        raise ValueError(
            "Qwen3.8 vibe loss mask must select only and all rendered assistant "
            "continuation tokens"
        )

    folds = _vibe_expected_folds(example)
    previous_slots = 0
    maximum_physical_source_tokens = 0
    maximum_source_sequence_tokens = 0
    for fold in folds:
        if not 1 <= len(fold) <= EVICTION_BLOCK_TOKENS:
            raise ValueError("Qwen3.8 vibe eviction fold width is invalid")
        physical_source_tokens = previous_slots + len(fold)
        source_sequence_tokens = len(prefix_ids) + physical_source_tokens
        if source_sequence_tokens > max_student_sequence_tokens:
            raise ValueError(
                "Qwen3.8 vibe recurrent source exceeds the 12288-token student cap; "
                "the complete round was not truncated"
            )
        maximum_physical_source_tokens = max(
            maximum_physical_source_tokens, physical_source_tokens
        )
        maximum_source_sequence_tokens = max(
            maximum_source_sequence_tokens, source_sequence_tokens
        )
        previous_slots = math.ceil(physical_source_tokens / COMPRESSION_RATIO)
    if int(example.soft_memory_tokens) != previous_slots:
        raise ValueError("Qwen3.8 vibe soft-memory slot count differs from serving")

    teacher_prefix_ids = (*prefix_ids, *raw_teacher_history_ids, *boundary_ids)
    teacher_sequence_tokens = len(teacher_prefix_ids) + len(target_ids)
    student_prefix_tokens = (
        len(prefix_ids) + previous_slots + len(working_ids) + len(boundary_ids)
    )
    student_sequence_tokens = student_prefix_tokens + len(target_ids)
    observed_counts = {
        "teacher_prefix_tokens": int(example.teacher_prefix_tokens),
        "teacher_sequence_tokens": int(example.teacher_sequence_tokens),
        "student_prefix_tokens": int(example.student_prefix_tokens),
        "student_sequence_tokens": int(example.student_sequence_tokens),
    }
    expected_counts = {
        "teacher_prefix_tokens": len(teacher_prefix_ids),
        "teacher_sequence_tokens": teacher_sequence_tokens,
        "student_prefix_tokens": student_prefix_tokens,
        "student_sequence_tokens": student_sequence_tokens,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            "Qwen3.8 vibe adapter sequence accounting differs from trainer replay: "
            f"observed={observed_counts}, expected={expected_counts}"
        )
    if teacher_sequence_tokens > max_teacher_sequence_tokens:
        raise ValueError(
            "Qwen3.8 vibe raw teacher exceeds its locked cap; no history was cropped"
        )
    if student_sequence_tokens > max_student_sequence_tokens:
        raise ValueError(
            "Qwen3.8 vibe final student exceeds 12288 tokens; no complete round was "
            "cropped"
        )
    return {
        "folds": folds,
        "teacher_prefix_ids": teacher_prefix_ids,
        "working_and_boundary_ids": (*working_ids, *boundary_ids),
        "teacher_sequence_tokens": teacher_sequence_tokens,
        "student_sequence_tokens": student_sequence_tokens,
        "maximum_physical_source_tokens": maximum_physical_source_tokens,
        "maximum_source_sequence_tokens": maximum_source_sequence_tokens,
    }


def _vibe_sequence_cap_skip_reason(
    example: Any,
    *,
    max_student_sequence_tokens: int,
    max_teacher_sequence_tokens: int,
) -> str | None:
    """Classify a whole-action skip without altering any rendered token segment."""

    if int(example.teacher_sequence_tokens) > max_teacher_sequence_tokens:
        return "teacher_over_cap"
    if int(example.student_sequence_tokens) > max_student_sequence_tokens:
        return "student_over_cap"
    previous_slots = 0
    for fold_ids in _vibe_expected_folds(example):
        physical_source_tokens = previous_slots + len(fold_ids)
        if (
            len(example.raw_prefix_input_ids) + physical_source_tokens
            > max_student_sequence_tokens
        ):
            return "student_over_cap"
        previous_slots = math.ceil(physical_source_tokens / COMPRESSION_RATIO)
    return None


def _vibe_recurrent_update(
    *,
    target: FrozenTP2Target,
    compressor: FSDP,
    prefix_ids: Sequence[int],
    fold_ids: Sequence[int],
    previous_memory: torch.Tensor | None,
    grad: bool,
    max_source_sequence_tokens: int,
) -> torch.Tensor:
    """Replay one serving fold; partial final chunks are legal and never padded."""

    if not 1 <= len(fold_ids) <= EVICTION_BLOCK_TOKENS:
        raise ValueError("Qwen3.8 vibe recurrent fold must contain 1..1024 tokens")
    with torch.no_grad():
        prefix_embeddings = target.embed_ids(prefix_ids)
        fold_embeddings = target.embed_ids(fold_ids)
        source_memory = (
            fold_embeddings
            if previous_memory is None
            else torch.cat((previous_memory.detach(), fold_embeddings), dim=1)
        )
        source_sequence_tokens = len(prefix_ids) + int(source_memory.shape[1])
        if source_sequence_tokens > max_source_sequence_tokens:
            raise ValueError("Qwen3.8 vibe recurrent source escaped its sequence cap")
        context_embeddings = torch.cat((prefix_embeddings, source_memory), dim=1)
        source_start = len(prefix_ids)
        source_features = target.source_features(
            context_embeddings,
            source_start=source_start,
            source_stop=source_start + int(source_memory.shape[1]),
        )
    source_lengths = torch.tensor(
        [int(source_memory.shape[1])],
        device=source_features.device,
        dtype=torch.long,
    )
    source_mask = torch.ones(
        source_features.shape[:2],
        device=source_features.device,
        dtype=torch.bool,
    )
    update_context = contextlib.nullcontext() if grad else torch.no_grad()
    with update_context:
        compressed = compressor(
            source_features,
            source_mask,
            source_token_lengths=source_lengths,
        )
        memory = compressed.embeddings[compressed.attention_mask][None]
    expected_slots = math.ceil(int(source_memory.shape[1]) / COMPRESSION_RATIO)
    if memory.shape != (1, expected_slots, target.hidden_size):
        raise RuntimeError(f"unexpected Qwen3.8 vibe memory shape: {memory.shape}")
    if bool(memory.requires_grad) != bool(grad):
        raise RuntimeError("Qwen3.8 vibe recurrent update violated explicit grad mode")
    return memory


def _select_vibe_action_for_epoch(
    examples: Sequence[Any],
    *,
    epoch: int,
    epochs: int,
    seed: int,
) -> Any:
    """Delegate the frozen stable-hash schedule and verify unit trajectory mass."""

    from context_compression.qwen38_coding_trajectory_data import (
        select_action_for_epoch,
    )

    if not examples:
        raise ValueError("Qwen3.8 vibe trajectory has no eligible final action")
    trajectory_ids = {str(example.trajectory_id) for example in examples}
    descriptor_ids = [str(example.descriptor_id) for example in examples]
    if len(trajectory_ids) != 1 or len(descriptor_ids) != len(set(descriptor_ids)):
        raise ValueError("Qwen3.8 vibe candidates cross or duplicate trajectories")
    selected = select_action_for_epoch(
        examples,
        epoch=epoch,
        epochs=epochs,
        seed=seed,
    )
    if sum(selected is candidate for candidate in examples) != 1:
        raise RuntimeError("Qwen3.8 vibe selector did not return exactly one candidate")
    return selected


def _backward_vibe_trajectory(
    *,
    target: FrozenTP2Target,
    compressor: FSDP,
    examples: Sequence[Any],
    manifest_descriptors: Sequence[dict[str, Any]],
    epoch: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[MindForgeObjectiveOutput, dict[str, Any]]], dict[str, int]]:
    """Backprop exactly one final serving-state action for one trajectory/epoch."""

    example = _select_vibe_action_for_epoch(
        examples,
        epoch=epoch,
        epochs=args.epochs,
        seed=args.seed,
    )
    geometry = _validate_vibe_serving_action_geometry(
        example,
        working_memory_rounds=args.vibe_working_memory_rounds,
        max_student_sequence_tokens=args.max_target_sequence_tokens,
        max_teacher_sequence_tokens=args.max_teacher_sequence_tokens,
        target_cap=args.kl_tokens,
    )
    target_ids = tuple(example.target_input_ids)
    teacher_hidden, teacher_cached_chunks, cached_teacher_parity = (
        _teacher_prediction_hidden(
            target=target,
            teacher_prefix=geometry["teacher_prefix_ids"],
            target_ids=target_ids,
            teacher_execution_mode=args.teacher_execution_mode,
            teacher_forward_cap_tokens=args.teacher_prefill_chunk_tokens,
            lm_head_chunk_size=args.lm_head_chunk_size,
            verify_cached_teacher=False,
            cached_teacher_max_output_kl=args.cached_teacher_max_output_kl,
            cached_teacher_min_top1_agreement=(args.cached_teacher_min_top1_agreement),
        )
    )
    previous_memory: torch.Tensor | None = None
    fold_count = len(geometry["folds"])
    for fold_index, fold_ids in enumerate(geometry["folds"]):
        previous_memory = _vibe_recurrent_update(
            target=target,
            compressor=compressor,
            prefix_ids=example.raw_prefix_input_ids,
            fold_ids=fold_ids,
            previous_memory=previous_memory,
            grad=fold_index == fold_count - 1,
            max_source_sequence_tokens=args.max_target_sequence_tokens,
        )
    if previous_memory is None or not previous_memory.requires_grad:
        raise RuntimeError("Qwen3.8 vibe final recurrent fold has no gradient")
    output = _mindforge_action_objective(
        target=target,
        prefix_ids=example.raw_prefix_input_ids,
        memory=previous_memory,
        working_ids=geometry["working_and_boundary_ids"],
        target_ids=target_ids,
        objective="kl",
        kl_loss_weight=args.kl_loss_weight,
        lm_head_chunk_size=args.lm_head_chunk_size,
        teacher_hidden=teacher_hidden,
    )
    output.loss.backward()
    action_descriptor_sha256 = str(example.descriptor()["descriptor_sha256"])
    matching_manifest_descriptors = [
        item
        for item in manifest_descriptors
        if item.get("descriptor_id") == example.descriptor_id
    ]
    if len(matching_manifest_descriptors) != 1:
        raise RuntimeError("selected Qwen3.8 vibe manifest descriptor is not unique")
    descriptor_sha256 = str(matching_manifest_descriptors[0]["descriptor_sha256"])
    lengths = {
        "physical_source_tokens": geometry["maximum_physical_source_tokens"],
        "memory_tokens": int(previous_memory.shape[1]),
        "working_memory_tokens": len(geometry["working_and_boundary_ids"]),
        "teacher_sequence_tokens": geometry["teacher_sequence_tokens"],
        "teacher_cached_chunks": teacher_cached_chunks,
        "teacher_forward_executed": True,
        "student_sequence_tokens": geometry["student_sequence_tokens"],
        "supervised_tokens": len(target_ids),
        "cached_teacher_parity": cached_teacher_parity,
        "state_replay_updates": fold_count - 1,
        "supervised_updates": 1,
        "metric_weight": 1.0,
        "descriptor_id": str(example.descriptor_id),
        "descriptor_sha256": descriptor_sha256,
        "action_descriptor_sha256": action_descriptor_sha256,
        "trajectory_id": str(example.trajectory_id),
        "target_message_index": int(example.target_message_index),
        "selection_epoch": int(epoch),
        "teacher_prefix_sha256": str(example.token_hashes["teacher_prefix_sha256"]),
        "student_prefix_sha256": str(example.token_hashes["student_layout_sha256"]),
        "final_serving_state_only": True,
        "working_memory_rounds": args.vibe_working_memory_rounds,
    }
    return [(_detach_mindforge_objective_output(output), lengths)], {
        "selected_actions": 1,
        "state_replay_updates": fold_count - 1,
        "supervised_compression_events": 1,
        "teacher_forwards": 1,
    }


def _backward_mindforge_manifest_row(
    *,
    target: FrozenTP2Target,
    compressor: FSDP,
    tokenizer: Any,
    manifest_row: dict[str, Any],
    verified_events: Sequence[MindForgeDecisionEvent],
    args: argparse.Namespace,
) -> tuple[list[tuple[MindForgeObjectiveOutput, dict[str, Any]]], dict[str, int]]:
    """Reconstruct, weight, and backprop one exact manifest trajectory row."""

    selected_records = manifest_row.get("events")
    if not isinstance(selected_records, list) or len(selected_records) != len(
        verified_events
    ):
        raise ValueError("MindForge verified event set differs from manifest row")
    action_weights = _mindforge_selected_action_weights(verified_events)
    event_metrics: list[tuple[MindForgeObjectiveOutput, dict[str, Any]]] = []
    stats = {
        "selected_actions": 0,
        "state_replay_updates": 0,
        "supervised_compression_events": 0,
        "teacher_forwards": 0,
    }
    row_metric_weight = 0.0
    for event, selected, expected_action_weight in zip(
        verified_events, selected_records, action_weights, strict=True
    ):
        selected_weight = float(selected["selected_event_weight"])
        if not math.isclose(
            selected_weight,
            expected_action_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("MindForge manifest selected action weight drifted")
        prepared = _prepare_mindforge_action(
            event,
            tokenizer,
            history_window_tokens=args.mindforge_history_window_tokens,
            max_action_tokens=args.mindforge_max_action_tokens,
            max_student_sequence_tokens=args.max_target_sequence_tokens,
        )
        reconstructed = {
            "event_id": prepared.event_id,
            "cutpoint_index": event.cutpoint_index,
            "rendered_sha256": prepared.rendered.rendered_sha256,
            "prompt_tokens": prepared.rendered.prompt_tokens,
            "target_tokens": len(prepared.target_ids),
            "raw_prefix_tokens": len(prepared.prefix_ids),
            "history_tokens": prepared.raw_history_tokens,
            "compression_cutpoint_count": prepared.plan.recurrent_updates,
            "anomaly_flags": list(event.anomaly_flags),
        }
        if any(selected.get(name) != value for name, value in reconstructed.items()):
            raise ValueError(
                "MindForge selected action differs after trainer reconstruction"
            )

        teacher_hidden = None
        teacher_cached_chunks = 0
        cached_teacher_parity = None
        teacher_sequence_tokens = 0
        if _objective_uses_raw_teacher(args.objective):
            teacher_prefix = (
                *prepared.prefix_ids,
                *prepared.history_window_ids,
            )
            teacher_sequence_tokens = len(teacher_prefix) + len(prepared.target_ids)
            if teacher_sequence_tokens > args.max_teacher_sequence_tokens:
                raise ValueError("MindForge raw teacher escaped its manifest cap")
            teacher_hidden, teacher_cached_chunks, cached_teacher_parity = (
                _teacher_prediction_hidden(
                    target=target,
                    teacher_prefix=teacher_prefix,
                    target_ids=prepared.target_ids,
                    teacher_execution_mode=args.teacher_execution_mode,
                    teacher_forward_cap_tokens=args.teacher_prefill_chunk_tokens,
                    lm_head_chunk_size=args.lm_head_chunk_size,
                    verify_cached_teacher=False,
                    cached_teacher_max_output_kl=args.cached_teacher_max_output_kl,
                    cached_teacher_min_top1_agreement=(
                        args.cached_teacher_min_top1_agreement
                    ),
                )
            )
            stats["teacher_forwards"] += 1

        cutpoint_weights = _mindforge_supervised_cutpoint_weights(prepared.plan)
        previous_memory: torch.Tensor | None = None
        supervised_offset = 0
        for update_index in range(prepared.plan.recurrent_updates):
            block_start = update_index * EVICTION_BLOCK_TOKENS
            block_stop = block_start + EVICTION_BLOCK_TOKENS
            is_supervised = update_index + 1 >= prepared.plan.first_supervised_update
            previous_slots = (
                0 if previous_memory is None else int(previous_memory.shape[1])
            )
            memory = _mindforge_recurrent_update(
                target=target,
                compressor=compressor,
                prefix_ids=prepared.prefix_ids,
                block_ids=prepared.history_window_ids[block_start:block_stop],
                previous_memory=previous_memory,
                grad=is_supervised,
            )
            if not is_supervised:
                previous_memory = memory.detach()
                stats["state_replay_updates"] += 1
                continue
            working_ids = prepared.history_window_ids[block_stop:]
            student_sequence_tokens = (
                len(prepared.prefix_ids)
                + int(memory.shape[1])
                + len(working_ids)
                + len(prepared.target_ids)
            )
            if student_sequence_tokens > args.max_target_sequence_tokens:
                raise RuntimeError("MindForge supervised state escaped its student cap")
            output = _mindforge_action_objective(
                target=target,
                prefix_ids=prepared.prefix_ids,
                memory=memory,
                working_ids=working_ids,
                target_ids=prepared.target_ids,
                objective=args.objective,
                kl_loss_weight=args.kl_loss_weight,
                lm_head_chunk_size=args.lm_head_chunk_size,
                teacher_hidden=teacher_hidden,
            )
            cutpoint_weight = cutpoint_weights[supervised_offset]
            metric_weight = selected_weight * cutpoint_weight
            (output.loss * metric_weight).backward()
            previous_memory = memory.detach()
            lengths = {
                "physical_source_tokens": (previous_slots + EVICTION_BLOCK_TOKENS),
                "memory_tokens": int(memory.shape[1]),
                "working_memory_tokens": len(working_ids),
                "teacher_sequence_tokens": teacher_sequence_tokens,
                "teacher_cached_chunks": teacher_cached_chunks,
                "teacher_forward_executed": (
                    _objective_uses_raw_teacher(args.objective)
                    and supervised_offset == 0
                ),
                "student_sequence_tokens": student_sequence_tokens,
                "supervised_tokens": len(prepared.target_ids),
                "cached_teacher_parity": (
                    cached_teacher_parity if supervised_offset == 0 else None
                ),
                "state_replay_updates": (
                    prepared.plan.state_replay_updates if supervised_offset == 0 else 0
                ),
                "supervised_updates": 1,
                "event_id": event.event_id,
                "selected_event_instance_id": selected["selected_event_instance_id"],
                "cutpoint_index": event.cutpoint_index,
                "compression_update_index": update_index + 1,
                "metric_weight": metric_weight,
            }
            event_metrics.append((_detach_mindforge_objective_output(output), lengths))
            row_metric_weight += metric_weight
            supervised_offset += 1
            stats["supervised_compression_events"] += 1
        if supervised_offset != prepared.plan.supervised_updates:
            raise RuntimeError("MindForge supervised cutpoint count changed")
        stats["selected_actions"] += 1
        del teacher_hidden
    if not math.isclose(row_metric_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("MindForge row objective weights do not sum to one")
    return event_metrics, stats


@dataclass(frozen=True)
class MindForgeEpochDriveResult:
    stopped: bool
    completed_exact_set: bool
    next_training_order: int
    trajectories_executed: int
    selected_instances_executed: int


@dataclass(frozen=True)
class VibeEpochDriveResult:
    stopped: bool
    completed_manifest_suffix: bool
    next_training_order: int
    trajectories_seen: int
    trajectories_supervised: int
    trajectories_without_eligible_action: int


def _drive_vibe_manifest_epoch(
    *,
    rows: Iterator[dict[str, Any]],
    expected_trajectories: int,
    line_floor: int,
    epoch: int,
    tokenizer: Any,
    target: Any,
    compressor: Any,
    args: argparse.Namespace,
    expected_descriptors_by_trajectory: dict[str, tuple[dict[str, Any], ...]],
    canary_schedule_selection: dict[str, Any] | None = None,
    commit_trajectory: Any,
    normalize_row: Any | None = None,
    build_examples: Any | None = None,
    backward_row: Any | None = None,
    before_backward_row: Any | None = None,
) -> VibeEpochDriveResult:
    """Drive a deterministic frozen snapshot with a checkpoint-safe row cursor."""

    if isinstance(line_floor, bool) or not isinstance(line_floor, int):
        raise TypeError("Qwen3.8 vibe resume cursor must be a plain int")
    if line_floor < 0 or line_floor > expected_trajectories:
        raise ValueError("Qwen3.8 vibe resume cursor is outside the frozen manifest")
    if normalize_row is None or build_examples is None:
        from context_compression.qwen38_coding_trajectory_data import (
            iter_wm4_action_examples,
            normalize_coding_trajectory,
        )

        normalize_row = normalize_coding_trajectory
        build_examples = iter_wm4_action_examples
    if backward_row is None:
        backward_row = _backward_vibe_trajectory
    from context_compression.qwen38_coding_trajectory_data import (
        action_manifest_descriptor,
    )

    seen_trajectory_ids: set[str] = set()
    trajectories_seen = 0
    trajectories_supervised = 0
    trajectories_without_eligible_action = 0
    next_training_order = line_floor
    for training_order, frozen_row in enumerate(rows):
        if training_order < line_floor:
            continue
        if training_order >= expected_trajectories:
            raise RuntimeError("Qwen3.8 vibe snapshot contains more rows than manifest")
        if int(frozen_row.trajectory_order) < 0:
            raise ValueError("Qwen3.8 vibe frozen trajectory order is invalid")
        episode = normalize_row(frozen_row.row)
        trajectory_id = str(episode.trajectory_id)
        if trajectory_id != str(frozen_row.trajectory_id):
            raise ValueError("Qwen3.8 vibe frozen row trajectory ID changed")
        if not trajectory_id or trajectory_id in seen_trajectory_ids:
            raise ValueError(
                "Qwen3.8 vibe frozen suffix contains an empty or duplicate trajectory"
            )
        seen_trajectory_ids.add(trajectory_id)
        all_examples = tuple(
            build_examples(
                episode,
                tokenizer,
                source_receipt=frozen_row.receipt,
                working_memory_rounds=args.vibe_working_memory_rounds,
                target_cap=args.kl_tokens,
                student_cap=None,
            )
        )
        observed_descriptors = []
        for example in all_examples:
            descriptor = action_manifest_descriptor(
                example,
                trajectory_order=int(frozen_row.trajectory_order),
            )
            observed_descriptors.append(descriptor)
        expected_descriptors = expected_descriptors_by_trajectory.get(trajectory_id)
        if expected_descriptors is None or observed_descriptors != list(
            expected_descriptors
        ):
            raise ValueError(
                "Qwen3.8 vibe runtime reconstruction differs from frozen action "
                "descriptors"
            )
        student_over_cap = 0
        teacher_over_cap = 0
        examples = []
        for example in all_examples:
            skip_reason = _vibe_sequence_cap_skip_reason(
                example,
                max_student_sequence_tokens=args.max_target_sequence_tokens,
                max_teacher_sequence_tokens=args.max_teacher_sequence_tokens,
            )
            if skip_reason == "student_over_cap":
                student_over_cap += 1
            elif skip_reason == "teacher_over_cap":
                teacher_over_cap += 1
            elif skip_reason is None:
                examples.append(example)
            else:
                raise RuntimeError("unknown Qwen3.8 vibe cap skip reason")
        examples = tuple(examples)
        trajectories_seen += 1
        next_training_order = training_order + 1
        if not examples:
            trajectories_without_eligible_action += 1
            stopped = bool(
                commit_trajectory(
                    training_order,
                    next_training_order,
                    trajectory_id,
                    [],
                    {
                        "selected_actions": 0,
                        "state_replay_updates": 0,
                        "supervised_compression_events": 0,
                        "teacher_forwards": 0,
                        "without_eligible_action": 1,
                        "student_over_cap": student_over_cap,
                        "teacher_over_cap": teacher_over_cap,
                    },
                )
            )
        else:
            if any(str(example.trajectory_id) != trajectory_id for example in examples):
                raise ValueError("Qwen3.8 vibe adapter crossed trajectory boundaries")
            selection_epoch = epoch
            expected_selected_descriptor_id: str | None = None
            if canary_schedule_selection is not None:
                if training_order not in {0, 1}:
                    raise ValueError("Qwen3.8 vibe canary has more than two rows")
                schedule_index = 0 if training_order == 0 else 2
                selection_epoch = int(
                    canary_schedule_selection["epoch_schedule"][schedule_index]
                )
                expected_selected_descriptor_id = str(
                    canary_schedule_selection["descriptor_schedule"][schedule_index]
                )
            if before_backward_row is not None:
                before_backward_row()
            event_metrics, row_stats = backward_row(
                target=target,
                compressor=compressor,
                examples=examples,
                manifest_descriptors=expected_descriptors,
                epoch=selection_epoch,
                args=args,
            )
            if len(event_metrics) != 1 or row_stats.get("selected_actions") != 1:
                raise RuntimeError(
                    "Qwen3.8 vibe row did not supervise exactly one final action"
                )
            if (
                expected_selected_descriptor_id is not None
                and str(event_metrics[0][1].get("descriptor_id"))
                != expected_selected_descriptor_id
            ):
                raise RuntimeError(
                    "Qwen3.8 vibe canary selected a descriptor outside its schedule"
                )
            trajectories_supervised += 1
            stopped = bool(
                commit_trajectory(
                    training_order,
                    next_training_order,
                    trajectory_id,
                    event_metrics,
                    {
                        **row_stats,
                        "without_eligible_action": 0,
                        "student_over_cap": student_over_cap,
                        "teacher_over_cap": teacher_over_cap,
                    },
                )
            )
        if stopped:
            return VibeEpochDriveResult(
                stopped=True,
                completed_manifest_suffix=(
                    next_training_order == expected_trajectories
                ),
                next_training_order=next_training_order,
                trajectories_seen=trajectories_seen,
                trajectories_supervised=trajectories_supervised,
                trajectories_without_eligible_action=(
                    trajectories_without_eligible_action
                ),
            )
    if next_training_order != expected_trajectories:
        raise RuntimeError(
            "Qwen3.8 vibe frozen snapshot row count differs from its manifest: "
            f"observed={next_training_order}, expected={expected_trajectories}"
        )
    return VibeEpochDriveResult(
        stopped=False,
        completed_manifest_suffix=True,
        next_training_order=next_training_order,
        trajectories_seen=trajectories_seen,
        trajectories_supervised=trajectories_supervised,
        trajectories_without_eligible_action=trajectories_without_eligible_action,
    )


def _drive_mindforge_manifest_epoch(
    *,
    train_rows: Sequence[dict[str, Any]],
    line_floor: int,
    source_path: Path,
    tokenizer: Any,
    policy: dict[str, Any],
    target: Any,
    compressor: Any,
    args: argparse.Namespace,
    commit_trajectory: Any,
    verify_row: Any | None = None,
    backward_row: Any | None = None,
) -> MindForgeEpochDriveResult:
    """Execute one strict manifest suffix in ``training_order``.

    ``commit_trajectory`` is invoked exactly once after all A actions and K
    supervised states for one trajectory have been backpropagated.  Keeping
    this control flow outside the CUDA setup makes its cursor/exact-set ABI
    directly regression-testable with CPU mocks.
    """

    if isinstance(line_floor, bool) or not isinstance(line_floor, int):
        raise TypeError("MindForge resume cursor must be a plain int")
    if line_floor < 0 or line_floor > len(train_rows):
        raise ValueError("MindForge resume cursor is outside manifest training_order")
    observed_orders = tuple(int(row["training_order"]) for row in train_rows)
    if observed_orders != tuple(range(len(train_rows))):
        raise ValueError("MindForge manifest training_order is not contiguous")
    if verify_row is None:
        from context_compression.mindforge_ce_manifest import (
            verify_manifest_row_events,
        )

        verify_row = verify_manifest_row_events
    if backward_row is None:
        backward_row = _backward_mindforge_manifest_row

    expected_instances = {
        str(event["selected_event_instance_id"])
        for row in train_rows[line_floor:]
        for event in row["events"]
    }
    executed_instances: set[str] = set()
    trajectories_executed = 0
    next_training_order = line_floor
    for manifest_row in train_rows[line_floor:]:
        training_order = int(manifest_row["training_order"])
        verified_events = verify_row(
            source_path,
            manifest_row,
            tokenizer,
            policy,
        )
        row_instances = {
            str(event["selected_event_instance_id"]) for event in manifest_row["events"]
        }
        if len(row_instances) != len(manifest_row["events"]):
            raise RuntimeError(
                "MindForge manifest row has duplicate selected instances"
            )
        if row_instances.intersection(executed_instances):
            raise RuntimeError(
                "MindForge selected instance executed twice in one epoch"
            )
        event_metrics, row_stats = backward_row(
            target=target,
            compressor=compressor,
            tokenizer=tokenizer,
            manifest_row=manifest_row,
            verified_events=verified_events,
            args=args,
        )
        metric_instances = {
            str(lengths["selected_event_instance_id"]) for _, lengths in event_metrics
        }
        if metric_instances != row_instances:
            raise RuntimeError(
                "MindForge backward path did not execute the exact selected set"
            )
        executed_instances.update(row_instances)
        trajectories_executed += 1
        next_training_order = training_order + 1
        stopped = bool(
            commit_trajectory(
                training_order,
                next_training_order,
                event_metrics,
                row_stats,
            )
        )
        if stopped:
            return MindForgeEpochDriveResult(
                stopped=True,
                completed_exact_set=executed_instances == expected_instances,
                next_training_order=next_training_order,
                trajectories_executed=trajectories_executed,
                selected_instances_executed=len(executed_instances),
            )
    if executed_instances != expected_instances:
        missing = expected_instances.difference(executed_instances)
        extra = executed_instances.difference(expected_instances)
        raise RuntimeError(
            "MindForge epoch suffix did not execute its exact manifest set: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return MindForgeEpochDriveResult(
        stopped=False,
        completed_exact_set=True,
        next_training_order=next_training_order,
        trajectories_executed=trajectories_executed,
        selected_instances_executed=len(executed_instances),
    )


def _teacher_prediction_hidden(
    *,
    target: FrozenTP2Target,
    teacher_prefix: Sequence[int],
    target_ids: Sequence[int],
    teacher_execution_mode: str,
    teacher_forward_cap_tokens: int,
    lm_head_chunk_size: int,
    verify_cached_teacher: bool,
    cached_teacher_max_output_kl: float,
    cached_teacher_min_top1_agreement: float,
) -> tuple[torch.Tensor, int, dict[str, float] | None]:
    """Run the teacher first and retain only compact target-aligned rows."""

    teacher_sequence_tokens = len(teacher_prefix) + len(target_ids)
    cached_teacher_parity: dict[str, float] | None = None
    if teacher_execution_mode == "monolithic_only":
        if teacher_sequence_tokens > teacher_forward_cap_tokens:
            raise RuntimeError(
                "monolithic teacher event escaped its hard sequence cap: "
                f"event={teacher_sequence_tokens}, cap={teacher_forward_cap_tokens}"
            )
        teacher_embeddings, _ = _prefix_and_target(target, teacher_prefix, target_ids)
        full_hidden = target.hidden(teacher_embeddings, grad=False)
        teacher_hidden = (
            full_hidden[
                :,
                len(teacher_prefix) - 1 : len(teacher_prefix) - 1 + len(target_ids),
                :,
            ]
            .clone()
            .detach()
        )
        del full_hidden, teacher_embeddings
        teacher_cached_chunks = 0
    elif teacher_execution_mode == "cached" and (
        teacher_sequence_tokens <= teacher_forward_cap_tokens
    ):
        teacher_embeddings, _ = _prefix_and_target(target, teacher_prefix, target_ids)
        full_hidden = target.hidden(teacher_embeddings, grad=False)
        teacher_hidden = (
            full_hidden[
                :,
                len(teacher_prefix) - 1 : len(teacher_prefix) - 1 + len(target_ids),
                :,
            ]
            .clone()
            .detach()
        )
        del full_hidden, teacher_embeddings
        teacher_cached_chunks = 1
    elif teacher_execution_mode == "cached":
        teacher_hidden = target.cached_teacher_prediction_hidden(
            teacher_prefix,
            target_ids,
            chunk_tokens=teacher_forward_cap_tokens,
        )
        teacher_cached_chunks = len(
            _cached_teacher_chunk_ranges(
                teacher_sequence_tokens,
                chunk_tokens=teacher_forward_cap_tokens,
                protected_suffix_tokens=len(target_ids) + 1,
            )
        )
        if verify_cached_teacher:
            monolithic_embeddings, _ = _prefix_and_target(
                target, teacher_prefix, target_ids
            )
            was_training = target.model.training
            target.model.eval()
            try:
                monolithic_full = target.hidden(monolithic_embeddings, grad=False)
            finally:
                target.model.train(was_training)
            monolithic_aligned = (
                monolithic_full[
                    :,
                    len(teacher_prefix) - 1 : len(teacher_prefix) - 1 + len(target_ids),
                    :,
                ]
                .clone()
                .detach()
            )
            del monolithic_full, monolithic_embeddings
            verification_target = torch.tensor(
                target_ids,
                device=target.device,
                dtype=torch.long,
            )[None]
            parity = frozen_lm_head_chunked_forward_kl(
                teacher_hidden,
                monolithic_aligned,
                verification_target,
                target.lm_head,
                student_prefix_width=1,
                teacher_prefix_width=1,
                chunk_size=lm_head_chunk_size,
            )
            absolute_difference = (
                teacher_hidden.float() - monolithic_aligned.float()
            ).abs()
            cached_teacher_parity = {
                "hidden_max_abs_difference": float(absolute_difference.max().item()),
                "hidden_mean_abs_difference": float(absolute_difference.mean().item()),
                "output_distribution_kl": float(parity.loss.item()),
                "top1_agreement": float(parity.top1_agreement.item()),
            }
            if not all(
                math.isfinite(value) for value in cached_teacher_parity.values()
            ):
                raise RuntimeError("cached teacher parity produced non-finite metrics")
            if (
                cached_teacher_parity["output_distribution_kl"]
                > cached_teacher_max_output_kl
                or cached_teacher_parity["output_distribution_kl"] < -1e-6
            ):
                raise RuntimeError(
                    "cached teacher output-distribution KL exceeds its contract: "
                    f"metrics={cached_teacher_parity}, "
                    f"maximum={cached_teacher_max_output_kl}"
                )
            if (
                cached_teacher_parity["top1_agreement"]
                < cached_teacher_min_top1_agreement
            ):
                raise RuntimeError(
                    "cached teacher top-1 agreement is below its contract: "
                    f"metrics={cached_teacher_parity}, "
                    f"minimum={cached_teacher_min_top1_agreement}"
                )
    else:
        raise ValueError(
            f"unsupported teacher execution mode: {teacher_execution_mode}"
        )
    if teacher_hidden.shape[1] != len(target_ids):
        raise RuntimeError("teacher target-aligned hidden width mismatch")
    return teacher_hidden, teacher_cached_chunks, cached_teacher_parity


def _one_event(
    *,
    target: FrozenTP2Target,
    compressor: FSDP,
    example: Qwen36TokenClockExample,
    event: TokenClockEvent,
    previous_memory: torch.Tensor | None,
    stage_one: bool,
    lm_head_chunk_size: int,
    teacher_prefill_chunk_tokens: int,
    teacher_execution_mode: str,
    verify_cached_teacher: bool,
    cached_teacher_max_output_kl: float,
    cached_teacher_min_top1_agreement: float,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    prefix_ids = example.raw_prefix_input_ids
    history = example.history_input_ids
    block_ids = history[event.block_start : event.block_stop]
    working_ids = history[event.working_memory_start : event.working_memory_stop]
    target_ids = history[event.target_start : event.target_stop]
    if (
        len(block_ids) != EVICTION_BLOCK_TOKENS
        or len(working_ids) != RAW_WORKING_MEMORY_TOKENS
    ):
        raise RuntimeError("token-clock event does not contain full raw blocks")

    if stage_one:
        earlier_ids = history[: event.block_start]
        physical_source_tokens = len(block_ids)
        student_discrete_prefix = (*prefix_ids, *earlier_ids)
        teacher_prefix = (*prefix_ids, *earlier_ids, *block_ids, *working_ids)
    else:
        physical_source_tokens = len(block_ids) + (
            0 if previous_memory is None else int(previous_memory.shape[1])
        )
        student_discrete_prefix = prefix_ids
        teacher_prefix = (*prefix_ids, *history[: event.working_memory_stop])

    teacher_sequence_tokens = len(teacher_prefix) + len(target_ids)
    teacher_hidden, teacher_cached_chunks, cached_teacher_parity = (
        _teacher_prediction_hidden(
            target=target,
            teacher_prefix=teacher_prefix,
            target_ids=target_ids,
            teacher_execution_mode=teacher_execution_mode,
            teacher_forward_cap_tokens=teacher_prefill_chunk_tokens,
            lm_head_chunk_size=lm_head_chunk_size,
            verify_cached_teacher=verify_cached_teacher,
            cached_teacher_max_output_kl=cached_teacher_max_output_kl,
            cached_teacher_min_top1_agreement=cached_teacher_min_top1_agreement,
        )
    )

    if stage_one:
        contextual_raw = (*prefix_ids, *earlier_ids, *block_ids)
        context_embeddings = target.embed_ids(contextual_raw)
        source_start = len(prefix_ids) + len(earlier_ids)
    else:
        prefix_embeddings = target.embed_ids(prefix_ids)
        block_embeddings = target.embed_ids(block_ids)
        source_memory = (
            block_embeddings
            if previous_memory is None
            else torch.cat((previous_memory.detach(), block_embeddings), dim=1)
        )
        context_embeddings = torch.cat((prefix_embeddings, source_memory), dim=1)
        source_start = len(prefix_ids)

    source_features = target.source_features(
        context_embeddings,
        source_start=source_start,
        source_stop=source_start + physical_source_tokens,
    )
    compressed = compressor(
        source_features,
        torch.ones(
            source_features.shape[:2], device=source_features.device, dtype=torch.bool
        ),
        source_token_lengths=torch.tensor(
            [physical_source_tokens], device=source_features.device, dtype=torch.long
        ),
    )
    memory = compressed.embeddings[compressed.attention_mask][None]
    expected_slots = math.ceil(physical_source_tokens / COMPRESSION_RATIO)
    if memory.shape != (1, expected_slots, target.hidden_size):
        raise RuntimeError(f"unexpected compressed memory shape: {memory.shape}")

    student_prefix_embeddings = torch.cat(
        (
            target.embed_ids(student_discrete_prefix),
            memory,
            target.embed_ids(working_ids),
        ),
        dim=1,
    )
    student_embeddings = torch.cat(
        (student_prefix_embeddings, target.embed_ids(target_ids)), dim=1
    )
    student_hidden = target.hidden(student_embeddings, grad=True)
    target_tensor = torch.tensor(target_ids, device=target.device, dtype=torch.long)[
        None
    ]
    kl = frozen_lm_head_chunked_forward_kl(
        student_hidden,
        teacher_hidden,
        target_tensor,
        target.lm_head,
        student_prefix_width=int(student_prefix_embeddings.shape[1]),
        teacher_prefix_width=1,
        chunk_size=lm_head_chunk_size,
    )
    lengths = {
        "physical_source_tokens": physical_source_tokens,
        "memory_tokens": int(memory.shape[1]),
        "working_memory_tokens": len(working_ids),
        "teacher_sequence_tokens": teacher_sequence_tokens,
        "teacher_cached_chunks": teacher_cached_chunks,
        "student_sequence_tokens": int(student_embeddings.shape[1]),
        "supervised_tokens": len(target_ids),
        "cached_teacher_parity": cached_teacher_parity,
    }
    return kl, memory, lengths


def _agentic_teacher_for_events(
    *,
    target: FrozenTP2Target,
    example: Qwen36AgenticExample,
    events: Sequence[AgenticCompressionEvent],
    lm_head_chunk_size: int,
    teacher_prefill_chunk_tokens: int,
    teacher_execution_mode: str,
    verify_cached_teacher: bool,
    cached_teacher_max_output_kl: float,
    cached_teacher_min_top1_agreement: float,
) -> tuple[tuple[int, ...], torch.Tensor, int, dict[str, float] | None, int]:
    """Run the shared final-chosen teacher once before any student event."""

    if not events:
        raise ValueError("agentic teacher requires at least one selected event")
    first = events[0]
    signature = (first.working_stop, first.target_start, first.target_stop)
    if any(
        (event.working_stop, event.target_start, event.target_stop) != signature
        for event in events[1:]
    ):
        raise ValueError("agentic row events must share one teacher/target span")
    teacher_prefix = (
        *example.raw_prefix_input_ids,
        *example.history_input_ids[: first.working_stop],
    )
    target_ids = tuple(
        example.history_input_ids[first.target_start : first.target_stop]
    )
    if not target_ids:
        raise ValueError("agentic chosen target is empty")
    teacher_hidden, teacher_cached_chunks, cached_teacher_parity = (
        _teacher_prediction_hidden(
            target=target,
            teacher_prefix=teacher_prefix,
            target_ids=target_ids,
            teacher_execution_mode=teacher_execution_mode,
            teacher_forward_cap_tokens=teacher_prefill_chunk_tokens,
            lm_head_chunk_size=lm_head_chunk_size,
            verify_cached_teacher=verify_cached_teacher,
            cached_teacher_max_output_kl=cached_teacher_max_output_kl,
            cached_teacher_min_top1_agreement=(cached_teacher_min_top1_agreement),
        )
    )
    teacher_sequence_tokens = len(teacher_prefix) + len(target_ids)
    return (
        target_ids,
        teacher_hidden,
        teacher_cached_chunks,
        cached_teacher_parity,
        teacher_sequence_tokens,
    )


def _agentic_one_event(
    *,
    target: FrozenTP2Target,
    compressor: FSDP,
    example: Qwen36AgenticExample,
    event: AgenticCompressionEvent,
    previous_memory: torch.Tensor | None,
    stage_one: bool,
    target_ids: Sequence[int],
    teacher_hidden: torch.Tensor,
    teacher_cached_chunks: int,
    cached_teacher_parity: dict[str, float] | None,
    teacher_sequence_tokens: int,
    teacher_forward_executed: bool,
    lm_head_chunk_size: int,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    """Train one variable-width A3B chunk against a row-shared teacher."""

    prefix_ids = example.raw_prefix_input_ids
    history = example.history_input_ids
    block_ids = history[event.block_start : event.block_stop]
    working_ids = history[event.working_start : event.working_stop]
    event_target_ids = history[event.target_start : event.target_stop]
    if (
        not block_ids
        or event.working_start != event.block_stop
        or tuple(event_target_ids) != tuple(target_ids)
    ):
        raise RuntimeError("agentic event spans do not match the rendered row")

    if stage_one:
        earlier_ids = history[: event.block_start]
        physical_source_tokens = len(block_ids)
        student_discrete_prefix = (*prefix_ids, *earlier_ids)
        contextual_raw = (*prefix_ids, *earlier_ids, *block_ids)
        context_embeddings = target.embed_ids(contextual_raw)
        source_start = len(prefix_ids) + len(earlier_ids)
    else:
        physical_source_tokens = len(block_ids) + (
            0 if previous_memory is None else int(previous_memory.shape[1])
        )
        student_discrete_prefix = prefix_ids
        prefix_embeddings = target.embed_ids(prefix_ids)
        block_embeddings = target.embed_ids(block_ids)
        source_memory = (
            block_embeddings
            if previous_memory is None
            else torch.cat((previous_memory.detach(), block_embeddings), dim=1)
        )
        context_embeddings = torch.cat((prefix_embeddings, source_memory), dim=1)
        source_start = len(prefix_ids)

    source_features = target.source_features(
        context_embeddings,
        source_start=source_start,
        source_stop=source_start + physical_source_tokens,
    )
    compressed = compressor(
        source_features,
        torch.ones(
            source_features.shape[:2],
            device=source_features.device,
            dtype=torch.bool,
        ),
        source_token_lengths=torch.tensor(
            [physical_source_tokens],
            device=source_features.device,
            dtype=torch.long,
        ),
    )
    memory = compressed.embeddings[compressed.attention_mask][None]
    expected_slots = slot_count_for_physical_tokens(physical_source_tokens)
    if memory.shape != (1, expected_slots, target.hidden_size):
        raise RuntimeError(f"unexpected compressed memory shape: {memory.shape}")

    student_prefix_embeddings = torch.cat(
        (
            target.embed_ids(student_discrete_prefix),
            memory,
            target.embed_ids(working_ids),
        ),
        dim=1,
    )
    student_embeddings = torch.cat(
        (student_prefix_embeddings, target.embed_ids(target_ids)), dim=1
    )
    student_hidden = target.hidden(student_embeddings, grad=True)
    target_tensor = torch.tensor(target_ids, device=target.device, dtype=torch.long)[
        None
    ]
    kl = frozen_lm_head_chunked_forward_kl(
        student_hidden,
        teacher_hidden,
        target_tensor,
        target.lm_head,
        student_prefix_width=int(student_prefix_embeddings.shape[1]),
        teacher_prefix_width=1,
        chunk_size=lm_head_chunk_size,
    )
    lengths = {
        "physical_source_tokens": physical_source_tokens,
        "memory_tokens": int(memory.shape[1]),
        "working_memory_tokens": len(working_ids),
        "teacher_sequence_tokens": teacher_sequence_tokens,
        "teacher_cached_chunks": teacher_cached_chunks,
        "teacher_forward_executed": bool(teacher_forward_executed),
        "student_sequence_tokens": int(student_embeddings.shape[1]),
        "supervised_tokens": len(target_ids),
        "cached_teacher_parity": (
            cached_teacher_parity if teacher_forward_executed else None
        ),
        "is_serving_state": event.is_serving_state,
    }
    return kl, memory, lengths


def _learning_rate(step: int, *, base: float, warmup: int) -> float:
    if warmup <= 0:
        return base
    return base * min(1.0, float(step + 1) / float(warmup))


def _build_adamw_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    memory_mode: str,
) -> torch.optim.Optimizer:
    if memory_mode not in OPTIMIZER_MEMORY_MODES:
        raise ValueError(f"unsupported optimizer memory mode: {memory_mode}")
    kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": weight_decay,
    }
    if memory_mode == "low_peak":
        # The foreach implementation materializes tensor lists proportional to
        # optimizer state.  The scalar loop implements the same AdamW update
        # without that transient allocation.
        kwargs["foreach"] = False
    return torch.optim.AdamW(parameters, **kwargs)


@dataclass
class WM4LowPeakAdamMomentResidency:
    """Keep only WM4 low-peak AdamW moments on CPU between optimizer uses.

    FSDP already keeps gradients sharded during normal synchronized gradient
    accumulation.  The two FP32 AdamW moments are different: after the first
    update they persist on every rank and overlap the next long WM4 forward.
    This narrowly scoped policy moves only ``exp_avg`` and ``exp_avg_sq`` to
    CPU after each optimizer use, restoring their exact bytes only after the
    preceding activation graph has been released.  It intentionally leaves
    all generic optimizer paths untouched.
    """

    enabled: bool
    restore_calls: int = 0
    offload_calls: int = 0
    cpu_moment_tensors: int = 0
    cpu_moment_bytes: int = 0
    forward_entry_checks: int = 0
    forward_entry_cache_trim_calls: int = 0
    max_forward_entry_cuda_moment_bytes: int = 0
    offload_calls_by_reason: dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_wm4_args(cls, args: argparse.Namespace) -> "WM4LowPeakAdamMomentResidency":
        return cls(
            enabled=(
                args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
                and args.optimizer_memory_mode == "low_peak"
            )
        )

    def _moment_items(
        self, optimizer: torch.optim.Optimizer
    ) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        if not self.enabled:
            return []
        if not isinstance(optimizer, torch.optim.AdamW):
            raise RuntimeError("WM4 low-peak moment residency requires AdamW")
        items: list[tuple[torch.Tensor, dict[str, Any]]] = []
        for group in optimizer.param_groups:
            parameters = group.get("params")
            if not isinstance(parameters, list):
                raise RuntimeError("WM4 AdamW parameter group is malformed")
            for parameter in parameters:
                if not isinstance(parameter, torch.Tensor):
                    raise RuntimeError("WM4 AdamW parameter is not a tensor")
                state = optimizer.state.get(parameter)
                if state is None:
                    continue
                if not isinstance(state, dict):
                    raise RuntimeError("WM4 AdamW state is not a dictionary")
                if not state:
                    continue
                if set(state) != WM4_ADAMW_MOMENT_STATE_KEYS:
                    raise RuntimeError(
                        "WM4 AdamW state keys differ from the pinned moment policy"
                    )
                for name in ("exp_avg", "exp_avg_sq"):
                    moment = state.get(name)
                    if not isinstance(moment, torch.Tensor):
                        raise RuntimeError(f"WM4 AdamW {name} is not a tensor")
                    if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                        raise RuntimeError(
                            f"WM4 AdamW {name} does not match its parameter"
                        )
                items.append((parameter, state))
        return items

    def restore_before_optimizer_use(self, optimizer: torch.optim.Optimizer) -> None:
        """Restore exact CPU moments after activations have been released."""

        if not self.enabled:
            return
        items = self._moment_items(optimizer)
        for parameter, state in items:
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                if parameter.device.type == "cpu":
                    if moment.device.type != "cpu":
                        raise RuntimeError(
                            "CPU WM4 AdamW parameter has a non-CPU moment"
                        )
                    continue
                if moment.device.type == "cpu":
                    state[name] = moment.to(
                        device=parameter.device, non_blocking=False
                    )
                elif moment.device != parameter.device:
                    raise RuntimeError(
                        "WM4 AdamW moment has an unexpected device before restore"
                    )
        self.restore_calls += 1

    def offload_after_optimizer_use(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        reason: str = "manual",
        tolerate_partial: bool = False,
    ) -> None:
        """Move only persistent AdamW moments to CPU after their last use."""

        if not self.enabled:
            return
        if not isinstance(reason, str) or not reason:
            raise RuntimeError("WM4 AdamW moment offload reason is invalid")
        if tolerate_partial:
            if not isinstance(optimizer, torch.optim.AdamW):
                raise RuntimeError("WM4 low-peak moment residency requires AdamW")
            items: list[tuple[torch.Tensor, dict[str, Any]]] = []
            for group in optimizer.param_groups:
                parameters = group.get("params")
                if not isinstance(parameters, list):
                    raise RuntimeError("WM4 AdamW parameter group is malformed")
                for parameter in parameters:
                    if not isinstance(parameter, torch.Tensor):
                        raise RuntimeError("WM4 AdamW parameter is not a tensor")
                    state = optimizer.state.get(parameter)
                    if state is None:
                        continue
                    if not isinstance(state, dict) or not set(state) <= (
                        WM4_ADAMW_MOMENT_STATE_KEYS
                    ):
                        raise RuntimeError("WM4 AdamW state is malformed during cleanup")
                    items.append((parameter, state))
        else:
            items = self._moment_items(optimizer)
        cpu_tensors = 0
        cpu_bytes = 0
        for parameter, state in items:
            for name in ("exp_avg", "exp_avg_sq"):
                if name not in state:
                    if tolerate_partial:
                        continue
                    raise RuntimeError(f"WM4 AdamW {name} is missing")
                moment = state[name]
                if parameter.device.type == "cpu":
                    if moment.device.type != "cpu":
                        raise RuntimeError(
                            "CPU WM4 AdamW parameter has a non-CPU moment"
                        )
                else:
                    if moment.device.type == "cpu":
                        pass
                    elif moment.device == parameter.device:
                        state[name] = moment.to(device="cpu", non_blocking=False)
                        moment = state[name]
                    else:
                        raise RuntimeError(
                            "WM4 AdamW moment has an unexpected device before offload"
                        )
                cpu_tensors += 1
                cpu_bytes += moment.numel() * moment.element_size()
        self.offload_calls += 1
        self.offload_calls_by_reason[reason] = (
            self.offload_calls_by_reason.get(reason, 0) + 1
        )
        self.cpu_moment_tensors = cpu_tensors
        self.cpu_moment_bytes = cpu_bytes

    def assert_cpu_resident_at_forward_entry(
        self, optimizer: torch.optim.Optimizer, *, record: bool = True
    ) -> None:
        """Fail closed on device moments and trim stale blocks before each action."""

        if not self.enabled:
            return
        cuda_bytes = 0
        for parameter, state in self._moment_items(optimizer):
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                if parameter.device.type != "cpu" and moment.device.type != "cpu":
                    cuda_bytes += moment.numel() * moment.element_size()
        if record:
            # A GA8 update traverses heterogeneous long trajectories.  The
            # completed backward graph is already detached before this hook,
            # but allocator-reserved blocks from the preceding action can
            # otherwise overlap the next 12K teacher/student forward.  Wait
            # for the prior action to retire, then return only unused cached
            # blocks to CUDA before admitting the next action.  This changes
            # neither gradients nor optimizer state.
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self.forward_entry_cache_trim_calls += 1
            self.forward_entry_checks += 1
            self.max_forward_entry_cuda_moment_bytes = max(
                self.max_forward_entry_cuda_moment_bytes, cuda_bytes
            )
        if cuda_bytes:
            raise RuntimeError(
                "WM4 forward entered with device-resident AdamW moments"
            )

    def contract_receipt(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "policy": WM4_LOW_PEAK_ADAM_MOMENT_RESIDENCY,
            "moment_keys": ["exp_avg", "exp_avg_sq"],
            "residency_between_optimizer_uses": "cpu",
            "restore_after_activation_release_before_optimizer_use": True,
            "offload_calls": self.offload_calls,
            "restore_calls": self.restore_calls,
            "cpu_moment_tensors": self.cpu_moment_tensors,
            "cpu_moment_bytes": self.cpu_moment_bytes,
            "forward_entry_checks": self.forward_entry_checks,
            "forward_entry_cache_trim": WM4_FORWARD_ENTRY_CACHE_TRIM,
            "forward_entry_cache_trim_calls": self.forward_entry_cache_trim_calls,
            "max_forward_entry_cuda_moment_bytes": (
                self.max_forward_entry_cuda_moment_bytes
            ),
            "offload_calls_by_reason": dict(sorted(self.offload_calls_by_reason.items())),
        }


def _zero_optimizer_grad(
    optimizer: torch.optim.Optimizer,
    *,
    memory_mode: str,
) -> None:
    if memory_mode not in OPTIMIZER_MEMORY_MODES:
        raise ValueError(f"unsupported optimizer memory mode: {memory_mode}")
    optimizer.zero_grad(set_to_none=True)
    if memory_mode == "low_peak":
        torch.cuda.empty_cache()


def _step_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    memory_mode: str,
    wm4_low_peak_moment_residency: WM4LowPeakAdamMomentResidency | None = None,
) -> None:
    if memory_mode not in OPTIMIZER_MEMORY_MODES:
        raise ValueError(f"unsupported optimizer memory mode: {memory_mode}")
    if memory_mode == "low_peak":
        torch.cuda.empty_cache()
    try:
        if wm4_low_peak_moment_residency is not None:
            wm4_low_peak_moment_residency.restore_before_optimizer_use(optimizer)
        optimizer.step()
    finally:
        if wm4_low_peak_moment_residency is not None:
            # Restore can partially succeed before an allocation or optimizer
            # failure.  The idempotent offload converges either mixed state
            # back to CPU residency without touching model parameters.
            wm4_low_peak_moment_residency.offload_after_optimizer_use(
                optimizer, reason="after_step"
            )
        if memory_mode == "low_peak":
            torch.cuda.empty_cache()
    _zero_optimizer_grad(optimizer, memory_mode=memory_mode)


def _normalize_accumulated_gradients(
    parameters: Iterator[torch.nn.Parameter],
    accumulated_examples: int,
) -> None:
    """Convert a sum of per-example mean losses to the exact batch mean."""

    if accumulated_examples <= 0:
        raise ValueError("accumulated_examples must be positive")
    inverse = 1.0 / float(accumulated_examples)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)


def _context_budget_contract(
    args: argparse.Namespace,
    target: FrozenTP2Target,
) -> dict[str, Any]:
    """Describe the teacher path without conflating native and run-time caps."""

    if not _objective_uses_raw_teacher(getattr(args, "objective", "kl")):
        if target.cached_teacher_enabled:
            raise RuntimeError("CE-only target unexpectedly enabled raw-teacher cache")
        return {
            "target_native_context_tokens": 262_144,
            "teacher_sequence_hard_cap_tokens": None,
            "teacher_execution_mode": "disabled_for_ce_only",
            "teacher_monolithic_forward_cap_tokens": None,
            "teacher_cached_prefill_chunk_tokens": None,
            "teacher_cached_chunk_schedule": None,
            "student_single_forward_tokens": args.max_target_sequence_tokens,
            "teacher_cache": "disabled_no_raw_actor_teacher_forward",
            "linear_attention_chunk_patch": {"enabled": False, "patched_layers": 0},
            "full_attention_cached_continuation_patch": {
                "enabled": False,
                "patched_layers": 0,
            },
            "sequence_parallelism": None,
        }
    if args.teacher_execution_mode == "monolithic_only":
        if target.cached_teacher_enabled:
            raise RuntimeError("monolithic-only target unexpectedly enabled cache")
        if (
            target.chunkable_linear_cache_layers
            or target.cached_lower_right_sdpa_layers
        ):
            raise RuntimeError(
                "monolithic-only target unexpectedly installed cache patches"
            )
        return {
            "target_native_context_tokens": 262_144,
            "teacher_sequence_hard_cap_tokens": args.max_teacher_sequence_tokens,
            "teacher_execution_mode": "monolithic_only",
            "teacher_monolithic_forward_cap_tokens": (
                args.teacher_prefill_chunk_tokens
            ),
            "teacher_cached_prefill_chunk_tokens": None,
            "teacher_cached_chunk_schedule": None,
            "student_single_forward_tokens": args.max_target_sequence_tokens,
            "teacher_cache": "disabled_no_past_key_values",
            "linear_attention_chunk_patch": {
                "enabled": False,
                "patched_layers": 0,
            },
            "full_attention_cached_continuation_patch": {
                "enabled": False,
                "patched_layers": 0,
            },
            "sequence_parallelism": None,
        }
    if args.teacher_execution_mode != "cached":
        raise ValueError(
            f"unsupported teacher execution mode: {args.teacher_execution_mode}"
        )
    if not target.cached_teacher_enabled:
        raise RuntimeError("cached teacher target did not enable cache support")
    return {
        "target_native_context_tokens": args.max_teacher_sequence_tokens,
        "teacher_sequence_hard_cap_tokens": args.max_teacher_sequence_tokens,
        "teacher_execution_mode": "cached",
        "teacher_cached_prefill_chunk_tokens": args.teacher_prefill_chunk_tokens,
        "teacher_cached_chunk_schedule": (
            "right_aligned_max_width_final_chunk_with_global_64_token_"
            "linear_attention_boundaries_and_full_target_shape"
        ),
        "student_single_forward_tokens": args.max_target_sequence_tokens,
        "teacher_cache": (
            "native_Qwen3_5_hybrid_full_attention_KV_plus_linear_recurrent_state"
        ),
        "linear_attention_chunk_patch": {
            "reason": "Transformers_5.3_stock_cache_only_carries_state_for_seq_len_1",
            "algorithm": "causal_conv_initial_states_plus_gated_delta_initial_state",
            "patched_layers": target.chunkable_linear_cache_layers,
            "parity_metric": "KL(monolithic_teacher||cached_teacher)_per_token",
            "maximum_output_distribution_kl": args.cached_teacher_max_output_kl,
            "minimum_top1_agreement": args.cached_teacher_min_top1_agreement,
            "hidden_difference_is_diagnostic_only": True,
        },
        "full_attention_cached_continuation_patch": {
            "scope": "unpadded_cached_teacher_non_square_forwards_only",
            "algorithm": QWEN35_CACHED_FULL_ATTENTION_PATCH,
            "registry_key": _QWEN35_CACHED_SDPA_IMPLEMENTATION,
            "square_or_inactive_forward": "transformers_stock_sdpa",
            "square_or_inactive_mask": "transformers_stock_sdpa_mask",
            "cached_continuation_mask": "non_materialized_torch_CausalBias",
            "patched_layers": target.cached_lower_right_sdpa_layers,
        },
        "sequence_parallelism": None,
    }


def _validate_training_input_identity(args: argparse.Namespace) -> dict[str, str | None]:
    """Bind raw data bytes and, for WM4, manifest semantics before CUDA/NCCL.

    ``--expected-data-sha256`` remains the generic transport-byte pin.  WM4's
    self-hashed JSON manifest has an additional semantic pin so canonical JSON
    never gets compared to the raw manifest file digest.
    """

    file_sha256 = _sha256(args.data)
    if args.expected_data_sha256 and file_sha256 != args.expected_data_sha256:
        raise ValueError(
            "data file SHA256 mismatch: "
            f"expected {args.expected_data_sha256}, got {file_sha256}"
        )
    receipt: dict[str, str | None] = {
        "data_file_sha256": file_sha256,
        "data_manifest_canonical_sha256": None,
    }
    if args.data_contract != QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        return receipt

    from context_compression.qwen38_coding_trajectory_data import (
        load_frozen_trajectory_manifest,
    )

    manifest = load_frozen_trajectory_manifest(args.data, verify_files=True)
    canonical_sha256 = manifest.get("canonical_sha256")
    if canonical_sha256 != args.expected_data_manifest_canonical_sha256:
        raise ValueError(
            "data manifest canonical SHA256 mismatch: "
            f"expected {args.expected_data_manifest_canonical_sha256}, "
            f"got {canonical_sha256}"
        )
    receipt["data_manifest_canonical_sha256"] = canonical_sha256
    return receipt


def train(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    input_identity = _validate_training_input_identity(args)
    if args.validate_input_only:
        print(
            json.dumps(
                {
                    "status": "input_identity_validated_cpu_only",
                    **input_identity,
                },
                sort_keys=True,
            )
        )
        return 0
    checkpoint_schema = _checkpoint_schema_for_data_contract(
        args.target_family,
        args.data_contract,
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("this experiment requires exactly two TP/FSDP ranks")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    if (
        args.resume_transition == QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION
        and args.resume_from is not None
        and (
            output_dir == args.resume_from.expanduser().resolve().parent
            or args.resume_from.expanduser().resolve().parent in output_dir.parents
        )
    ):
        raise ValueError(
            "the progressive transition must write to a new output directory"
        )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    data_digest_box = [input_identity["data_file_sha256"] if rank == 0 else None]
    dist.broadcast_object_list(data_digest_box, src=0)
    data_sha256 = str(data_digest_box[0])
    if args.expected_data_sha256 and data_sha256 != args.expected_data_sha256:
        raise ValueError(
            "data file SHA256 mismatch after distributed broadcast: "
            f"expected {args.expected_data_sha256}, got {data_sha256}"
        )
    progressive_resume_box = [_progressive_resume_context(args) if rank == 0 else None]
    dist.broadcast_object_list(progressive_resume_box, src=0)
    progressive_resume_context = progressive_resume_box[0]
    if args.recursive_event_caps is not None and not isinstance(
        progressive_resume_context, dict
    ):
        raise RuntimeError("progressive resume context broadcast failed")
    if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT and (
        args.data.stat().st_size != MINDFORGE_DATASET_FILE_BYTES
        or data_sha256 != MINDFORGE_DATASET_FILE_SHA256
    ):
        raise ValueError(
            "MindForge source does not match the pinned immutable JSONL object"
        )

    init_receipt_box = [
        (
            _portable_compressor_init_receipt(args.init_compressor)
            if rank == 0 and args.init_compressor is not None
            else None
        )
    ]
    dist.broadcast_object_list(init_receipt_box, src=0)
    init_compressor_receipt = init_receipt_box[0]
    if args.init_compressor is not None and not isinstance(
        init_compressor_receipt, dict
    ):
        raise RuntimeError("init-compressor receipt broadcast failed")

    agentic_audit_box = [
        (
            _validate_agentic_data_audit(args, data_sha256=data_sha256)
            if rank == 0 and args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT
            else None
        )
    ]
    dist.broadcast_object_list(agentic_audit_box, src=0)
    agentic_audit_receipt = agentic_audit_box[0]
    if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT and not isinstance(
        agentic_audit_receipt, dict
    ):
        raise RuntimeError("agentic data audit receipt broadcast failed")

    target_identity_box = [
        _target_content_identity(args.target_model) if rank == 0 else None
    ]
    dist.broadcast_object_list(target_identity_box, src=0)
    target_content_identity = target_identity_box[0]
    if not isinstance(target_content_identity, dict):
        raise RuntimeError("target content identity broadcast failed")
    vibe_initialization_receipt: dict[str, Any] | None = None
    if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        if args.resume_from is None:
            if not isinstance(init_compressor_receipt, dict):
                raise RuntimeError("Qwen3.8 vibe step-1000 init receipt is missing")
            init_compressor_receipt = _validate_vibe_step1000_init_receipt(
                init_compressor_receipt,
                target_content_identity=target_content_identity,
            )
            vibe_initialization_receipt = init_compressor_receipt
        elif init_compressor_receipt is not None:
            raise RuntimeError("Qwen3.8 vibe resume unexpectedly has init weights")
        else:
            vibe_initialization_receipt = _vibe_resume_initialization_receipt(
                args.resume_from,
                target_content_identity=target_content_identity,
            )
    target_manifest_sha256 = next(
        item["sha256"]
        for item in target_content_identity["metadata"]
        if item["name"] == "text_snapshot_manifest.json"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    parity_descriptor = _resolve_parity_descriptor(args, tokenizer)
    parity_descriptors: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(parity_descriptors, parity_descriptor)
    if any(descriptor != parity_descriptors[0] for descriptor in parity_descriptors):
        raise RuntimeError("cached-teacher parity descriptor differs across ranks")
    parity_descriptor = parity_descriptors[0]

    implementation_identity = _implementation_identity(args.data_contract)
    implementation_identities: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(implementation_identities, implementation_identity)
    if any(
        identity != implementation_identities[0]
        for identity in implementation_identities
    ):
        raise RuntimeError("training implementation identity differs across ranks")
    implementation_identity = implementation_identities[0]

    mindforge_manifest: dict[str, Any] | None = None
    mindforge_manifest_receipt: dict[str, Any] | None = None
    mindforge_train_rows: tuple[dict[str, Any], ...] = ()
    if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
        manifest_box: list[Any] = [None]
        if rank == 0:
            from context_compression.mindforge_ce_manifest import (
                load_and_validate_ce_manifest,
            )

            loaded_manifest = load_and_validate_ce_manifest(
                args.mindforge_event_manifest,
                expected_source_path=args.data,
            )
            loaded_receipt = _validate_mindforge_manifest_training_contract(
                loaded_manifest,
                manifest_path=args.mindforge_event_manifest,
                args=args,
                target_content_identity=target_content_identity,
                adapter_source_sha256=implementation_identity["local_files"][
                    "mindforge_data"
                ]["sha256"],
                tokenizer=tokenizer,
                init_compressor_receipt=init_compressor_receipt,
            )
            manifest_box[0] = (loaded_manifest, loaded_receipt)
        dist.broadcast_object_list(manifest_box, src=0)
        manifest_payload = manifest_box[0]
        if (
            not isinstance(manifest_payload, tuple)
            or len(manifest_payload) != 2
            or not isinstance(manifest_payload[0], dict)
            or not isinstance(manifest_payload[1], dict)
        ):
            raise RuntimeError("MindForge decision manifest broadcast failed")
        mindforge_manifest, mindforge_manifest_receipt = manifest_payload
        from context_compression.mindforge_ce_manifest import iter_manifest_rows

        mindforge_train_rows = tuple(
            dict(row) for row in iter_manifest_rows(mindforge_manifest, "train")
        )
        if len(mindforge_train_rows) != int(
            mindforge_manifest_receipt["train_trajectories"]
        ):
            raise RuntimeError("MindForge train row count differs after broadcast")

    vibe_manifest: dict[str, Any] | None = None
    vibe_manifest_receipt: dict[str, Any] | None = None
    vibe_descriptors_by_trajectory: dict[str, tuple[dict[str, Any], ...]] = {}
    if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        vibe_manifest_box: list[Any] = [None]
        if rank == 0:
            from context_compression.qwen38_coding_trajectory_data import (
                load_frozen_trajectory_manifest,
            )

            late_manifest_file_sha256 = _sha256(args.data)
            if late_manifest_file_sha256 != data_sha256:
                raise RuntimeError(
                    "Qwen3.8 vibe manifest changed after CPU input validation"
                )
            loaded_vibe_manifest = load_frozen_trajectory_manifest(
                args.data,
                verify_files=True,
            )
            if (
                loaded_vibe_manifest.get("canonical_sha256")
                != args.expected_data_manifest_canonical_sha256
            ):
                raise RuntimeError(
                    "Qwen3.8 vibe manifest canonical identity changed after CPU "
                    "input validation"
                )
            loaded_vibe_receipt, loaded_vibe_descriptors = (
                _validate_vibe_manifest_training_contract(
                    loaded_vibe_manifest,
                    manifest_path=args.data,
                    manifest_file_sha256=data_sha256,
                    args=args,
                    adapter_source_sha256=implementation_identity["local_files"][
                        "vibe_trajectory_data"
                    ]["sha256"],
                )
            )
            vibe_manifest_box[0] = (
                loaded_vibe_manifest,
                loaded_vibe_receipt,
                loaded_vibe_descriptors,
            )
        dist.broadcast_object_list(vibe_manifest_box, src=0)
        vibe_payload = vibe_manifest_box[0]
        if (
            not isinstance(vibe_payload, tuple)
            or len(vibe_payload) != 3
            or not isinstance(vibe_payload[0], dict)
            or not isinstance(vibe_payload[1], dict)
            or not isinstance(vibe_payload[2], dict)
        ):
            raise RuntimeError("Qwen3.8 vibe frozen manifest broadcast failed")
        (
            vibe_manifest,
            vibe_manifest_receipt,
            vibe_descriptors_by_trajectory,
        ) = vibe_payload

    parity_evidence_receipts: list[dict[str, Any]] = []
    if args.cached_teacher_parity_artifact:
        if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
            parity_evidence_receipts = (
                _validate_cross_dataset_cached_teacher_parity_artifacts(
                    args.cached_teacher_parity_artifact,
                    required_chunk_tokens=args.required_parity_chunk_tokens,
                    target_model=args.target_model,
                    target_manifest_sha256=target_manifest_sha256,
                    target_content_identity=target_content_identity,
                    implementation_identity=implementation_identity,
                    maximum_output_distribution_kl=(args.cached_teacher_max_output_kl),
                    minimum_top1_agreement=(args.cached_teacher_min_top1_agreement),
                )
            )
        else:
            from context_compression.qwen36_teacher_parity import (
                validate_cached_teacher_parity_artifacts,
            )

            parity_evidence_receipts = validate_cached_teacher_parity_artifacts(
                args.cached_teacher_parity_artifact,
                required_chunk_tokens=args.required_parity_chunk_tokens,
                data_path=args.data,
                data_sha256=data_sha256,
                target_model=args.target_model,
                target_manifest_sha256=target_manifest_sha256,
                target_content_identity=target_content_identity,
                implementation_identity=implementation_identity,
                maximum_output_distribution_kl=args.cached_teacher_max_output_kl,
                minimum_top1_agreement=args.cached_teacher_min_top1_agreement,
            )

    model = load_qwen35_text_target(
        args.target_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        tp_plan="auto",
    )
    target = FrozenTP2Target(
        model,
        gradient_checkpointing=not args.disable_target_gradient_checkpointing,
        cached_teacher_enabled=(
            _objective_uses_raw_teacher(args.objective)
            and args.teacher_execution_mode == "cached"
        ),
    )
    with torch.no_grad():
        embedding_sample = _full_tensor(model.get_input_embeddings().weight)[:4096]
        output_embedding_rms = float(
            embedding_sample.float().square().mean().sqrt().item()
        )

    compressor_base = _build_compressor_for_target_family(
        args.target_family,
        output_embedding_rms=output_embedding_rms,
        gradient_checkpointing=not args.disable_compressor_gradient_checkpointing,
    ).float()
    if rank == 0 and args.init_compressor is not None:
        if not isinstance(init_compressor_receipt, dict):
            raise RuntimeError("init-compressor receipt is missing on rank zero")
        _load_portable_compressor_weights_only(
            compressor_base,
            args.init_compressor,
            receipt=init_compressor_receipt,
        )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    compressor = FSDP(
        compressor_base,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    if any(parameter.dtype != torch.float32 for parameter in compressor.parameters()):
        raise RuntimeError("FSDP compressor master parameters must remain FP32")
    optimizer = _build_adamw_optimizer(
        compressor.parameters(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        memory_mode=args.optimizer_memory_mode,
    )
    wm4_low_peak_moment_residency = WM4LowPeakAdamMomentResidency.for_wm4_args(
        args
    )
    compressor.train()

    contract_args = _jsonable_args(args)
    for non_semantic in (
        "output_dir",
        "resume_from",
        "resume_at_recursive_epoch",
        "resume_transition",
        "recursive_event_caps",
        "max_steps",
        "log_interval",
        "save_interval",
        "skip_final_checkpoint",
        "verify_parameters_finite_after_step",
        "objective",
        "kl_loss_weight",
        "mindforge_event_manifest",
        "expected_mindforge_manifest_sha256",
        "expected_mindforge_manifest_contract_sha256",
        "mindforge_history_window_tokens",
        "mindforge_max_action_tokens",
        "init_compressor",
    ):
        contract_args.pop(non_semantic, None)
    recursive_event_caps = _parse_recursive_event_caps(args.recursive_event_caps)
    if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
        data_contract_receipt = {
            "name": QWEN36_AGENTIC_DATA_CONTRACT,
            "source_format": "parquet_exact_ordered_columns_tools_messages",
            "source_dataset": {
                "id": args.source_dataset_id,
                "revision": args.source_dataset_revision,
            },
            "normalization": (
                "strict_json_v1_tool_arguments_object_reasoning_to_"
                "reasoning_content_drop_null_optional_fields_reject_"
                "undeclared_or_duplicate_tool_names"
            ),
            "expected_rows": {
                "total": args.expected_agentic_total_rows,
                "accepted": args.expected_agentic_valid_rows,
                "rejected": args.expected_agentic_rejected_rows,
                "reject_categories": _parse_expected_reject_categories(
                    args.expected_agentic_reject_category
                ),
            },
            "adapter_source_sha256": implementation_identity["local_files"][
                "agentic_data"
            ]["sha256"],
            "native_chat_template": {
                "enable_thinking": True,
                "preserve_thinking": QWEN36_PRESERVE_THINKING,
                "upstream_sampling_report": (
                    "enable_thinking=true; preserve_thinking was not reported"
                ),
                "local_abi_choice": (
                    "preserve_thinking=true for append-only full-trajectory serving"
                ),
                "tool_result_batch_closing_text": ("\\n</tool_response><|im_end|>\\n"),
                "tool_result_batch_closing_ids": list(
                    QWEN36_TOOL_RESULT_BATCH_CLOSING_IDS
                ),
            },
            "eligibility": (
                "at_least_two_complete_maximal_tool_result_batches_no_fallback"
            ),
            "eviction": (
                "history_zero_through_penultimate_tool_batch_closure_"
                "chunks_at_most_1024_never_cross_batch_closure"
            ),
            "target_provenance": {
                "final_assistant": ("dataset_selected_A3B_chosen_completion_only"),
                "preceding_assistant_tool_and_user_messages": (
                    "prompt_context_only_never_KL_targets"
                ),
                "online_teacher": (
                    "frozen_local_target_one_raw_forward_per_row_reused_for_events"
                ),
                "upstream_generator": "Qwen/Qwen3.6-35B-A3B-FP8",
                "unknown_upstream_fields": (
                    "generator revision, sampling seed, vLLM version, and numeric top_p"
                ),
                "policy_relation": (
                    "same-family regenerated offline context approximation; local "
                    "BF16 teacher is not proven bit-identical or exact on-policy"
                ),
            },
            "chosen_target_window": {
                "policy": "early_prefix_of_final_chosen_assistant_continuation",
                "maximum_tokens": args.kl_tokens,
                "action_boundary_aware": False,
                "limitation": (
                    "the window may truncate or precede part of a long tool call; "
                    "the pinned audit must match this exact target cap"
                ),
            },
        }
        curriculum_contract = {
            "epoch_1": ("one_random_legal_variable_chunk_without_recurrent_input"),
            "epoch_2_plus": (
                "all_legal_contiguous_chunks_stop_gradient_between_state_updates_"
                "no_recursion_cap"
            ),
        }
        prefix_contract = "native_tools_plus_complete_initial_system_user_turn"
        raw_working_memory_contract: int | str = (
            "variable_raw_suffix_from_chunk_stop_through_final_generation_prompt"
        )
        trajectory_supervision = (
            "A3B regenerated chosen final assistant supplies the sole target span; "
            "the frozen target distribution is computed online once per row"
        )
        reasoning_contract = (
            "reasoning maps to reasoning_content and native preserve_thinking is true"
        )
    elif args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
        if not isinstance(mindforge_manifest_receipt, dict):
            raise RuntimeError("MindForge manifest receipt is missing")
        data_contract_receipt = {
            "name": MINDFORGE_FRONTIER_CE_DATA_CONTRACT,
            "source_format": "pinned_jsonl_byte_seek_exact_manifest_set_v1",
            "pilot_profile": args.mindforge_pilot_profile,
            "source_dataset": {
                "id": MINDFORGE_DATASET_REPO,
                "revision": MINDFORGE_DATASET_REVISION,
                "file": MINDFORGE_DATASET_FILE,
                "size_bytes": MINDFORGE_DATASET_FILE_BYTES,
                "sha256": MINDFORGE_DATASET_FILE_SHA256,
            },
            "adapter_contract": MINDFORGE_CONTRACT,
            "manifest": mindforge_manifest_receipt,
            "pilot_geometry": (
                MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT_H4096
                if args.mindforge_pilot_profile == MINDFORGE_FULL_VIEW_H4096_PROFILE
                else MINDFORGE_FRONTIER_ACTION_PILOT_CONTRACT
            ),
            "runtime_abi": MINDFORGE_RUNTIME_ABI,
            "history_window": {
                "tokens": args.mindforge_history_window_tokens,
                "history_before_window_excluded": True,
                "initial_memory": None,
                "long_trajectory_position_used_for_action_selection_only": True,
                "scope": (
                    f"cold_start_{args.mindforge_history_window_tokens}_token_"
                    "compatibility_pilot_not_full_"
                    "trajectory_recurrent_memory"
                ),
            },
            "action_target": {
                "source": "GLM-5.2_frontier_assistant_action",
                "native_reasoning_content_and_content_and_tool_calls": True,
                "maximum_complete_tokens": args.mindforge_max_action_tokens,
                "truncation": "forbidden_fail_closed",
                "prompt_and_tool_observation_labels": -100,
                "harness_submit_literal_targets": "excluded_by_manifest",
            },
            "weighting": {
                "trajectory": "unit_mass",
                "selected_actions": "uniform_1_over_A",
                "eligible_compression_states": "uniform_1_over_K_within_action",
                "token_loss": "mean_within_complete_action",
            },
        }
        curriculum_contract = {
            "all_epochs": (
                "recursive_stopgrad_cold_start_window; replay_only_states_over_"
                "sequence_cap; every_eligible_1024_eviction_directly_action_"
                "supervised; no_BPTT"
            )
        }
        prefix_contract = "native_tools_plus_complete_initial_system_user_turn"
        raw_working_memory_contract = RAW_WORKING_MEMORY_TOKENS
        trajectory_supervision = (
            "four manifest-selected late GLM-5.2 actions per trajectory; late "
            "position selects actions only; model context is the pinned cold-start "
            f"{args.mindforge_history_window_tokens}-token suffix"
        )
        reasoning_contract = (
            "native Qwen3.8 rendering of GLM reasoning/content/tool calls; full "
            "assistant continuation only"
        )
    elif args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        if not isinstance(vibe_manifest_receipt, dict):
            raise RuntimeError("Qwen3.8 vibe manifest receipt is missing")
        data_contract_receipt = dict(vibe_manifest_receipt)
        data_contract_receipt.update(
            {
                "serving_state_layout": (
                    "system+user+soft_memory+last_four_complete_rounds+"
                    "assistant_generation_boundary_to_assistant_output"
                ),
                "recurrence": {
                    "fold_order": ("oldest_evicted_complete_round_first_left_to_right"),
                    "fold_width_tokens": "1_to_1024_never_cross_round",
                    "previous_soft_memory": "detached_before_every_next_fold",
                    "gradient_folds_per_action": 1,
                    "gradient_fold": "final_fold_only",
                },
                "recurrence_gradient": "stopgrad",
                "supervision": {
                    "actions_per_cap_safe_trajectory_per_epoch": 1,
                    "selection": "progressive_monotone_depth_quantiles",
                    "assistant_target_only": True,
                    "generation_boundary_targeted": False,
                    "final_serving_state_only": True,
                    "teacher_forwards_per_selected_action": 1,
                    "KL_objectives_per_selected_action": 1,
                },
                "sequence_cap_policy": (
                    "skip_whole_action_and_record_coverage_never_crop_complete_round"
                ),
            }
        )
        curriculum_contract = {
            "all_10_epochs": (
                "one_deterministic_eligible_final_serving_state_action_per_"
                "cap_safe_trajectory; replay_all_prior_eviction_folds_stopgrad; "
                "final_fold_only_grad; exactly_one_action_KL"
            )
        }
        prefix_contract = (
            "native_tools_plus_complete_initial_system_user_then_soft_memory_then_"
            "last_four_complete_rounds_then_native_assistant_generation_boundary"
        )
        raw_working_memory_contract = {
            "unit": "complete_normalized_assistant_tool_rounds",
            "rounds": args.vibe_working_memory_rounds,
            "parallel_tool_calls": "canonical_single_tool_observation_batch",
            "token_truncation": "forbidden",
        }
        trajectory_supervision = (
            "accepted coding trajectory supplies context and one deterministic "
            "assistant-only target per trajectory/epoch; frozen Qwen3.8 target "
            "distribution is computed online from the full raw serving history"
        )
        reasoning_contract = (
            "native Qwen3.8 reasoning/content/tool-call rendering; only the selected "
            "assistant continuation after the native generation boundary is targeted"
        )
    else:
        data_contract_receipt = {
            "name": LEGACY_TOKEN_CLOCK_DATA_CONTRACT,
            "source_format": "jsonl_conversations",
        }
        if recursive_event_caps:
            curriculum_contract = {
                "schema": "progressive_recursive_event_cap_v1",
                "epoch_1": (
                    "source_checkpoint_one_shot_prefix_skipped_by_explicit_transition"
                ),
                "epoch_2_to_11": {
                    "maximum_chronological_events": [
                        "all" if cap is None else cap for cap in recursive_event_caps
                    ],
                    "cap_application_order": (
                        "teacher_and_student_sequence_caps_then_causal_prefix_cap"
                    ),
                    "recurrence": (
                        "previous_compressor_memory_plus_next_1024_block_with_"
                        "stop_gradient_between_state_updates"
                    ),
                    "loss_weighting": (
                        "mean_over_actual_selected_events_then_mean_over_examples"
                    ),
                },
            }
        else:
            curriculum_contract = {
                "epoch_1": (
                    "one_random_legal_1024_to_64_compression_without_recurrent_input"
                ),
                "epoch_2_plus": (
                    "all_legal_events_stop_gradient_between_state_updates_no_recursion_cap"
                ),
            }
        prefix_contract = (
            "native_complete_system_and_initial_user_add_generation_prompt_false"
        )
        raw_working_memory_contract = RAW_WORKING_MEMORY_TOKENS
        trajectory_supervision = (
            "offline regenerated SFT trajectory supplies token-clock contexts only; "
            "the frozen target distribution is computed online at every event"
        )
        reasoning_contract = "preserve native reasoning_content when present; content remains the final answer"
    progressive_lineage = (
        json.loads(json.dumps(progressive_resume_context["lineage"]))
        if isinstance(progressive_resume_context, dict)
        else None
    )
    if progressive_lineage is not None:
        _validate_progressive_lineage(progressive_lineage)
        transition_source_run_config = progressive_resume_context.get(
            "transition_source_run_config"
        )
        if transition_source_run_config is not None:
            source_implementation_identity = transition_source_run_config.get(
                "implementation_identity"
            )
            if not isinstance(source_implementation_identity, dict):
                raise ValueError(
                    "progressive transition source implementation identity is invalid"
                )
            progressive_lineage["implementation_transition"] = (
                _progressive_implementation_transition_receipt(
                    source_identity=source_implementation_identity,
                    destination_identity=implementation_identity,
                    expected_trainer_sha256=(args.expected_progressive_trainer_sha256),
                )
            )
        _validate_progressive_implementation_transition_receipt(
            progressive_lineage.get("implementation_transition"),
            destination_identity=implementation_identity,
            expected_trainer_sha256=args.expected_progressive_trainer_sha256,
        )
    run_contract = {
        "format_version": FORMAT_VERSION,
        "schema": checkpoint_schema,
        "target_family": args.target_family,
        "target_model": str(args.target_model.resolve()),
        "target_text_snapshot_manifest_sha256": target_manifest_sha256,
        "target_content_identity": target_content_identity,
        **(
            {
                "target_loader_runtime_receipt": (
                    _qwen38_progressive_target_loader_runtime_receipt(
                        model,
                        target_content_identity=target_content_identity,
                    )
                )
            }
            if recursive_event_caps
            else {}
        ),
        "data": str(args.data.resolve()),
        "data_stat": {
            "size": args.data.stat().st_size,
            "mtime_ns": args.data.stat().st_mtime_ns,
        },
        "data_sha256": data_sha256,
        "agentic_data_audit": agentic_audit_receipt,
        "data_contract": data_contract_receipt,
        "curriculum": curriculum_contract,
        **(
            {"resume_lineage": progressive_lineage}
            if progressive_lineage is not None
            else {}
        ),
        "teacher_execution_mode": (
            args.teacher_execution_mode
            if _objective_uses_raw_teacher(args.objective)
            else "disabled_for_ce_only"
        ),
        "context_budget": _context_budget_contract(args, target),
        "fixed_probe_selector": parity_descriptor,
        "cached_teacher_parity_selector": (
            parity_descriptor if args.verify_cached_teacher else None
        ),
        "cached_teacher_formal_evidence": parity_evidence_receipts,
        "cached_teacher_parity_evidence_scope": (
            "implementation_only_cross_dataset"
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            else (
                "exact_current_data_and_implementation"
                if parity_evidence_receipts
                else None
            )
        ),
        "implementation_identity": implementation_identity,
        "prefix": prefix_contract,
        "eviction_block_tokens": EVICTION_BLOCK_TOKENS,
        "raw_working_memory_tokens": raw_working_memory_contract,
        "compression_ratio": COMPRESSION_RATIO,
        "target_layer_ids": list(target.target_layer_ids),
        "compressor_architecture_source": _compressor_architecture_source(
            args.target_family
        ),
        "compressor_architecture_upstream_receipt": (
            _compressor_architecture_provenance(args.target_family)
        ),
        "trajectory_supervision": trajectory_supervision,
        "assistant_reasoning_contract": reasoning_contract,
        **(
            {
                "data_manifest_file_sha256": data_sha256,
                "data_manifest_canonical_sha256": vibe_manifest_receipt[
                    "manifest_canonical_sha256"
                ],
                "final_serving_state_only": True,
                "working_memory_rounds": args.vibe_working_memory_rounds,
                "recurrence_gradient": "stopgrad",
                "teacher_prefix_contract": (
                    "system_user_plus_all_complete_raw_rounds_plus_generation_boundary"
                ),
                "student_prefix_contract": (
                    "system_user_plus_soft_memory_plus_last_four_complete_rounds_"
                    "plus_generation_boundary"
                ),
            }
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            else {}
        ),
        "compressor": compressor.module.export_config(),
        "compressor_initialization": (
            vibe_initialization_receipt
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            else (
                init_compressor_receipt
                if init_compressor_receipt is not None
                else (
                    {
                        "source": "resume_lineage",
                        "source_global_step": progressive_lineage["source_global_step"],
                        "restored_state": progressive_lineage["restored_state"],
                    }
                    if progressive_lineage is not None
                    else (
                        "fresh_random_matching_architecture_at_run_origin; "
                        "resume_requires_the_same_run_contract"
                    )
                )
            )
        ),
        "objective": (
            "exact_full_vocabulary_forward_kl_teacher_raw_to_student_compressed_T1_no_CE"
            if args.objective == "kl"
            else {
                "name": args.objective,
                "ce": "complete_native_frontier_assistant_action_only",
                "kl": (
                    "exact_full_vocabulary_forward_kl_on_same_complete_action"
                    if args.objective == "ce_kl"
                    else None
                ),
                "kl_loss_weight": (
                    args.kl_loss_weight if args.objective == "ce_kl" else None
                ),
                "prompt_and_tool_observation_labels": -100,
            }
        ),
        "gradient_accumulation_contract": (
            "one_final_action_per_trajectory_then_mean_over_actual_trajectories; "
            "short_epoch_tail_is_flushed; checkpoint_cursor_is_next_frozen_"
            "manifest_training_order"
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            else "mean_over_events_within_each_example_then_mean_over_actual_examples; "
            "short_epoch_tail_is_flushed_before_curriculum_transition; "
            "checkpoint_cursor_points_after_the_flushed_rows"
        ),
        "kl_tokens_per_event": (
            args.kl_tokens
            if args.data_contract != MINDFORGE_FRONTIER_CE_DATA_CONTRACT
            else None
        ),
        "precision": "FP32_FSDP_master_and_Adam_with_BF16_forward",
        "distributed": "TP2_frozen_target_plus_FULL_SHARD_compressor_same_two_ranks",
        "args": contract_args,
        **(
            {
                "wm4_low_peak_adam_moment_residency": (
                    WM4_LOW_PEAK_ADAM_MOMENT_RESIDENCY
                ),
                "wm4_forward_entry_cache_trim": WM4_FORWARD_ENTRY_CACHE_TRIM,
            }
            if wm4_low_peak_moment_residency.enabled
            else {}
        ),
    }
    transition_source_run_config = (
        progressive_resume_context.get("transition_source_run_config")
        if isinstance(progressive_resume_context, dict)
        else None
    )
    if transition_source_run_config is not None:
        if not isinstance(transition_source_run_config, dict):
            raise RuntimeError("progressive transition source contract is invalid")
        _validate_qwen38_progressive_destination(
            transition_source_run_config,
            run_contract,
        )
    wm4_resume_binding = (
        _wm4_resume_binding(
            run_contract,
            operational_controls=_wm4_resume_operational_controls(args),
        )
        if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
        else None
    )
    wm4_resume_source_checkpoint_identity = (
        _wm4_resume_checkpoint_identity(args.resume_from)
        if (
            args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            and args.resume_from is not None
        )
        else None
    )
    run_contract_sha256 = _run_contract_sha256(run_contract)
    resume_checkpoint_run_contract_sha256 = (
        progressive_resume_context.get("resume_checkpoint_run_contract_sha256")
        if isinstance(progressive_resume_context, dict)
        else None
    )
    if (
        resume_checkpoint_run_contract_sha256 is not None
        and resume_checkpoint_run_contract_sha256 != run_contract_sha256
    ):
        raise ValueError(
            "progressive checkpoint run contract does not match this invocation"
        )
    run_config = {
        **run_contract,
        "run_contract_sha256": run_contract_sha256,
        "output_dir": str(output_dir),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "init_compressor": (
            str(args.init_compressor.resolve()) if args.init_compressor else None
        ),
        "resume_cursor_override": (
            "skip_remaining_epoch_1_and_start_epoch_2_line_0"
            if args.resume_at_recursive_epoch
            else None
        ),
        **(
            {"wm4_resume_binding": wm4_resume_binding}
            if wm4_resume_binding is not None
            else {}
        ),
        **(
            {
                "wm4_resume_source_checkpoint_identity": (
                    wm4_resume_source_checkpoint_identity
                )
            }
            if wm4_resume_source_checkpoint_identity is not None
            else {}
        ),
        **(
            {"resume_transition_invocation": args.resume_transition}
            if progressive_lineage is not None
            else {}
        ),
    }
    if rank == 0:
        _atomic_json(output_dir / "run_config.json", run_config)
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None

    global_step = 0
    start_epoch = 0
    start_line_index = 0
    if args.resume_from is not None:
        initial_progressive_transition = (
            args.resume_transition == QWEN38_STEP1000_TO_PROGRESSIVE_TRANSITION
        )
        checkpoint_run_contract_sha256 = (
            progressive_lineage["source_run_contract_sha256"]
            if initial_progressive_transition
            else run_contract_sha256
        )
        checkpoint_schema_to_load = (
            progressive_lineage["source_checkpoint_schema"]
            if initial_progressive_transition
            else checkpoint_schema
        )
        global_step, start_epoch, start_line_index = _load_checkpoint(
            args.resume_from.expanduser().resolve(),
            compressor=compressor,
            optimizer=optimizer,
            rank=rank,
            run_contract_sha256=checkpoint_run_contract_sha256,
            checkpoint_schema=checkpoint_schema_to_load,
            wm4_resume_binding=wm4_resume_binding,
            current_run_contract=(
                run_contract
                if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
                else None
            ),
            wm4_resume_source_checkpoint_identity=(
                wm4_resume_source_checkpoint_identity
                if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
                else None
            ),
            wm4_low_peak_moment_residency=(
                wm4_low_peak_moment_residency
                if wm4_low_peak_moment_residency.enabled
                else None
            ),
        )
        if initial_progressive_transition and (
            global_step,
            start_epoch,
            start_line_index,
        ) != (
            QWEN38_STEP1000_SOURCE_CONTRACT["global_step"],
            QWEN38_STEP1000_SOURCE_CONTRACT["next_epoch"],
            QWEN38_STEP1000_SOURCE_CONTRACT["next_line_index"],
        ):
            raise RuntimeError(
                "loaded progressive transition cursor changed unexpectedly"
            )
        start_epoch, start_line_index = _resolve_resume_cursor(
            start_epoch,
            start_line_index,
            resume_at_recursive_epoch=args.resume_at_recursive_epoch,
        )
        if initial_progressive_transition and (start_epoch, start_line_index) != (1, 0):
            raise RuntimeError("progressive transition did not start at epoch 2 line 0")
    accumulated_examples = 0
    _zero_optimizer_grad(optimizer, memory_mode=args.optimizer_memory_mode)
    start_time = time.monotonic()
    stop = False
    counters = {
        "rows": 0,
        "eligible": 0,
        "invalid": 0,
        "too_long": 0,
        "rows_without_required_tool_batches": 0,
        "rows_partially_sequence_capped": 0,
        "events_over_sequence_cap": 0,
    }
    if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
        counters.update(
            {
                "selected_actions": 0,
                "state_replay_updates": 0,
                "supervised_compression_events": 0,
                "raw_teacher_forwards": 0,
                "complete_action_over_cap_failures": 0,
                "exact_manifest_epoch_suffixes_completed": 0,
            }
        )
    elif args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
        counters.update(
            {
                "selected_actions": 0,
                "state_replay_updates": 0,
                "supervised_compression_events": 0,
                "raw_teacher_forwards": 0,
                "trajectories_without_eligible_action": 0,
                "student_over_12288_skipped": 0,
                "teacher_over_12288_skipped": 0,
                "complete_rounds_truncated": 0,
                "exact_manifest_epoch_suffixes_completed": 0,
            }
        )
    pending_metrics: list[tuple[Any, dict[str, Any]]] = []
    parity_records: list[dict[str, Any]] = []
    fixed_probe_endpoint_executed = False
    parity_endpoint_verified = False
    last_update_metrics: dict[str, Any] | None = None
    wm4_update_trace: list[dict[str, Any]] = []
    next_epoch = start_epoch
    next_line_index = start_line_index

    def apply_pending_update(*, epoch: int, stage_one: bool) -> bool:
        nonlocal accumulated_examples, global_step, last_update_metrics
        if accumulated_examples <= 0 or not pending_metrics:
            raise RuntimeError("optimizer update has no accumulated examples")
        examples_in_update = accumulated_examples
        _normalize_accumulated_gradients(
            compressor.parameters(),
            examples_in_update,
        )
        grad_norm = compressor.clip_grad_norm_(args.max_grad_norm)
        if (
            recursive_event_caps
            or args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
        ) and not math.isfinite(float(grad_norm)):
            raise RuntimeError("recursive training gradient norm is non-finite")
        current_lr = _learning_rate(
            global_step,
            base=args.learning_rate,
            warmup=args.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        _step_optimizer(
            optimizer,
            memory_mode=args.optimizer_memory_mode,
            wm4_low_peak_moment_residency=(
                wm4_low_peak_moment_residency
                if wm4_low_peak_moment_residency.enabled
                else None
            ),
        )
        accumulated_examples = 0
        global_step += 1
        # This receipt is deliberately taken after the optimizer boundary, not
        # at result serialization.  A long GA8 formal run can therefore prove
        # immediately after every update that its persistent AdamW moments no
        # longer overlap the following forward graph.  ``record=False`` keeps
        # the counter an exact count of actual forward entries.
        wm4_moment_update_receipt: dict[str, Any] | None = None
        if wm4_low_peak_moment_residency.enabled:
            wm4_low_peak_moment_residency.assert_cpu_resident_at_forward_entry(
                optimizer, record=False
            )
            wm4_moment_update_receipt = (
                wm4_low_peak_moment_residency.contract_receipt()
            )
            if wm4_moment_update_receipt is None:
                raise RuntimeError("WM4 low-peak moment receipt was not produced")

        if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
            metric_denominator = sum(
                float(lengths["metric_weight"]) for _, lengths in pending_metrics
            )
            if not math.isclose(
                metric_denominator,
                float(examples_in_update),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "MindForge pending metric weights do not equal trajectory count"
                )

            def weighted_optional(name: str) -> float | None:
                selected = [
                    (getattr(item, name), float(lengths["metric_weight"]))
                    for item, lengths in pending_metrics
                    if getattr(item, name) is not None
                ]
                if not selected:
                    return None
                weight_sum = sum(weight for _, weight in selected)
                return (
                    sum(
                        float(value.detach().item()) * weight
                        for value, weight in selected
                    )
                    / weight_sum
                )

            mean_objective = (
                sum(
                    float(item.loss.detach().item()) * float(lengths["metric_weight"])
                    for item, lengths in pending_metrics
                )
                / metric_denominator
            )
            mean_ce = weighted_optional("ce_loss")
            mean_kl = weighted_optional("kl_loss")
            mean_action_accuracy = weighted_optional("token_accuracy")
            mean_agreement = weighted_optional("top1_agreement")
        else:
            token_total = sum(
                int(item.token_count.item()) for item, _ in pending_metrics
            )
            mean_kl = (
                sum(
                    float(item.loss.detach().item()) * int(item.token_count.item())
                    for item, _ in pending_metrics
                )
                / token_total
            )
            mean_objective = mean_kl
            mean_ce = None
            mean_action_accuracy = None
            mean_agreement = (
                sum(
                    float(item.top1_agreement.item()) * int(item.token_count.item())
                    for item, _ in pending_metrics
                )
                / token_total
            )
        max_source = max(
            lengths["physical_source_tokens"] for _, lengths in pending_metrics
        )
        max_student = max(
            lengths["student_sequence_tokens"] for _, lengths in pending_metrics
        )
        max_teacher = max(
            lengths["teacher_sequence_tokens"] for _, lengths in pending_metrics
        )
        max_teacher_chunks = max(
            lengths["teacher_cached_chunks"] for _, lengths in pending_metrics
        )
        teacher_forwards_in_update = sum(
            int(lengths.get("teacher_forward_executed", True))
            for _, lengths in pending_metrics
        )
        teacher_token_mass_in_update = sum(
            lengths["teacher_sequence_tokens"]
            for _, lengths in pending_metrics
            if lengths.get("teacher_forward_executed", True)
        )
        events_in_update = len(pending_metrics)
        vibe_descriptor_receipts = (
            [
                {
                    name: lengths[name]
                    for name in (
                        "descriptor_id",
                        "descriptor_sha256",
                        "action_descriptor_sha256",
                        "trajectory_id",
                        "target_message_index",
                        "teacher_prefix_sha256",
                        "student_prefix_sha256",
                        "final_serving_state_only",
                        "working_memory_rounds",
                    )
                }
                for _, lengths in pending_metrics
            ]
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            else []
        )
        pending_metrics.clear()
        last_update_metrics = {
            "training_objective": mean_objective,
            "training_kl": mean_kl,
            "training_ce": mean_ce,
            "action_token_accuracy": mean_action_accuracy,
            "top1_agreement": mean_agreement,
            "gradient_norm": float(grad_norm),
            "learning_rate": current_lr,
            "examples_in_update": float(examples_in_update),
        }
        if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
            last_update_metrics["descriptor_receipts"] = vibe_descriptor_receipts
            if wm4_moment_update_receipt is not None:
                last_update_metrics["wm4_low_peak_adam_moment_residency"] = (
                    wm4_moment_update_receipt
                )
            if len(vibe_descriptor_receipts) == 1:
                last_update_metrics.update(vibe_descriptor_receipts[0])
        if args.data_contract != MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
            last_update_metrics.pop("training_objective")
            last_update_metrics.pop("training_ce")
            last_update_metrics.pop("action_token_accuracy")
        if (
            recursive_event_caps
            or args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
        ) and any(
            not math.isfinite(float(value))
            for name, value in last_update_metrics.items()
            if name
            in {
                "training_objective",
                "training_kl",
                "training_ce",
                "action_token_accuracy",
                "top1_agreement",
                "gradient_norm",
                "learning_rate",
                "examples_in_update",
            }
            if value is not None
        ):
            raise RuntimeError("recursive update metrics are non-finite")
        if writer is not None:
            if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
                writer.add_scalar("train/objective", mean_objective, global_step)
            if mean_kl is not None:
                writer.add_scalar("train/forward_kl", mean_kl, global_step)
            if mean_ce is not None:
                writer.add_scalar("train/action_ce", mean_ce, global_step)
            if mean_action_accuracy is not None:
                writer.add_scalar(
                    "train/action_token_accuracy",
                    mean_action_accuracy,
                    global_step,
                )
            if mean_agreement is not None:
                writer.add_scalar("train/top1_agreement", mean_agreement, global_step)
            writer.add_scalar("train/learning_rate", current_lr, global_step)
            writer.add_scalar("train/gradient_norm", float(grad_norm), global_step)
            writer.add_scalar(
                "train/curriculum_stage", 1 if stage_one else 2, global_step
            )
            writer.add_scalar(
                "train/examples_in_update", examples_in_update, global_step
            )
            writer.add_scalar("train/events_in_update", events_in_update, global_step)
            writer.add_scalar(
                "train/teacher_forwards_in_update",
                teacher_forwards_in_update,
                global_step,
            )
            writer.add_scalar(
                "train/teacher_token_mass_in_update",
                teacher_token_mass_in_update,
                global_step,
            )
            writer.add_scalar(
                "train/max_physical_source_tokens", max_source, global_step
            )
            writer.add_scalar(
                "train/max_student_sequence_tokens", max_student, global_step
            )
            writer.add_scalar(
                "train/max_teacher_sequence_tokens", max_teacher, global_step
            )
            writer.add_scalar(
                "train/max_teacher_cached_chunks", max_teacher_chunks, global_step
            )
            if wm4_moment_update_receipt is not None:
                writer.add_scalar(
                    "train/wm4_adam_moment_cpu_bytes",
                    wm4_moment_update_receipt["cpu_moment_bytes"],
                    global_step,
                )
                writer.add_scalar(
                    "train/wm4_adam_moment_cpu_tensors",
                    wm4_moment_update_receipt["cpu_moment_tensors"],
                    global_step,
                )
                writer.add_scalar(
                    "train/wm4_adam_moment_forward_entry_cuda_bytes",
                    wm4_moment_update_receipt[
                        "max_forward_entry_cuda_moment_bytes"
                    ],
                    global_step,
                )
                writer.add_scalar(
                    "train/wm4_adam_moment_restore_calls",
                    wm4_moment_update_receipt["restore_calls"],
                    global_step,
                )
                writer.add_scalar(
                    "train/wm4_adam_moment_offload_calls",
                    wm4_moment_update_receipt["offload_calls"],
                    global_step,
                )
            writer.add_scalar(
                "train/peak_allocated_gib",
                torch.cuda.max_memory_allocated(device) / 1024**3,
                global_step,
            )
            writer.flush()
        if rank == 0 and global_step % args.log_interval == 0:
            log_record = {
                "step": global_step,
                "epoch": epoch + 1,
                "stage": ("one_shot" if stage_one else "recursive_stopgrad"),
                "objective": args.objective,
                "objective_loss": mean_objective,
                "kl": mean_kl,
                "ce": mean_ce,
                "action_token_accuracy": mean_action_accuracy,
                "top1_agreement": mean_agreement,
                "grad_norm": float(grad_norm),
                "lr": current_lr,
                "examples": examples_in_update,
                "events": events_in_update,
                "teacher_forwards": teacher_forwards_in_update,
                "teacher_token_mass": teacher_token_mass_in_update,
                "max_teacher_tokens": max_teacher,
                "max_teacher_cached_chunks": max_teacher_chunks,
                "peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "elapsed_seconds": time.monotonic() - start_time,
            }
            if wm4_moment_update_receipt is not None:
                log_record["wm4_adam_moment_residency"] = {
                    name: wm4_moment_update_receipt[name]
                    for name in (
                        "cpu_moment_bytes",
                        "cpu_moment_tensors",
                        "max_forward_entry_cuda_moment_bytes",
                        "restore_calls",
                        "offload_calls",
                        "offload_calls_by_reason",
                    )
                }
            if args.data_contract != MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
                for key in (
                    "objective",
                    "objective_loss",
                    "ce",
                    "action_token_accuracy",
                ):
                    log_record.pop(key)
            if recursive_event_caps and not stage_one:
                active_cap = _recursive_event_cap_for_epoch(
                    recursive_event_caps,
                    epoch=epoch,
                )
                log_record["recursive_event_cap"] = (
                    "all" if active_cap is None else active_cap
                )
            print(
                json.dumps(
                    log_record,
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
            wm4_update_trace.append(
                {
                    "global_step": global_step,
                    "next_epoch": next_epoch,
                    "next_line_index": next_line_index,
                    "last_update": last_update_metrics,
                }
            )
        if args.save_interval and global_step % args.save_interval == 0:
            _save_checkpoint(
                output_dir,
                compressor=compressor,
                optimizer=optimizer,
                rank=rank,
                global_step=global_step,
                next_epoch=next_epoch,
                next_line_index=next_line_index,
                run_contract_sha256=run_contract_sha256,
                checkpoint_schema=checkpoint_schema,
                wm4_resume_binding=wm4_resume_binding,
                wm4_low_peak_moment_residency=(
                    wm4_low_peak_moment_residency
                    if wm4_low_peak_moment_residency.enabled
                    else None
                ),
            )
        return args.max_steps > 0 and global_step >= args.max_steps

    for epoch in range(start_epoch, args.epochs):
        if args.curriculum_mode == "auto":
            stage_one = epoch == 0
        else:
            stage_one = args.curriculum_mode == "one_shot"
        line_floor = start_line_index if epoch == start_epoch else 0
        if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT:
            if stage_one:
                raise RuntimeError("Qwen3.8 vibe action-KL cannot enter one-shot")
            if not isinstance(vibe_manifest, dict) or not isinstance(
                vibe_manifest_receipt, dict
            ):
                raise RuntimeError("Qwen3.8 vibe manifest disappeared before training")
            from context_compression.qwen38_coding_trajectory_data import (
                iter_frozen_trajectory_rows,
            )

            next_epoch = epoch

            def commit_vibe_trajectory(
                training_order: int,
                next_training_order: int,
                trajectory_id: str,
                event_metrics: list[tuple[MindForgeObjectiveOutput, dict[str, Any]]],
                row_stats: dict[str, int],
            ) -> bool:
                del training_order, trajectory_id
                nonlocal accumulated_examples, next_line_index
                counters["rows"] += 1
                next_line_index = next_training_order
                counters["student_over_12288_skipped"] += row_stats["student_over_cap"]
                counters["teacher_over_12288_skipped"] += row_stats["teacher_over_cap"]
                counters["trajectories_without_eligible_action"] += row_stats[
                    "without_eligible_action"
                ]
                if not event_metrics:
                    if row_stats["selected_actions"] != 0:
                        raise RuntimeError(
                            "skipped Qwen3.8 vibe row selected an action"
                        )
                    return False
                counters["eligible"] += 1
                counters["selected_actions"] += row_stats["selected_actions"]
                counters["state_replay_updates"] += row_stats["state_replay_updates"]
                counters["supervised_compression_events"] += row_stats[
                    "supervised_compression_events"
                ]
                counters["raw_teacher_forwards"] += row_stats["teacher_forwards"]
                pending_metrics.extend(event_metrics)
                accumulated_examples += 1
                if accumulated_examples < args.grad_accumulation:
                    return False
                if accumulated_examples != args.grad_accumulation:
                    raise RuntimeError(
                        "Qwen3.8 vibe gradient accumulation exceeded its boundary"
                    )
                return apply_pending_update(epoch=epoch, stage_one=False)

            vibe_epoch_result = _drive_vibe_manifest_epoch(
                rows=iter_frozen_trajectory_rows(
                    vibe_manifest,
                    verify_files=False,
                    eligible_only=True,
                ),
                expected_trajectories=int(
                    vibe_manifest_receipt["eligible_trajectories"]
                ),
                line_floor=line_floor,
                epoch=epoch,
                tokenizer=tokenizer,
                target=target,
                compressor=compressor,
                args=args,
                expected_descriptors_by_trajectory=(vibe_descriptors_by_trajectory),
                canary_schedule_selection=vibe_manifest_receipt.get(
                    "canary_schedule_selection"
                ),
                before_backward_row=(
                    lambda: wm4_low_peak_moment_residency.assert_cpu_resident_at_forward_entry(
                        optimizer
                    )
                    if wm4_low_peak_moment_residency.enabled
                    else None
                ),
                commit_trajectory=commit_vibe_trajectory,
            )
            next_line_index = vibe_epoch_result.next_training_order
            if vibe_epoch_result.stopped:
                stop = True
                break
            if not vibe_epoch_result.completed_manifest_suffix:
                raise RuntimeError(
                    "Qwen3.8 vibe epoch driver returned an incomplete suffix"
                )
            counters["exact_manifest_epoch_suffixes_completed"] += 1
            next_epoch = epoch + 1
            next_line_index = 0
            if accumulated_examples:
                stop = apply_pending_update(epoch=epoch, stage_one=False)
                if stop:
                    break
            continue
        if args.data_contract == MINDFORGE_FRONTIER_CE_DATA_CONTRACT:
            if stage_one:
                raise RuntimeError("MindForge manifest training cannot enter one-shot")
            if mindforge_manifest is None:
                raise RuntimeError("MindForge manifest disappeared before training")
            next_epoch = epoch

            def commit_mindforge_trajectory(
                training_order: int,
                next_training_order: int,
                event_metrics: list[tuple[MindForgeObjectiveOutput, dict[str, Any]]],
                row_stats: dict[str, int],
            ) -> bool:
                del training_order
                nonlocal accumulated_examples, next_line_index
                counters["rows"] += 1
                # The checkpoint cursor is the manifest training order, never
                # the physical JSONL source line selected by that row.
                next_line_index = next_training_order
                counters["eligible"] += 1
                counters["selected_actions"] += row_stats["selected_actions"]
                counters["state_replay_updates"] += row_stats["state_replay_updates"]
                counters["supervised_compression_events"] += row_stats[
                    "supervised_compression_events"
                ]
                counters["raw_teacher_forwards"] += row_stats["teacher_forwards"]
                pending_metrics.extend(event_metrics)
                # One trajectory has unit objective mass regardless of A actions
                # and K directly-supervised compression states.
                accumulated_examples += 1
                if accumulated_examples < args.grad_accumulation:
                    return False
                if accumulated_examples != args.grad_accumulation:
                    raise RuntimeError(
                        "MindForge gradient accumulation exceeded its boundary"
                    )
                return apply_pending_update(epoch=epoch, stage_one=False)

            epoch_result = _drive_mindforge_manifest_epoch(
                train_rows=mindforge_train_rows,
                line_floor=line_floor,
                source_path=args.data,
                tokenizer=tokenizer,
                policy=mindforge_manifest["policy"],
                target=target,
                compressor=compressor,
                args=args,
                commit_trajectory=commit_mindforge_trajectory,
            )
            next_line_index = epoch_result.next_training_order
            if epoch_result.stopped:
                stop = True
                break
            if not epoch_result.completed_exact_set:
                raise RuntimeError("MindForge epoch driver returned an incomplete set")
            counters["exact_manifest_epoch_suffixes_completed"] += 1
            next_epoch = epoch + 1
            next_line_index = 0
            if accumulated_examples:
                stop = apply_pending_update(epoch=epoch, stage_one=False)
                if stop:
                    break
            continue
        for line_index, row in _iter_training_rows(args.data, args.data_contract):
            if line_index < line_floor:
                continue
            if parity_descriptor is not None:
                selected_line = int(parity_descriptor["line_index"])
                if line_index < selected_line:
                    continue
                if line_index > selected_line:
                    break
            counters["rows"] += 1
            next_epoch = epoch
            next_line_index = line_index + 1
            try:
                if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
                    example = render_qwen36_agentic_example(
                        row,
                        tokenizer,
                        target_cap=args.kl_tokens,
                    )
                else:
                    example = build_qwen36_token_clock_example(row, tokenizer)
            except (TypeError, ValueError):
                counters["invalid"] += 1
                continue
            if len(example.history_input_ids) > args.max_history_tokens:
                counters["too_long"] += 1
                continue
            if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
                if not isinstance(example, Qwen36AgenticExample):
                    raise RuntimeError(
                        "agentic adapter returned the wrong example type"
                    )
                candidate_events = example.events
            else:
                candidate_events = token_clock_events(
                    len(example.history_input_ids), target_tokens=args.kl_tokens
                )
            if not candidate_events:
                if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
                    counters["rows_without_required_tool_batches"] += 1
                continue
            if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
                events = _agentic_events_within_caps(
                    candidate_events,
                    prefix_tokens=len(example.raw_prefix_input_ids),
                    max_student_sequence_tokens=args.max_target_sequence_tokens,
                    max_teacher_sequence_tokens=args.max_teacher_sequence_tokens,
                    stage_one=stage_one,
                )
            elif stage_one:
                # Before the compressor has a recurrent state, the untouched
                # earlier history remains raw in both source and student paths.
                events = _events_within_sequence_cap(
                    candidate_events,
                    prefix_tokens=len(example.raw_prefix_input_ids),
                    max_sequence_tokens=args.max_target_sequence_tokens,
                )
            else:
                # Recursive student/source contexts stay compact.  Only the
                # no-grad raw teacher grows, and it is streamed through cache.
                events = _recursive_events_within_caps(
                    candidate_events,
                    prefix_tokens=len(example.raw_prefix_input_ids),
                    max_student_sequence_tokens=args.max_target_sequence_tokens,
                    max_teacher_sequence_tokens=args.max_teacher_sequence_tokens,
                )
            capped_event_count = len(candidate_events) - len(events)
            if capped_event_count:
                counters["rows_partially_sequence_capped"] += 1
                counters["events_over_sequence_cap"] += capped_event_count
            if not events:
                counters["too_long"] += 1
                continue
            if stage_one:
                selected_events = (
                    _stage1_event(
                        events,
                        example_id=example.example_id,
                        epoch=epoch,
                        seed=args.seed,
                    ),
                )
            elif recursive_event_caps:
                recursive_event_cap = _recursive_event_cap_for_epoch(
                    recursive_event_caps,
                    epoch=epoch,
                )
                selected_events = _apply_recursive_event_cap(
                    events,
                    recursive_event_cap,
                )
            else:
                selected_events = events
            if parity_descriptor is not None:
                if line_index != parity_descriptor["line_index"]:
                    raise RuntimeError(
                        "parity selector reached an unexpected source record"
                    )
                endpoint = int(parity_descriptor["endpoint_event_index"])
                selected_events = tuple(events[: endpoint + 1])
                if [
                    event.event_index for event in selected_events
                ] != parity_descriptor["selected_event_indices"]:
                    raise RuntimeError("parity event causal-prefix contract changed")
            counters["eligible"] += 1
            previous_memory = None
            event_metrics = []
            agentic_teacher_state = None
            if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
                if not isinstance(example, Qwen36AgenticExample):
                    raise RuntimeError("agentic training example type changed")
                agentic_teacher_state = _agentic_teacher_for_events(
                    target=target,
                    example=example,
                    events=selected_events,
                    lm_head_chunk_size=args.lm_head_chunk_size,
                    teacher_prefill_chunk_tokens=(args.teacher_prefill_chunk_tokens),
                    teacher_execution_mode=args.teacher_execution_mode,
                    verify_cached_teacher=args.verify_cached_teacher,
                    cached_teacher_max_output_kl=(args.cached_teacher_max_output_kl),
                    cached_teacher_min_top1_agreement=(
                        args.cached_teacher_min_top1_agreement
                    ),
                )
            for event_offset, event in enumerate(selected_events):
                if args.data_contract == QWEN36_AGENTIC_DATA_CONTRACT:
                    if agentic_teacher_state is None or not isinstance(
                        event, AgenticCompressionEvent
                    ):
                        raise RuntimeError("agentic event/teacher state is missing")
                    (
                        shared_target_ids,
                        shared_teacher_hidden,
                        shared_teacher_cached_chunks,
                        shared_cached_teacher_parity,
                        shared_teacher_sequence_tokens,
                    ) = agentic_teacher_state
                    kl, memory, lengths = _agentic_one_event(
                        target=target,
                        compressor=compressor,
                        example=example,
                        event=event,
                        previous_memory=previous_memory,
                        stage_one=stage_one,
                        target_ids=shared_target_ids,
                        teacher_hidden=shared_teacher_hidden,
                        teacher_cached_chunks=shared_teacher_cached_chunks,
                        cached_teacher_parity=shared_cached_teacher_parity,
                        teacher_sequence_tokens=shared_teacher_sequence_tokens,
                        teacher_forward_executed=event_offset == 0,
                        lm_head_chunk_size=args.lm_head_chunk_size,
                    )
                else:
                    if not isinstance(event, TokenClockEvent):
                        raise RuntimeError("legacy token-clock event type changed")
                    kl, memory, lengths = _one_event(
                        target=target,
                        compressor=compressor,
                        example=example,
                        event=event,
                        previous_memory=previous_memory,
                        stage_one=stage_one,
                        lm_head_chunk_size=args.lm_head_chunk_size,
                        teacher_prefill_chunk_tokens=(
                            args.teacher_prefill_chunk_tokens
                        ),
                        teacher_execution_mode=args.teacher_execution_mode,
                        verify_cached_teacher=args.verify_cached_teacher,
                        cached_teacher_max_output_kl=(
                            args.cached_teacher_max_output_kl
                        ),
                        cached_teacher_min_top1_agreement=(
                            args.cached_teacher_min_top1_agreement
                        ),
                    )
                # Sum exact per-example means.  The optimizer boundary divides
                # by the actual number of rows, including a short epoch tail.
                scaled_loss = kl.loss / float(len(selected_events))
                scaled_loss.backward()
                previous_memory = memory.detach()
                event_metrics.append((kl, lengths))
                parity = lengths["cached_teacher_parity"]
                if (
                    parity_descriptor is not None
                    and event.event_index == parity_descriptor["endpoint_event_index"]
                ):
                    fixed_probe_endpoint_executed = True
                if parity is not None:
                    parity_record = {
                        "line_index": line_index,
                        "example_id": example.example_id,
                        "event_index": event.event_index,
                        "teacher_sequence_tokens": lengths["teacher_sequence_tokens"],
                        "student_sequence_tokens": lengths["student_sequence_tokens"],
                        "teacher_cached_chunks": lengths["teacher_cached_chunks"],
                        **parity,
                    }
                    parity_records.append(parity_record)
                    if (
                        parity_descriptor is not None
                        and event.event_index
                        == parity_descriptor["endpoint_event_index"]
                    ):
                        parity_endpoint_verified = True
            if agentic_teacher_state is not None:
                del agentic_teacher_state
            if parity_descriptor is not None and not fixed_probe_endpoint_executed:
                raise RuntimeError("selected fixed-probe endpoint did not execute")
            if args.verify_cached_teacher and not parity_endpoint_verified:
                raise RuntimeError(
                    "selected parity endpoint did not execute cached verification"
                )
            pending_metrics.extend(event_metrics)
            accumulated_examples += 1
            if accumulated_examples < args.grad_accumulation:
                continue
            if accumulated_examples != args.grad_accumulation:
                raise RuntimeError("gradient accumulation exceeded its boundary")
            stop = apply_pending_update(epoch=epoch, stage_one=stage_one)
            if stop:
                break
        if stop:
            break
        next_epoch = epoch + 1
        next_line_index = 0
        if accumulated_examples:
            # Flush before the curriculum changes.  This also preserves the
            # final short batch instead of silently discarding its gradients.
            stop = apply_pending_update(epoch=epoch, stage_one=stage_one)
            if stop:
                break

    parameters_finite_after_step: bool | None = None
    if args.verify_parameters_finite_after_step:
        local_finite = torch.ones((), device=device, dtype=torch.int32)
        for parameter in compressor.parameters():
            if not bool(torch.isfinite(parameter.detach()).all().item()):
                local_finite.zero_()
                break
        dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
        parameters_finite_after_step = bool(local_finite.item())
        if not parameters_finite_after_step:
            raise RuntimeError(
                "compressor parameters are non-finite after the controlled update"
            )

    final_checkpoint = None
    if args.verify_cached_teacher:
        endpoint_records = [
            record
            for record in parity_records
            if record["line_index"] == parity_descriptor["line_index"]
            and record["event_index"] == parity_descriptor["endpoint_event_index"]
        ]
        if len(endpoint_records) != 1:
            raise RuntimeError(
                "parity endpoint must produce exactly one verification record"
            )
    rank_parity_records: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rank_parity_records, parity_records)
    if args.verify_cached_teacher and rank_parity_records[0] != rank_parity_records[1]:
        raise RuntimeError("cached-teacher parity metrics differ across TP ranks")
    if not args.skip_final_checkpoint:
        expected = output_dir / f"checkpoint-{global_step:08d}"
        if expected.exists():
            final_checkpoint = expected
        else:
            final_checkpoint = _save_checkpoint(
                output_dir,
                compressor=compressor,
                optimizer=optimizer,
                rank=rank,
                global_step=global_step,
                next_epoch=next_epoch,
                next_line_index=next_line_index,
                run_contract_sha256=run_contract_sha256,
                checkpoint_schema=checkpoint_schema,
                wm4_resume_binding=wm4_resume_binding,
                wm4_low_peak_moment_residency=(
                    wm4_low_peak_moment_residency
                    if wm4_low_peak_moment_residency.enabled
                    else None
                ),
            )

    if writer is not None:
        writer.close()
    if args.verify_cached_teacher:
        if not parity_records:
            raise RuntimeError(
                "verify-cached-teacher completed without exercising cache"
            )
    if parity_descriptor is not None:
        if global_step != 1 or last_update_metrics is None:
            raise RuntimeError("fixed-probe smoke did not complete one update")
        if not all(math.isfinite(value) for value in last_update_metrics.values()):
            raise RuntimeError("fixed-probe smoke update is non-finite")
        if parameters_finite_after_step is None:
            local_finite = torch.ones((), device=device, dtype=torch.int32)
            for parameter in compressor.parameters():
                if not bool(torch.isfinite(parameter.detach()).all().item()):
                    local_finite.zero_()
                    break
            dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
            parameters_finite_after_step = bool(local_finite.item())
        if not parameters_finite_after_step:
            raise RuntimeError("compressor parameters are non-finite after fixed probe")
    wm4_low_peak_moment_rank_receipts: list[dict[str, Any]] | None = None
    if wm4_low_peak_moment_residency.enabled:
        wm4_low_peak_moment_residency.assert_cpu_resident_at_forward_entry(
            optimizer, record=False
        )
        local_moment_receipt = wm4_low_peak_moment_residency.contract_receipt()
        if local_moment_receipt is None:
            raise RuntimeError("WM4 low-peak moment residency receipt disappeared")
        if global_step > 0 and (
            local_moment_receipt["cpu_moment_tensors"] <= 0
            or local_moment_receipt["cpu_moment_bytes"] <= 0
        ):
            raise RuntimeError("WM4 AdamW moments were not CPU-resident after update")
        local_moment_receipt = {
            **local_moment_receipt,
            "rank": rank,
            "world_size": dist.get_world_size(),
        }
        rank_receipts: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rank_receipts, local_moment_receipt)
        if not all(isinstance(item, dict) for item in rank_receipts):
            raise RuntimeError("WM4 low-peak moment residency rank receipts are invalid")
        if any(
            item.get("rank") != index
            or item.get("world_size") != dist.get_world_size()
            or item.get("policy") != WM4_LOW_PEAK_ADAM_MOMENT_RESIDENCY
            or item.get("max_forward_entry_cuda_moment_bytes") != 0
            or item.get("forward_entry_checks", 0) <= 0
            for index, item in enumerate(rank_receipts)
        ):
            raise RuntimeError(
                "WM4 AdamW moment residency was not CPU-only at forward entry"
            )
        common_fields = (
            "policy",
            "moment_keys",
            "residency_between_optimizer_uses",
            "restore_after_activation_release_before_optimizer_use",
            "offload_calls",
            "restore_calls",
            "forward_entry_checks",
            "max_forward_entry_cuda_moment_bytes",
            "offload_calls_by_reason",
        )
        if any(
            {name: item.get(name) for name in common_fields}
            != {name: rank_receipts[0].get(name) for name in common_fields}
            for item in rank_receipts[1:]
        ):
            raise RuntimeError(
                "WM4 AdamW moment residency control flow differs between ranks"
            )
        wm4_low_peak_moment_rank_receipts = [
            dict(item) for item in rank_receipts
        ]
    result = {
        "status": "completed",
        "global_step": global_step,
        "run_contract_sha256": run_contract_sha256,
        "epochs_requested": args.epochs,
        "recursive_epochs_requested": (
            len(recursive_event_caps) if recursive_event_caps else None
        ),
        "progressive_recursive_event_caps": (
            ["all" if cap is None else cap for cap in recursive_event_caps]
            if recursive_event_caps
            else None
        ),
        "next_epoch": next_epoch,
        "next_line_index": next_line_index,
        "counters": counters,
        "peak_allocated_gib_per_rank": torch.cuda.max_memory_allocated(device)
        / 1024**3,
        "final_checkpoint_saved": final_checkpoint is not None,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "last_update": last_update_metrics,
        "parameters_finite_after_step": parameters_finite_after_step,
        "fixed_probe_endpoint_executed": fixed_probe_endpoint_executed,
        "cached_teacher_parity_artifact": (
            str(output_dir / "cached_teacher_parity.json")
            if args.verify_cached_teacher
            else None
        ),
        **(
            {
                "data_manifest_file_sha256": data_sha256,
                "data_manifest_canonical_sha256": vibe_manifest_receipt[
                    "manifest_canonical_sha256"
                ],
                "data_manifest_cap_coverage": vibe_manifest_receipt["cap_coverage"],
                "final_serving_state_only": True,
                "working_memory_rounds": args.vibe_working_memory_rounds,
                "recurrence_gradient": "stopgrad",
                "complete_rounds_truncated": 0,
                "checkpoint_cursor_semantics": (
                    "next_frozen_eligible_manifest_training_order"
                ),
                "wm4_update_trace": wm4_update_trace,
                "wm4_low_peak_adam_moment_residency": (
                    {
                        "policy": WM4_LOW_PEAK_ADAM_MOMENT_RESIDENCY,
                        "forward_entry_cache_trim": WM4_FORWARD_ENTRY_CACHE_TRIM,
                        "rank_receipts": wm4_low_peak_moment_rank_receipts,
                    }
                ),
                **(
                    {
                        "wm4_resume_source_checkpoint_identity": (
                            wm4_resume_source_checkpoint_identity
                        )
                    }
                    if wm4_resume_source_checkpoint_identity is not None
                    else {}
                ),
            }
            if args.data_contract == QWEN38_VIBE_WM4_ACTION_KL_DATA_CONTRACT
            else {}
        ),
    }
    if rank == 0:
        _atomic_json(output_dir / "result.json", result)
        if args.verify_cached_teacher:
            result_sha256 = _sha256(output_dir / "result.json")
            parity_artifact = {
                "format_version": PARITY_FORMAT_VERSION,
                "schema": PARITY_SCHEMA,
                "status": "passed",
                "direction": "KL(monolithic_teacher||cached_teacher)",
                "maximum_output_distribution_kl": (args.cached_teacher_max_output_kl),
                "minimum_top1_agreement": (args.cached_teacher_min_top1_agreement),
                "hidden_difference_is_diagnostic_only": True,
                "teacher_prefill_chunk_tokens": (args.teacher_prefill_chunk_tokens),
                "kl_tokens": args.kl_tokens,
                "lm_head_chunk_size": args.lm_head_chunk_size,
                "target_family": args.target_family,
                "target_model": str(args.target_model.resolve()),
                "target_text_snapshot_manifest_sha256": target_manifest_sha256,
                "target_content_identity": target_content_identity,
                "data": str(args.data.resolve()),
                "data_sha256": data_sha256,
                "selector": parity_descriptor,
                "rank_records": [
                    {"rank": index, "events": records}
                    for index, records in enumerate(rank_parity_records)
                ],
                "implementation_identity": implementation_identity,
                "run_contract_sha256": run_contract_sha256,
                "result_sha256": result_sha256,
            }
            _atomic_json(output_dir / "cached_teacher_parity.json", parity_artifact)
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
