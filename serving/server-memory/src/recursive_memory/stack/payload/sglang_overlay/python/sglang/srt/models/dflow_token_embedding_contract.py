"""Fail-closed objective ABI checks for token-embedding ELF checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional


LEGACY_L2_OBJECTIVE = "elf_velocity_to_target_token_embedding"
LEGACY_DOT_OBJECTIVE = "elf_velocity_mse_plus_frozen_tied_lm_head_ce"
ONPOLICY_DOT_OBJECTIVE = "elf_onpolicy_rollout_endpoint_mse_plus_frozen_tied_lm_head_ce"
FIXED_MASK_DOT_OBJECTIVE = "elf_endpoint_mse_plus_frozen_tied_lm_head_ce"

OBJECTIVES_BY_RETRIEVAL_METRIC = {
    "l2": frozenset({LEGACY_L2_OBJECTIVE}),
    "dot": frozenset(
        {
            LEGACY_DOT_OBJECTIVE,
            ONPOLICY_DOT_OBJECTIVE,
            FIXED_MASK_DOT_OBJECTIVE,
        }
    ),
}


def _positive_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def validate_token_embedding_objective_abi(
    dflow_config: Mapping,
    *,
    runtime_flow_steps: Optional[int] = None,
) -> tuple[str, str]:
    """Validate the retrieval/objective pairing and objective-specific ABI.

    Historical L2 and dot-product objectives retain their existing serving ABI.
    The on-policy endpoint objective is intentionally stricter: its checkpoint
    must identify student Euler states, use the same discrete grid resolution as
    serving, and disable self-conditioning.  ``runtime_flow_steps`` is supplied
    by the worker/launcher and omitted by the model constructor, where equality
    between the two checkpoint fields is still enforced.
    """

    if not isinstance(dflow_config, Mapping):
        raise ValueError(
            "Token-embedding ELF objective ABI requires dflow_config to be a mapping"
        )

    retrieval_metric = str(dflow_config.get("retrieval_metric", "")).lower()
    allowed_objectives = OBJECTIVES_BY_RETRIEVAL_METRIC.get(retrieval_metric)
    if allowed_objectives is None:
        raise ValueError(
            "Token-embedding ELF objective ABI retrieval_metric must be one of "
            f"{sorted(OBJECTIVES_BY_RETRIEVAL_METRIC)}, got "
            f"{dflow_config.get('retrieval_metric')!r}"
        )

    training_objective = dflow_config.get("training_objective")
    if training_objective not in allowed_objectives:
        raise ValueError(
            "Token-embedding ELF objective ABI training_objective is incompatible "
            f"with retrieval_metric={retrieval_metric!r}: expected one of "
            f"{sorted(allowed_objectives)!r}, got {training_objective!r}"
        )

    if training_objective != ONPOLICY_DOT_OBJECTIVE:
        return retrieval_metric, str(training_objective)

    violations: list[str] = []
    training_state_source = dflow_config.get("training_state_source")
    if training_state_source != "student_euler_rollout":
        violations.append(
            "training_state_source must be 'student_euler_rollout', got "
            f"{training_state_source!r}"
        )

    training_steps = dflow_config.get("training_time_grid_steps")
    serving_steps = dflow_config.get("serving_num_flow_steps")
    if not _positive_int(training_steps):
        violations.append(
            "training_time_grid_steps must be a positive JSON integer, got "
            f"{training_steps!r}"
        )
    if not _positive_int(serving_steps):
        violations.append(
            "serving_num_flow_steps must be a positive JSON integer, got "
            f"{serving_steps!r}"
        )
    if _positive_int(training_steps) and _positive_int(serving_steps):
        if training_steps != serving_steps:
            violations.append(
                "training_time_grid_steps must equal serving_num_flow_steps, got "
                f"{training_steps} and {serving_steps}"
            )

    if dflow_config.get("serving_self_conditioning") is not False:
        violations.append(
            "serving_self_conditioning must be exactly false for the on-policy "
            f"objective, got {dflow_config.get('serving_self_conditioning')!r}"
        )

    exact_solver_fields = {
        "training_time_sampling": "uniform_onpolicy_rollout_step",
        "training_time_sampling_unit": "synchronized_microbatch",
        "training_time_grid_kind": "uniform_linear",
        "training_time_grid_excludes_endpoint": True,
        "training_rollout_start_state": "anchored_gaussian",
        "training_rollout_prefix_model": "online_current_drafter",
        "training_rollout_prefix_grad": False,
        "training_rollout_update_rule": "endpoint_euler",
        "training_rollout_final_step_direct_endpoint": True,
    }
    for field, expected in exact_solver_fields.items():
        actual = dflow_config.get(field)
        if actual != expected or type(actual) is not type(expected):
            violations.append(f"{field} must be exactly {expected!r}, got {actual!r}")

    if _positive_int(training_steps):
        expected_grid = [index / training_steps for index in range(training_steps)]
        actual_grid = dflow_config.get("training_time_grid_values")
        if actual_grid != expected_grid:
            violations.append(
                "training_time_grid_values must be the exact uniform serving "
                f"grid {expected_grid!r}, got {actual_grid!r}"
            )
        serving_forwards = dflow_config.get("serving_drafter_forwards")
        if not _positive_int(serving_forwards) or serving_forwards != training_steps:
            violations.append(
                "serving_drafter_forwards must equal training_time_grid_steps, "
                f"got {serving_forwards!r} and {training_steps}"
            )

    training_noise = dflow_config.get("training_input_noise_scale")
    serving_noise = dflow_config.get("serving_denoiser_noise_scale")
    if training_noise != 2.0 or serving_noise != 2.0:
        violations.append(
            "training_input_noise_scale and serving_denoiser_noise_scale must "
            f"both equal 2.0, got {training_noise!r} and {serving_noise!r}"
        )
    training_t_eps = dflow_config.get("training_t_eps")
    serving_t_eps = dflow_config.get("serving_t_eps")
    if training_t_eps != 0.05 or serving_t_eps != 0.05:
        violations.append(
            "training_t_eps and serving_t_eps must both equal 0.05, got "
            f"{training_t_eps!r} and {serving_t_eps!r}"
        )

    if runtime_flow_steps is not None:
        if not _positive_int(runtime_flow_steps):
            violations.append(
                "runtime flow steps must be a positive integer, got "
                f"{runtime_flow_steps!r}"
            )
        else:
            if _positive_int(training_steps) and training_steps != runtime_flow_steps:
                violations.append(
                    "training_time_grid_steps must equal runtime NFE, got "
                    f"checkpoint={training_steps} runtime={runtime_flow_steps}"
                )
            if _positive_int(serving_steps) and serving_steps != runtime_flow_steps:
                violations.append(
                    "serving_num_flow_steps must equal runtime NFE, got "
                    f"checkpoint={serving_steps} runtime={runtime_flow_steps}"
                )

    if violations:
        raise ValueError(
            "Token-embedding ELF on-policy objective ABI mismatch: "
            + "; ".join(violations)
        )
    return retrieval_metric, str(training_objective)


__all__ = [
    "LEGACY_DOT_OBJECTIVE",
    "LEGACY_L2_OBJECTIVE",
    "FIXED_MASK_DOT_OBJECTIVE",
    "OBJECTIVES_BY_RETRIEVAL_METRIC",
    "ONPOLICY_DOT_OBJECTIVE",
    "validate_token_embedding_objective_abi",
]
