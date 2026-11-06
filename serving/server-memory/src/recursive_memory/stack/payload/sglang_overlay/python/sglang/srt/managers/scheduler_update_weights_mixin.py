from __future__ import annotations

import logging
import os
import traceback
import uuid
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.memory_handle_registry import (
    GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV,
    get_global_memory_handle_registry,
    global_memory_handle_registry_is_configured,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def _memory_handle_weight_update_preflight(
        self: Scheduler, *, flush_cache: bool
    ) -> Tuple[bool, str]:
        """Make the hot-update decision identical on every TP rank."""

        active = global_memory_handle_registry_is_configured()
        local = {"active": active, "registry": None}
        if active:
            stats = get_global_memory_handle_registry().stats()
            local["registry"] = {
                "server_epoch": stats["server_epoch"],
                "hook_abi_digest": stats["hook_abi_digest"],
            }
        outcomes = [local]
        if int(getattr(self, "tp_size", 1)) > 1:
            outcomes = [None] * int(self.tp_size)
            try:
                torch.distributed.all_gather_object(
                    outcomes, local, group=self.tp_cpu_group
                )
            except Exception as exc:
                return False, f"memory-handle hot-update preflight failed: {exc}"
        if any(outcome != outcomes[0] for outcome in outcomes[1:]):
            return False, (
                "memory-handle registry state diverged across TP ranks; "
                "hot weight update rejected"
            )
        if outcomes[0]["active"] and not flush_cache:
            return False, (
                "hot weight update with flush_cache=false is forbidden while "
                "GPU memory handles are active"
            )
        return True, ""

    def _rotate_memory_handle_epoch_after_weight_update(
        self: Scheduler,
    ) -> Tuple[bool, str]:
        if not global_memory_handle_registry_is_configured():
            return True, ""
        registry = get_global_memory_handle_registry()
        epoch_holder = [uuid.uuid4().hex if int(self.tp_rank) == 0 else None]
        try:
            if int(self.tp_size) > 1:
                torch.distributed.broadcast_object_list(
                    epoch_holder,
                    src=self.tp_group.ranks[0],
                    group=self.tp_cpu_group,
                )
            new_epoch = epoch_holder[0]
            if not isinstance(new_epoch, str) or not new_epoch:
                raise RuntimeError("TP epoch broadcast returned an invalid value")
            registry.rotate_epoch(new_epoch)
            os.environ[GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV] = new_epoch
            return True, ""
        except Exception as exc:
            # Weights may already have changed.  Clear every locally reachable
            # handle under a private emergency epoch so stale memory can never
            # be consumed, even if TP epoch synchronization failed.
            emergency_epoch = uuid.uuid4().hex
            try:
                if emergency_epoch == registry.server_epoch:
                    emergency_epoch = uuid.uuid4().hex
                registry.rotate_epoch(emergency_epoch)
                os.environ[GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV] = emergency_epoch
            except Exception as invalidate_exc:
                return False, (
                    "memory-handle epoch rotation failed after weight update: "
                    f"{exc}; emergency invalidation also failed: {invalidate_exc}"
                )
            return False, (
                "memory-handle TP epoch synchronization failed after weight update; "
                f"local handles were invalidated: {exc}"
            )

    def _finish_memory_handle_weight_update(
        self: Scheduler, *, success: bool, message: str
    ) -> Tuple[bool, str]:
        if not success:
            return success, message
        epoch_success, epoch_message = (
            self._rotate_memory_handle_epoch_after_weight_update()
        )
        if not epoch_success:
            return False, f"{message} {epoch_message}".strip()
        return success, message

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        allowed, rejection = self._memory_handle_weight_update_preflight(
            flush_cache=recv_req.flush_cache
        )
        if not allowed:
            return UpdateWeightFromDiskReqOutput(False, rejection, 0)
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
            success, message = self._finish_memory_handle_weight_update(
                success=success, message=message
            )
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        allowed, rejection = self._memory_handle_weight_update_preflight(
            flush_cache=recv_req.flush_cache
        )
        if not allowed:
            return UpdateWeightsFromDistributedReqOutput(False, rejection)
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
            success, message = self._finish_memory_handle_weight_update(
                success=success, message=message
            )
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        allowed, rejection = self._memory_handle_weight_update_preflight(
            flush_cache=recv_req.flush_cache
        )
        if not allowed:
            return UpdateWeightsFromTensorReqOutput(False, rejection)
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
            success, message = self._finish_memory_handle_weight_update(
                success=success, message=message
            )
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        allowed, rejection = self._memory_handle_weight_update_preflight(
            flush_cache=recv_req.flush_cache
        )
        if not allowed:
            return UpdateWeightsFromIPCReqOutput(False, rejection)
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
            success, message = self._finish_memory_handle_weight_update(
                success=success, message=message
            )
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
