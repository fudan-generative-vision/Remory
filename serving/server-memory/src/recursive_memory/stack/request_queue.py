"""Single-event-loop admission with FIFO groups and round-robin fairness."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Backend:
    name: str
    url: str
    model: str
    capacity: int
    native_model: str = ""
    healthy: bool = False
    active: int = 0
    observed_load: int = 0
    starts: int = 0
    observed_starts: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    identity: str | None = None
    error: str | None = None
    load_info: dict[str, Any] = field(default_factory=dict)
    # Remote replicas may share base weights without sharing this registry.
    supports_local_encoder_versions: bool = True
    stateless_only: bool = False
    allow_tensor_recover: bool = False
    tensor_recover_artifact_identity: dict[str, Any] | None = None

    @property
    def can_route_tensor_recover(self):
        return (self.stateless_only and self.allow_tensor_recover
                and self.tensor_recover_artifact_identity is not None
                and self.info.get("remory_artifact_identity") == self.tensor_recover_artifact_identity
                and {"qwen38_summary_residual_sglang_v1", "qwen38_residual_cpu_concurrent_v1",
                     "request_cache_bypass_v1"}.issubset(self.info.get("request_capabilities", [])))

    @property
    def can_route_local_encoder_versions(self):
        return (not self.stateless_only and self.supports_local_encoder_versions
                and "remory_encoder_versions_v1" in self.info.get("request_capabilities", []))

    @property
    def load(self):
        # Include requests sent since the latest external-load probe began.
        # A slightly stale probe may undercount these, never reserve them twice.
        return max(self.active, self.observed_load + self.starts - self.observed_starts)


@dataclass(eq=False)
class Ticket:
    group: str
    model: str
    pinned: str | None
    units: int
    future: asyncio.Future
    backend: Backend | None = None
    released: bool = False
    requires_local_encoder_versions: bool = False
    requires_full_backend: bool = False
    tensor_recover: bool = False


class QueueFull(Exception):
    pass


class AdmissionQueue:
    def __init__(self, backends, *, max_pending=256):
        self.backends = list(backends)
        self.max_pending = max_pending
        self.groups = OrderedDict()
        self.closed = False
        self.admitted = 0

    @property
    def pending(self):
        return sum(len(group) for group in self.groups.values())

    def submit(self, *, group, model, pinned=None, units=1, requires_local_encoder_versions=False,
               requires_full_backend=False, tensor_recover=False):
        if type(units) is not int or units < 1:
            raise ValueError("request units must be a positive integer")
        if self.closed:
            raise RuntimeError("gateway is draining")
        candidates = [b for b in self.backends if b.model == model and
                      (pinned is None or b.name == pinned) and
                      (not requires_full_backend or not b.stateless_only
                       or tensor_recover and b.can_route_tensor_recover) and
                      (not requires_local_encoder_versions or b.can_route_local_encoder_versions)]
        if requires_local_encoder_versions and not candidates:
            raise ValueError("no matching backend supports the local remory_encoder_version registry")
        if requires_full_backend and not candidates:
            raise ValueError("memory/state requests cannot use a stateless_only backend")
        if not candidates or not any(units <= b.capacity for b in candidates):
            raise ValueError("no matching backend has capacity for this request batch")
        if self.pending >= self.max_pending:
            raise QueueFull("request queue is full")
        ticket = Ticket(group, model, pinned, units, asyncio.get_running_loop().create_future(),
                        requires_local_encoder_versions=requires_local_encoder_versions,
                        requires_full_backend=requires_full_backend, tensor_recover=tensor_recover)
        self.groups.setdefault(group, deque()).append(ticket)
        self.pump()
        return ticket

    def pump(self):
        # Visit each group once per round. A blocked/pinned group does not block
        # other groups. FIFO is retained within a group, without preemption.
        while self.groups:
            progress = False
            for name in list(self.groups):
                group = self.groups[name]
                ticket = group[0]
                available = [b for b in self.backends if b.healthy
                             and b.model == ticket.model
                             and (ticket.pinned is None or b.name == ticket.pinned)
                             and (not ticket.requires_full_backend or not b.stateless_only
                                  or ticket.tensor_recover and b.can_route_tensor_recover)
                             and (not ticket.requires_local_encoder_versions
                                  or b.can_route_local_encoder_versions)
                             and b.load + ticket.units <= b.capacity]
                if not available:
                    continue
                backend = min(available, key=lambda b: (b.load / b.capacity, b.starts, b.name))
                group.popleft()
                if group:
                    self.groups.move_to_end(name)
                else:
                    del self.groups[name]
                ticket.backend = backend
                backend.active += ticket.units
                backend.starts += ticket.units
                self.admitted += 1
                ticket.future.set_result(backend)
                progress = True
            if not progress:
                break

    def release(self, ticket):
        if ticket.released:
            return
        ticket.released = True
        if ticket.backend is None:
            group = self.groups.get(ticket.group)
            if group is not None:
                group.remove(ticket)
                if not group:
                    del self.groups[ticket.group]
            ticket.future.cancel()
        else:
            ticket.backend.active -= ticket.units
        self.pump()

    def close(self):
        self.closed = True
        for group in self.groups.values():
            for ticket in group:
                ticket.future.cancel()
                ticket.released = True
        self.groups.clear()
