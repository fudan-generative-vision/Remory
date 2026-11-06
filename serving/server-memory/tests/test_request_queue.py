import asyncio

import pytest

from recursive_memory.stack.request_queue import AdmissionQueue, Backend, QueueFull


def available(backend, queue):
    backend.observed_load = backend.active
    backend.observed_starts = backend.starts
    queue.pump()


def test_groups_rotate_without_starvation_and_preserve_fifo():
    async def check():
        b = Backend("a", "http://a", "model", 1, healthy=True)
        q = AdmissionQueue([b])
        first = q.submit(group="a", model="model")
        a2 = q.submit(group="a", model="model")
        b1 = q.submit(group="b", model="model")
        a3 = q.submit(group="a", model="model")
        q.release(first)
        available(b, q)
        assert a2.backend is b and b1.backend is None
        q.release(a2)
        available(b, q)
        assert b1.backend is b and a3.backend is None
        q.release(b1)
        available(b, q)
        assert a3.backend is b
        q.release(a3)
        assert b.active == 0
    asyncio.run(check())


def test_load_limits_batches_pinning_and_cancel_do_not_leak_capacity():
    async def check():
        a = Backend("a", "http://a", "model", 2, healthy=True, observed_load=2)
        b = Backend("b", "http://b", "model", 2, healthy=True)
        q = AdmissionQueue([a, b], max_pending=1)
        pinned = q.submit(group="pinned", model="model", pinned="a")
        with pytest.raises(QueueFull):
            q.submit(group="full", model="model")
        q.release(pinned)
        running = q.submit(group="other", model="model", units=2)
        assert running.backend is b and b.active == 2
        blocked = q.submit(group="pinned", model="model", pinned="a")
        q.release(blocked)
        q.release(running)
        q.release(running)
        assert q.pending == 0 and b.active == 0
        with pytest.raises(ValueError):
            q.submit(group="too-big", model="model", units=3)
    asyncio.run(check())


def test_dead_pinned_backend_does_not_block_other_groups_or_fail_over():
    async def check():
        a = Backend("a", "http://a", "model", 1)
        b = Backend("b", "http://b", "model", 1, healthy=True)
        q = AdmissionQueue([a, b])
        pinned = q.submit(group="pinned", model="model", pinned="a")
        normal = q.submit(group="other", model="model")
        assert normal.backend is b and pinned.backend is None
        q.close()
        assert pinned.future.cancelled()
        q.release(normal)
        assert b.active == 0
    asyncio.run(check())


def test_queued_encoder_versions_recheck_registry_and_advertised_capability():
    async def check():
        capability = "remory_encoder_versions_v1"
        local = Backend("local", "http://local", "model", 1, healthy=True,
                        observed_load=1, info={"request_capabilities": [capability]})
        remote = Backend("remote", "http://remote", "model", 1, healthy=True,
                         supports_local_encoder_versions=False,
                         info={"request_capabilities": [capability]})
        q = AdmissionQueue([local, remote])
        versioned = q.submit(group="versioned", model="model", requires_local_encoder_versions=True)
        assert versioned.backend is None
        with pytest.raises(ValueError, match="registry"):
            q.submit(group="bad-pin", model="model", pinned="remote", requires_local_encoder_versions=True)
        plain = q.submit(group="plain", model="model")
        assert plain.backend is remote
        q.release(plain)
        available(remote, q)
        assert versioned.backend is None
        local.observed_load = 0
        local.info["request_capabilities"] = []
        q.pump()
        assert versioned.backend is None
        local.info["request_capabilities"] = [capability]
        local.supports_local_encoder_versions = False
        q.pump()
        assert versioned.backend is None
        local.supports_local_encoder_versions = True
        q.pump()
        assert await versioned.future is local
        q.release(versioned)
        assert not q.pending and local.active == remote.active == 0
    asyncio.run(check())


def test_memory_ticket_preserves_stateless_exclusion_during_queue_pumping():
    async def check():
        local = Backend("local", "http://local", "model", 1, healthy=True, observed_load=1)
        remote = Backend("remote", "http://remote", "model", 1, healthy=True, stateless_only=True)
        queue = AdmissionQueue([local, remote])
        memory = queue.submit(group="memory", model="model", requires_full_backend=True)
        assert memory.backend is None
        with pytest.raises(ValueError, match="stateless_only"):
            queue.submit(group="bad-pin", model="model", pinned="remote", requires_full_backend=True)
        plain = queue.submit(group="plain", model="model")
        assert plain.backend is remote
        queue.release(plain)
        available(remote, queue)
        assert memory.backend is None
        local.observed_load = 0
        local.stateless_only = True
        queue.pump()
        assert memory.backend is None
        local.stateless_only = False
        queue.pump()
        assert await memory.future is local
        queue.release(memory)
    asyncio.run(check())
