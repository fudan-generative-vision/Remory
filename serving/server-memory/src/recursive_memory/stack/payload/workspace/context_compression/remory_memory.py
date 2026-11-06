"""Tensor-free residual memory references and transition identities."""
from dataclasses import dataclass
import hashlib
import json

CAPABILITY = "remory_server_memory_v1"
ROUTE = "/remory/memory"


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def transition_digest(input_ids, payload):
    return digest(dict(schema=CAPABILITY, input_ids=list(input_ids),
                       payload={k: v for k, v in payload.items() if k != "memory_handle_publish"}))


@dataclass(frozen=True)
class RemoteMemory:
    """Only an identity/shape descriptor; never a client tensor."""
    descriptor: dict
    backend: str

    def __post_init__(self):
        from context_compression.sglang_compressor_hook import normalize_memory_handle_descriptor
        object.__setattr__(self, "descriptor", normalize_memory_handle_descriptor(self.descriptor))

    @property
    def shape(self):
        return tuple(self.descriptor["shape"])

    @property
    def ndim(self):
        return 2

    def to_json(self):
        return dict(schema=CAPABILITY, backend=self.backend, handle=dict(self.descriptor))

    @classmethod
    def from_json(cls, value):
        if set(value) != {"schema", "backend", "handle"} or value["schema"] != CAPABILITY:
            raise ValueError("invalid remote memory reference")
        return cls(value["handle"], value["backend"])
