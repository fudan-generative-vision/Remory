"""Opt-in, fixed artifact-pair acceptance backed by local tensor verification.

This leaves both raw identities unchanged. It cannot accept arbitrary mappings
from server responses or manifests and is enabled only by the operator for this fixed, verified pair.
"""
import json
import hashlib
import os
from pathlib import Path

POLICY = 'h20_online250_verified_tensor_pair_20261109_v1'
LOCAL = dict(schema='remory_artifact_identity_v1',
    compressor_sha256='cf1fe99c2f51f330b4b93055b0da4f930e1b01d1841d0b9b9825a8e4dc386732',
    contract_sha256='ea4598529c83cd14447c967210f62d357c304104e76d58f08421c4e854a2b3c3',
    target_metadata_sha256='68dca6ab72fe8b44b54228bb020909c54ef960d6d8a779fbeff72bcdaa27e08c')
REMOTE = dict(LOCAL,
    compressor_sha256='59d36bcec81c44bb314b63afea4dae79d8cda53a8918a5364143d8ecb2a499e5',
    contract_sha256='29dc20537a10309f63077dc41c8bb7020fbfe22c8074c1e94817e90b697d8654')
HOOK = 'sha256:9dbfd8eca2c92481b9b09f918cb92bf9aecaddc7328438536213eaa68c1eed51'


def verified_pair(local, info, config):
    if os.environ.get('REMORY_H20_VERIFIED_PAIR') != POLICY:
        return False
    if local != LOCAL or info.get('remory_artifact_identity') != REMOTE:
        return False
    if (info.get('specforge_gpu_memory_handle') or {}).get('hook_abi_digest') != HOOK:
        return False
    proof = Path(os.environ['REMORY_H20_EQUIVALENCE_PROOF'])
    raw = proof.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '3ced7662cb3f103d19e2227de2c9b7b7750451badd6a90882bb2a6948ae85db5':
        return False
    data = json.loads(raw)
    # This is a local verification result, never obtained from the remote endpoint.
    for key, digest in [('current_safetensors', REMOTE['compressor_sha256']),
                        ('original_pt', LOCAL['compressor_sha256'])]:
        item = data['comparisons'][key]
        if (item['file_sha256'] != digest or item['tensor_count'] != 117
                or item['all_tensor_metadata_and_value_hashes_equal'] is not True
                or item['differences'] or item['all_floating_tensors_finite'] is not True):
            return False
    # Explicitly retain current inference policy and semantic geometry.
    return (config['residual']['block_tokens'] == 1024
        and config['residual']['compression_ratio'] == 16
        and config['residual']['memory_budget'] == 4096
        and config['compressor']['target_hidden_size'] == 5120)
