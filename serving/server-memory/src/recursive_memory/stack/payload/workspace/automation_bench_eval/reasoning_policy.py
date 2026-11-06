"""Paired native compact profiles retain explicit xhigh reasoning."""


def validate_profile(config):
    expected = {'mode': 'native_compact',
                'reasoning_effort': 'xhigh', 'serving_eviction_policy': 'checkpoint_1024_v1'}
    if config.get('compressor') not in {'residual', 'none'} or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('This release only runs native compact + xhigh with residual or text-only memory; use a new pool')
    if config.get('reasoning_sweep'):
        raise ValueError('Reasoning sweeps are not distributed in this release')


def task_identity(args, task_id):
    validate_profile(vars(args))
    if getattr(args, 'source_task_id', None) not in (None, task_id):
        raise ValueError('Reasoning-sweep cells cannot be reused by this release')
    return task_id, {'reasoning_effort': 'xhigh'}
