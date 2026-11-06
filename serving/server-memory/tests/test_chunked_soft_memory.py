import importlib.util
from pathlib import Path
import sys, unittest
p=Path(__file__).resolve().parents[1]/'src/recursive_memory/stack/payload/sglang_overlay/python/sglang/srt/mem_cache/soft_memory_cache_contract.py'
spec=importlib.util.spec_from_file_location('soft_memory_contract',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class ChunkMapping(unittest.TestCase):
    def test_partition_scatter_exactly_once(self):
        positions=[0,31,8190,8191,8192,8193,16383,16384,17000]
        seen=[]
        for start,end in [(0,8192),(8192,16384),(16384,18000)]:
            plan=m.plan_packed_soft_memory_rows(memory_positions_per_request=[positions],extend_prefix_lens=[start],extend_seq_lens=[end-start],request_total_lens=[18000])[0]
            seen.extend(plan.embedding_row_indices)
            self.assertEqual([positions[i]-start for i in plan.embedding_row_indices],list(plan.packed_token_positions))
        self.assertEqual(seen,list(range(len(positions))))
    def test_packed_mixed_requests(self):
        plans=m.plan_packed_soft_memory_rows(memory_positions_per_request=[[1,5,10],None,[2,6,12]],extend_prefix_lens=[4,0,4],extend_seq_lens=[4,3,4],request_total_lens=[16,3,16])
        self.assertEqual(plans[0].embedding_row_indices,(1,));self.assertEqual(plans[0].packed_token_positions,(1,))
        self.assertEqual(plans[2].embedding_row_indices,(1,));self.assertEqual(plans[2].packed_token_positions,(9,))
    def test_invalid_full_request_still_rejected(self):
        with self.assertRaises(ValueError):m.plan_packed_soft_memory_rows(memory_positions_per_request=[[16]],extend_prefix_lens=[0],extend_seq_lens=[8],request_total_lens=[16])
    def test_legacy_unbounded_position_rejected(self):
        with self.assertRaises(ValueError):m.plan_packed_soft_memory_rows(memory_positions_per_request=[[9]],extend_prefix_lens=[0],extend_seq_lens=[8])
    def test_invalid_total_rejected(self):
        for total in [True,7]:
            with self.assertRaises(ValueError):m.plan_packed_soft_memory_rows(memory_positions_per_request=[[1]],extend_prefix_lens=[0],extend_seq_lens=[8],request_total_lens=[total])
if __name__=='__main__':unittest.main()


def test_atomic_prefill_preserves_multimodal_memory_coordinates():
    # Extract the scheduler policy without importing CUDA runtime dependencies.
    import ast
    from types import SimpleNamespace
    scheduler = p.parents[1] / 'managers/scheduler.py'
    tree = ast.parse(scheduler.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_specforge_requires_atomic_prefill')
    method.decorator_list = []
    method.args.args[0].annotation = None
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(scheduler), 'exec'), scope)
    atomic = scope[method.name]
    for mode, images, expected in [
        ('recover', object(), True), ('encode', None, True),
        ('encode', object(), True), ('recover', None, False),
        (None, object(), False), (None, None, False),
    ]:
        assert atomic(SimpleNamespace(specforge_compressor_mode=mode,
                                      multimodal_inputs=images)) is expected
