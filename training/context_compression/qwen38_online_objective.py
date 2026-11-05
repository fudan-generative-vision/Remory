"""Raw-history teacher and summary-conditioned recursive student, frozen reader."""
from dataclasses import dataclass
import torch
from context_compression.qwen38_residual_data import ResidualEvent
from context_compression.qwen38_residual_training import ResidualObjective, ResidualMemory
from context_compression.qwen38_residual_pyramid import plan_pyramid


@dataclass(frozen=True)
class OnlineEvent(ResidualEvent):
    raw_history_ids: tuple = ()
    source_memory: object = None
    source_memory_positions: tuple = ()
    source_memory_depths: tuple = ()


class OnlineObjective(ResidualObjective):
    def raw_teacher_and_source(self, event):
        if event.raw_history_ids == event.history_ids and event.source_memory is None:
            return super().raw_teacher_and_source(event)
        raw = event.prefix_ids + event.raw_history_ids + event.opening_ids
        self._check_length(len(raw) + len(event.target_ids), 'complete raw teacher')
        with self._frozen_eval():
            hidden = self.target.student_hidden(self._embed(raw + event.target_ids), grad=False)
            teacher = hidden[:, len(raw)-1:len(raw)-1+len(event.target_ids)].detach().to('cpu', copy=True)
            del hidden
            source = self._embed(event.prefix_ids + event.history_ids).clone()
            if event.source_memory is not None:
                source[:, list(event.source_memory_positions)] = event.source_memory.detach().to(source)
            features, hidden = self._capture(source, len(event.prefix_ids), source.shape[1])
            del hidden, source
        return features, teacher

    def compress_nodes(self, items, summary_features, input_depth):
        return super().compress_nodes(items, summary_features, min(input_depth, self.max_depth-1))

    def build_memory(self, event, summary, raw_features, summary_features):
        depths = [0] * ((len(event.history_ids)+self.block-1)//self.block)
        for p,d in zip(event.source_memory_positions,event.source_memory_depths):
            b=(p-len(event.prefix_ids))//self.block
            depths[b]=max(depths[b],d)
        plan=plan_pyramid(len(event.history_ids),budget=self.memory_budget,block=self.block,
                          ratio=self.ratio,max_depth=None,leaf_input_depths=depths)
        frontier=set(plan.frontier);outputs={};replay={}
        with torch.no_grad():
            position=0
            while position<len(plan.nodes):
                first=plan.nodes[position];group=[]
                for node in plan.nodes[position:position+self.node_batch_size]:
                    if node.depth!=first.depth or bool(node.children)!=bool(first.children):break
                    if node.children and any(k not in outputs for k in node.children):break
                    group.append(node)
                if not group:raise RuntimeError('unresolved recursive dependency')
                if first.children:
                    children=torch.cat([torch.cat([outputs[k] for k in n.children],dim=1) for n in group],dim=0)
                    with self._frozen_eval():
                        prefix=self._embed(event.prefix_ids).expand(len(group),-1,-1)
                        inp=torch.cat([prefix,children.to(prefix)],dim=1)
                        encoded,_=self._capture(inp,prefix.shape[1],inp.shape[1])
                    features=list(encoded.split(1))
                    for node in group:
                        for k in node.children:del outputs[k]
                else:features=[raw_features[:,n.start:n.stop] for n in group]
                values=self.compress_nodes(features,summary_features,first.depth-1)
                for node,feature,value in zip(group,features,values.split(1)):
                    outputs[node.index]=self._cache(value)
                    if node.index in frontier:replay[node.index]=self._cache(feature)
                position+=len(group)
            memory=torch.cat([outputs[k] for k in plan.frontier],dim=1).to(self.target.device)
        return ResidualMemory(memory,summary['summary_sha256'],event.history_sha256,plan,replay)
