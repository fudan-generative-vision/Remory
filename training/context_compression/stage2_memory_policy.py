"""Spend activation memory to avoid recomputing a bounded number of reader layers."""
from context_compression.qwen38_residual_training import ResidualObjective
from context_compression.qwen38_residual_data import summary_envelope
from context_compression.qwen38_residual_pyramid import plan_pyramid


def retained_layer_count(sequence_tokens, layer_count, layer_token_budget):
    if sequence_tokens < 1 or layer_count < 1 or layer_token_budget < 0:
        raise ValueError("invalid reader activation budget")
    return min(layer_count, layer_token_budget // sequence_tokens)


class MemoryBudgetResidualObjective(ResidualObjective):
    def __init__(self, *args, reader_layer_token_budget=0, reader_retain_min_tokens=0,
                 short_response_node_batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reader_checkpoint_after_tokens:
            raise ValueError("layer budget and whole-reader threshold are exclusive")
        self.reader_layer_token_budget = reader_layer_token_budget
        if reader_retain_min_tokens < 0 or short_response_node_batch_size < 1:
            raise ValueError("invalid conditional memory policy")
        self.reader_retain_min_tokens = reader_retain_min_tokens
        self.long_response_node_batch_size = self.node_batch_size
        self.short_response_node_batch_size = short_response_node_batch_size
        self.reader_retained_layers = 0
        self.reader_sequence_tokens = 0

    def student_length(self, event, summary, memory_tokens):
        opening = event.opening_ids if event.student_opening_ids is None else event.student_opening_ids
        return (len(event.prefix_ids) + len(summary_envelope(summary["ids"], self.contract))
                  + len(self.contract["student_memory_before_ids"]) + memory_tokens
                  + len(self.contract["student_memory_after_ids"]) + len(opening) + len(event.target_ids))

    def response_loss(self, event, summary, memory, teacher):
        length = self.student_length(event, summary, memory.shape[1])
        layers = self.target.language_model.layers
        budget = self.reader_layer_token_budget if length > self.reader_retain_min_tokens else 0
        retained = retained_layer_count(length, len(layers), budget)
        # These native flags affect only activation retention/recomputation.
        # All layers still run, and every input gradient is propagated.
        for index, layer in enumerate(layers):
            layer.gradient_checkpointing = index >= retained
        self.reader_retained_layers, self.reader_sequence_tokens = retained, length
        return super().response_loss(event, summary, memory, teacher)

    def backward_response(self, event, summary):
        plan = plan_pyramid(len(event.history_ids), budget=self.memory_budget,
                            block=self.block, ratio=self.ratio, max_depth=self.max_depth)
        length = self.student_length(event, summary, plan.soft_tokens)
        self.node_batch_size = (self.long_response_node_batch_size if length > self.reader_retain_min_tokens
                               else self.short_response_node_batch_size)
        metrics = super().backward_response(event, summary)
        metrics.update(reader_retained_layers=self.reader_retained_layers,
                       reader_sequence_tokens=self.reader_sequence_tokens,
                       effective_node_batch_size=self.node_batch_size)
        return metrics
