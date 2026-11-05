import types

import pytest
import torch

from context_compression.qwen38_reverse_kl import (
    ReverseKLOutput,
    backward_full_responses,
    chunked_reverse_kl,
    response_inputs,
)


@pytest.mark.parametrize("chunk", [1, 3, 32])
def test_reverse_direction_dense_gradient_and_teacher_detached(chunk):
    torch.manual_seed(3)
    head = torch.nn.Linear(4, 7, bias=False).requires_grad_(False)
    student = torch.randn(1, 5, 4, requires_grad=True)
    teacher = torch.randn(1, 5, 4, requires_grad=True)
    weights = torch.tensor([1.7, 1.2, 1.1, 1.05, 1.01])
    log_s = head(student).float().log_softmax(-1)
    log_t = head(teacher.detach()).float().log_softmax(-1)
    reference = (
        (log_s.exp() * (log_s - log_t)).sum(-1)[0] * weights
    ).sum() / weights.sum()
    gradient = torch.autograd.grad(reference, student)[0]
    forward = (
        (log_t.exp() * (log_t - log_s)).sum(-1)[0] * weights
    ).sum() / weights.sum()
    result = chunked_reverse_kl(
        student, teacher, head, weights=weights, chunk_tokens=chunk
    )
    result.loss.backward()
    torch.testing.assert_close(result.loss, reference)
    torch.testing.assert_close(student.grad, gradient)
    assert teacher.grad is None and head.weight.grad is None
    assert not torch.isclose(result.loss, forward)
    expected_agreement = (log_s.argmax(-1) == log_t.argmax(-1)).float().mean()
    torch.testing.assert_close(result.top1_agreement, expected_agreement)
    assert result.token_count == 5


def test_zero_kl_for_same_distribution():
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    x = torch.randn(1, 9, 4, requires_grad=True)
    r = chunked_reverse_kl(x, x.detach(), head, weights=torch.ones(9), chunk_tokens=4)
    assert abs(float(r.loss)) < 1e-7
    assert float(r.top1_agreement) == 1.0


def test_raw_prefix_and_compressed_history_have_no_future():
    example = {
        "prefix_ids": [10],
        "segments": [
            {"context_ids": [11], "opening_ids": [12], "response_ids": [13, 14]},
            {"context_ids": [15], "opening_ids": [12], "response_ids": [16, 17]},
        ],
    }
    a, b = list(response_inputs(example))
    assert a["teacher_prefix_ids"] == [10, 11, 12]
    assert a["source_ids"] == [11]
    assert b["teacher_prefix_ids"] == [10, 11, 12, 13, 14, 15, 12]
    assert b["source_ids"] == [12, 13, 14, 15]
    assert b["target_ids"] == [16, 17]


def test_detached_folds_and_one_forward_for_a_32781_token_response():
    parameter = torch.nn.Parameter(torch.tensor(0.3))
    observed = []
    example = {
        "prefix_ids": [10],
        "segments": [
            {
                "context_ids": [11] * 1025,
                "opening_ids": [12],
                "response_ids": [13] * 32781,
            },
            {"context_ids": [15], "opening_ids": [12], "response_ids": [16, 17]},
        ],
    }

    def fold(event, previous):
        assert previous is None or not previous.requires_grad
        value = parameter + (previous.sum() if previous is not None else 0.0)
        return value.reshape(1, 1, 1)

    def full_response(event, memory, teacher):
        observed.append(len(event["target_ids"]))
        assert memory.requires_grad
        return ReverseKLOutput(
            memory.square().mean(), torch.tensor(0.5), len(event["target_ids"])
        )

    objective = types.SimpleNamespace(
        fold=fold, raw_teacher=lambda p, t: torch.zeros(1), full_response=full_response
    )
    metrics = backward_full_responses(example, objective)
    assert observed == [32781, 2]
    assert metrics["reader_forwards"] == metrics["teacher_forwards"] == 2
    assert metrics["tokens"] == 32783
    assert parameter.grad is not None and torch.isfinite(parameter.grad)


def test_current_fold_gradient_is_not_detached_with_the_previous_memory():
    a = torch.nn.Parameter(torch.tensor(0.2))
    previous = torch.tensor([[[4.0]]], requires_grad=True)
    current = a * previous.detach()
    r = current.square().sum()
    r.backward()
    assert previous.grad is None
    assert float(a.grad) == pytest.approx(6.4)
