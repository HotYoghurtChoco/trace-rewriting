"""CPU tests for text protection, completion masks and input-gradient search.

Run: python -m unittest experiments.maxloss_v1.test_optimizer -v
"""

from __future__ import annotations

import re
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.maxloss_v1.optimizer import FixedStudent, UnscorableTrace, optimize_trace
from experiments.maxloss_v1.text_edits import Edit, answer_start, propose_edits, validate_edit


TRACE = "First, calculate the total.\nTherefore, compute the result.\n\n**Final Answer:**\n\\boxed{12}"


class SmallTokenizer:
    def __init__(self):
        self.vocab = {}

    def ids(self, text):
        return [self.vocab.setdefault(word, len(self.vocab) + 3) for word in re.findall(r"\w+|[^\w\s]", text)]

    def encode(self, problem, trace):
        prompt = [1] + self.ids(problem) + [2]
        completion = self.ids(trace) + [0]  # EOS deliberately shares PAD=0.
        return {"input_ids": [prompt + completion], "completion_mask": [[0] * len(prompt) + [1] * len(completion)]}


class TinyCausalModel(nn.Module):
    """A differentiable causal LM with a real shifted cross-entropy objective."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 12)
        self.output = nn.Linear(12, 256)
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        denominator = torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        logits = self.output(torch.tanh(x.cumsum(1) / denominator))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


class TextPolicyTests(unittest.TestCase):
    def test_math_answer_and_echoed_problem_are_protected(self):
        problem = "Calculate the number of pages."
        trace = problem + "\nFirst, calculate the total.\n\\[\\text{calculate total}=12\\]\n\\text{compute the total}\n\n**Final Answer:**\nTherefore, \\boxed{12}"
        edits = propose_edits(trace, problem)
        self.assertTrue(edits)
        cutoff = answer_start(trace)
        for edit in edits:
            changed = edit.apply(trace)
            self.assertGreaterEqual(edit.start, len(problem))
            self.assertLess(edit.end, cutoff)
            self.assertEqual(validate_edit(trace, edit, problem), [])
            self.assertIn(r"\[\text{calculate total}=12\]", changed)
            self.assertIn(r"\text{compute the total}", changed)
            self.assertEqual(trace[cutoff:], changed[answer_start(changed):])

    def test_arithmetic_relation_and_negation_are_not_candidates(self):
        trace = "She has twice as many apples, not half.\nFirst, calculate the total.\n\n**Final Answer:** 12"
        for before, after in (("twice", "half"), ("not", "now"), ("12", "13"), ("calculate", "subtract")):
            start = trace.index(before)
            edit = Edit(start, start + len(before), before, after, "wording")
            self.assertTrue(validate_edit(trace, edit))

    def test_connective_only_at_sentence_boundary(self):
        trace = "If ready then, calculate it.\nThen, compute it.\n\n**Final Answer:** 12"
        connective_edits = [e for e in propose_edits(trace) if e.rule == "connective"]
        self.assertTrue(connective_edits)
        self.assertTrue(all(e.start == trace.index("Then,") for e in connective_edits))

    def test_no_answer_region_keeps_trace_unchanged(self):
        self.assertEqual(propose_edits("First, calculate something."), [])


class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.tokenizer = SmallTokenizer()
        self.model = TinyCausalModel().double()
        self.student = FixedStudent(self.model, self.tokenizer.encode)
        self.problem = "Find the total."

    def test_mask_keeps_eos_and_excludes_prompt(self):
        feature = self.student.feature(self.problem, TRACE)
        batch = self.student.batch(feature)
        self.assertTrue(all(label == -100 for label, mask in zip(batch["labels"][0].tolist(), feature.mask) if not mask))
        self.assertEqual(batch["labels"][0, -1].item(), 0)
        altered = self.student.feature(self.problem, TRACE.replace("calculate", "compute"))
        self.assertFalse(torch.equal(batch["labels"], self.student.batch(altered)["labels"]))

    def test_input_gradient_agrees_with_finite_difference_and_freezes_model(self):
        feature = self.student.feature(self.problem, TRACE)
        before = [p.detach().clone() for p in self.model.parameters()]
        _, gradient = self.student.loss_and_gradient(feature)
        batch = self.student.batch(feature)
        embeddings = self.model.get_input_embeddings()(batch.pop("input_ids")).detach()
        direction = torch.randn_like(embeddings)
        epsilon = 1e-5
        with torch.no_grad():
            plus = self.model(inputs_embeds=embeddings + epsilon * direction, **batch).loss
            minus = self.model(inputs_embeds=embeddings - epsilon * direction, **batch).loss
        numerical = float((plus - minus) / (2 * epsilon))
        analytical = float((gradient * direction).sum())
        self.assertAlmostEqual(numerical, analytical, places=7)
        for original, parameter in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(original, parameter))
            self.assertIsNone(parameter.grad)
            self.assertFalse(parameter.requires_grad)

    def test_true_loss_ascent_and_edit_budget(self):
        result = optimize_trace(self.student, self.problem, TRACE, max_rounds=3, candidates_per_round=4)
        self.assertGreater(result["accepted_edits"], 0)
        self.assertLessEqual(result["accepted_edits"], 3)
        self.assertGreater(result["loss_after"], result["loss_before"])
        for step in result["rounds"]:
            self.assertLessEqual(len(step["trials"]), 4)
            if step["accepted_edit"]:
                self.assertGreaterEqual(step["loss_after"] - step["loss_before"], 1e-4)
        self.assertAlmostEqual(result["loss_after"], self.student.loss(self.student.feature(self.problem, result["rewrite_trace"])))

    def test_no_improvement_preserves_original(self):
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
        result = optimize_trace(self.student, self.problem, TRACE)
        self.assertEqual(result["rewrite_trace"], TRACE)
        self.assertEqual(result["accepted_edits"], 0)
        self.assertEqual(result["stop_reason"], "no_loss_improvement")

    def test_gradient_shortlist_filters_a_small_pool_before_rescoring(self):
        from unittest.mock import patch

        edits = propose_edits(TRACE, self.problem)
        candidates = []
        for index, after in enumerate(("Firstly", "compute", "Hence")):
            edit = next(edit for edit in edits if edit.after == after)
            text = edit.apply(TRACE)
            candidates.append((3.0 - index, edit, text, self.student.feature(self.problem, text)))
        initial = self.student.loss(self.student.feature(self.problem, TRACE))
        # The lowest-gradient candidate would win an exhaustive loss search.
        losses = {
            tuple(candidate[3].ids): initial + gain
            for candidate, gain in zip(candidates, (0.1, 0.2, 1.0))
        }
        with patch("experiments.maxloss_v1.optimizer.rank_proposals", return_value=(candidates, {})):
            with patch.object(self.student, "loss", side_effect=lambda feature: losses[tuple(feature.ids)]) as score:
                result = optimize_trace(self.student, self.problem, TRACE, max_rounds=1, candidates_per_round=16)
        self.assertEqual(score.call_count, 2)
        self.assertEqual(result["rewrite_trace"], candidates[1][2])
        self.assertEqual(result["rounds"][0]["shortlist_size"], 2)

    def test_overlong_trace_is_not_silently_truncated(self):
        self.student.max_tokens = 5
        with self.assertRaises(UnscorableTrace):
            self.student.feature(self.problem, TRACE)


if __name__ == "__main__":
    unittest.main()
