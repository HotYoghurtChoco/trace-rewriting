"""Frozen-student input-gradient proposals followed by exact text rescoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Callable

import torch

from experiments.maxloss_v1.text_edits import propose_edits, validate_edit


class UnscorableTrace(ValueError):
    """An input cannot be scored in full under the configured token budget."""


@dataclass
class Feature:
    ids: list[int]
    mask: list[int]


class FixedStudent:
    """Freeze all parameters; differentiate only the detached input embeddings.

    `encode` returns the existing training helper's input_ids/completion_mask
    for one problem/response. No padding-ID equality is used to mask labels,
    so EOS remains a target even when EOS and PAD share an ID.
    """

    def __init__(self, model, encode: Callable, *, max_tokens: int = 1024):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.encode = encode
        self.max_tokens = max_tokens
        self.device = next(model.parameters()).device

    def feature(self, problem: str, trace: str) -> Feature:
        encoded = self.encode(problem, trace)
        ids, mask = encoded["input_ids"][0], encoded["completion_mask"][0]
        if len(ids) != len(mask) or any(value not in (0, 1) for value in mask):
            raise ValueError("Invalid completion mask")
        if len(ids) > self.max_tokens:
            raise UnscorableTrace(f"over_token_budget:{len(ids)}>{self.max_tokens}")
        if not sum(mask[1:]):
            raise UnscorableTrace("no_completion_targets")
        return Feature(ids, mask)

    def batch(self, feature: Feature) -> dict[str, torch.Tensor]:
        ids = torch.tensor([feature.ids], dtype=torch.long, device=self.device)
        labels = ids.clone()
        mask = torch.tensor([feature.mask], dtype=torch.bool, device=self.device)
        labels[~mask] = -100
        return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}

    def loss_and_gradient(self, feature: Feature) -> tuple[float, torch.Tensor]:
        batch = self.batch(feature)
        embeddings = self.model.get_input_embeddings()(batch.pop("input_ids")).detach().requires_grad_(True)
        loss = self.model(inputs_embeds=embeddings, **batch).loss
        gradient, = torch.autograd.grad(loss, embeddings)
        value = float(loss.detach())
        if not math.isfinite(value) or not torch.isfinite(gradient).all().item():
            raise RuntimeError("Non-finite input loss or gradient")
        return value, gradient.detach()

    @torch.no_grad()
    def loss(self, feature: Feature) -> float:
        value = float(self.model(**self.batch(feature)).loss)
        if not math.isfinite(value):
            raise RuntimeError("Non-finite candidate loss")
        return value

    @torch.no_grad()
    def directional_score(self, before: Feature, after: Feature, gradient: torch.Tensor) -> float:
        if len(before.ids) != len(after.ids) or before.mask != after.mask:
            raise ValueError("Directional proposal requires aligned token positions")
        positions = [i for i, (a, b) in enumerate(zip(before.ids, after.ids)) if a != b]
        if not positions:
            return 0.0
        embedding = self.model.get_input_embeddings()
        old_ids = torch.tensor([before.ids[i] for i in positions], device=self.device)
        new_ids = torch.tensor([after.ids[i] for i in positions], device=self.device)
        delta = embedding(new_ids).float() - embedding(old_ids).float()
        return float((gradient[0, positions].float() * delta).sum())


def rank_proposals(student, problem, trace, feature, gradient, seen):
    """Rank local, equal-token-length alternatives by g_E dot (E_new-E_old).

    This estimates the input-side change with the current labels held fixed.
    The subsequent true-loss computation rebuilds labels for each new trace.
    """
    ranked, rejected = [], Counter()
    for edit in propose_edits(trace, problem):
        reasons = validate_edit(trace, edit, problem)
        if reasons:
            rejected.update(reasons)
            continue
        candidate = edit.apply(trace)
        if candidate in seen:
            rejected["already_visited"] += 1
            continue
        try:
            candidate_feature = student.feature(problem, candidate)
        except UnscorableTrace:
            rejected["over_token_budget"] += 1
            continue
        if len(candidate_feature.ids) != len(feature.ids) or candidate_feature.mask != feature.mask:
            rejected["token_alignment_changed"] += 1
            continue
        if candidate_feature.ids == feature.ids:
            rejected["no_token_change"] += 1
            continue
        score = student.directional_score(feature, candidate_feature, gradient)
        ranked.append((score, edit, candidate, candidate_feature))
    ranked.sort(key=lambda item: (-item[0], item[1].start, item[1].after))
    return ranked, dict(rejected)


def optimize_trace(student, problem: str, trace: str, *, max_rounds=5, candidates_per_round=16, min_gain=1e-4):
    """Greedy ascent with at most one accepted span replacement per round."""
    if max_rounds < 1 or candidates_per_round < 1 or min_gain <= 0:
        raise ValueError("Search budgets and minimum gain must be positive")
    feature = student.feature(problem, trace)
    current, seen, rounds = trace, {trace}, []
    initial_loss = None
    current_loss = None
    accepted_count = 0
    stop_reason = "round_budget"
    for round_index in range(max_rounds):
        measured_loss, gradient = student.loss_and_gradient(feature)
        if initial_loss is None:
            initial_loss = measured_loss
        current_loss = measured_loss
        ranked, rejected = rank_proposals(student, problem, current, feature, gradient, seen)
        best = None
        best_loss = current_loss
        trials = []
        shortlist_size = min(candidates_per_round, (len(ranked) + 1) // 2)
        for directional, edit, candidate, candidate_feature in ranked[:shortlist_size]:
            exact_loss = student.loss(candidate_feature)
            trials.append({"edit": edit.to_dict(), "directional_score": directional, "loss": exact_loss})
            if exact_loss > best_loss and exact_loss - current_loss >= min_gain:
                best, best_loss = (edit, candidate, candidate_feature), exact_loss
        record = {
            "round": round_index + 1, "loss_before": current_loss,
            "eligible_candidates": len(ranked), "shortlist_size": shortlist_size,
            "rejected": rejected, "trials": trials,
            "accepted_edit": None, "loss_after": best_loss,
        }
        rounds.append(record)
        if best is None:
            stop_reason = "no_eligible_edit" if not ranked else "no_loss_improvement"
            break
        edit, current, feature = best
        record["accepted_edit"] = edit.to_dict()
        current_loss = best_loss
        seen.add(current)
        accepted_count += 1
    return {
        "original_trace": trace, "rewrite_trace": current,
        "loss_before": initial_loss, "loss_after": current_loss,
        "loss_gain": current_loss - initial_loss, "accepted_edits": accepted_count,
        "total_tokens": len(feature.ids), "completion_tokens": sum(feature.mask[1:]),
        "stop_reason": stop_reason, "rounds": rounds,
    }
