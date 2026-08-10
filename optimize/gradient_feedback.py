"""Reusable completion-only gradient scoring for Stage 6C.

The numerical procedure in this module is intentionally the same as the
formal Stage 6B-1c scorer:

* the supplied model is evaluated with dropout disabled;
* only trainable LoRA parameters contribute to the gradient;
* each example is processed separately with completion-only loss;
* inputs retain the first 1,024 total tokens;
* reference gradients are averaged with equal example weights in FP32; and
* candidate dot products and norms are accumulated in FP32.

This module never creates an optimizer and never updates model parameters.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from optimize.score_candidates import _tokenize_completion_only_batch


@dataclass(frozen=True)
class TokenMetadata:
    total_tokens_before_truncation: int
    total_tokens_kept: int
    completion_tokens_before_truncation: int
    completion_tokens_kept: int
    completion_tokens_removed: int
    eos_in_kept_completion: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceGradient:
    tensors: Dict[str, torch.Tensor]
    example_count: int
    loss_mean: float
    mean_gradient_norm: float
    per_example: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class CandidateGradientScore:
    loss: float
    gradient_norm: float
    gradient_dot: float
    gradient_cosine: float
    predicted_reference_loss_change_per_unit_step: float
    tokens: TokenMetadata

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tokens"] = self.tokens.to_dict()
        return payload


class CompletionOnlyGradientScorer:
    """Score traces against an equal-weight mean reference gradient."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        instruction: str,
        model_name: str,
        device: torch.device,
        compute_dtype: torch.dtype = torch.bfloat16,
        maximum_total_tokens: int = 1024,
    ) -> None:
        if maximum_total_tokens <= 0:
            raise ValueError("maximum_total_tokens must be positive")

        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.model_name = model_name
        self.device = device
        self.compute_dtype = compute_dtype
        self.maximum_total_tokens = maximum_total_tokens
        self.collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=True,
        )

        self.model.eval()
        self.model.config.use_cache = False
        self.trainable = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not self.trainable:
            raise RuntimeError("The scoring model has no trainable parameters")
        if not all("lora_" in name for name, _ in self.trainable):
            raise RuntimeError("A non-LoRA parameter is trainable")

    @property
    def trainable_tensor_count(self) -> int:
        return len(self.trainable)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.trainable)

    def build_feature(
        self,
        problem: str,
        response: str,
    ) -> Tuple[Dict[str, Any], TokenMetadata]:
        encoded = _tokenize_completion_only_batch(
            tokenizer=self.tokenizer,
            instruction=self.instruction,
            problems=[problem],
            responses=[response],
            model_name=self.model_name,
        )
        full_ids = encoded["input_ids"][0]
        full_mask = encoded["completion_mask"][0]
        if len(full_ids) != len(full_mask):
            raise RuntimeError("Token/mask length mismatch")

        kept_ids = full_ids[: self.maximum_total_tokens]
        kept_mask = full_mask[: self.maximum_total_tokens]
        if sum(kept_mask) <= 0:
            raise RuntimeError("Completion has no retained tokens")

        metadata = TokenMetadata(
            total_tokens_before_truncation=len(full_ids),
            total_tokens_kept=len(kept_ids),
            completion_tokens_before_truncation=sum(full_mask),
            completion_tokens_kept=sum(kept_mask),
            completion_tokens_removed=sum(
                full_mask[self.maximum_total_tokens :]
            ),
            eos_in_kept_completion=any(
                token_id == self.tokenizer.eos_token_id
                and mask_value == 1
                for token_id, mask_value in zip(
                    kept_ids,
                    kept_mask,
                    strict=True,
                )
            ),
        )
        return {
            "input_ids": kept_ids,
            "completion_mask": kept_mask,
        }, metadata

    def _model_batch(
        self,
        feature: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        collated = self.collator([dict(feature)])
        labels = collated["labels"][0]
        expected_labels = sum(feature["completion_mask"])
        actual_labels = int((labels != -100).sum().item())
        if actual_labels != expected_labels:
            raise RuntimeError("Completion-only label mask mismatch")
        return {
            key: value.to(self.device)
            for key, value in collated.items()
            if key
            in {
                "input_ids",
                "attention_mask",
                "labels",
                "position_ids",
            }
        }

    def _backward(self, feature: Mapping[str, Any]) -> float:
        self.model.zero_grad(set_to_none=True)
        batch = self._model_batch(feature)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
        ):
            output = self.model(**batch)
            loss = output.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")
        loss.backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for name, parameter in self.trainable:
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            if not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"Non-finite gradient: {name}")
        result = float(loss.detach().cpu())
        del batch, output, loss
        return result

    def _current_gradient_norm(self) -> float:
        squared_norm = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )
        for _, parameter in self.trainable:
            gradient = parameter.grad.detach().float()
            squared_norm.add_(gradient.square().sum())
        norm = math.sqrt(float(squared_norm.item()))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"Invalid gradient norm: {norm}")
        return norm

    def mean_reference_gradient(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        problem_key: str = "problem",
        response_key: str = "solution",
        progress_every: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ReferenceGradient:
        if not rows:
            raise ValueError("At least one reference row is required")

        reference_sum = {
            name: torch.zeros_like(
                parameter,
                dtype=torch.float32,
                device=self.device,
            )
            for name, parameter in self.trainable
        }
        per_example = []

        for index, row in enumerate(rows):
            feature, metadata = self.build_feature(
                str(row[problem_key]),
                str(row[response_key]),
            )
            loss = self._backward(feature)
            gradient_norm = self._current_gradient_norm()
            for name, parameter in self.trainable:
                reference_sum[name].add_(
                    parameter.grad.detach().float()
                )
            per_example.append(
                {
                    "reference_index": index,
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "tokens": metadata.to_dict(),
                }
            )
            if (
                progress_callback is not None
                and progress_every > 0
                and (index + 1) % progress_every == 0
            ):
                progress_callback(index + 1, len(rows))

        for value in reference_sum.values():
            value.div_(len(rows))

        squared_norm = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )
        for value in reference_sum.values():
            squared_norm.add_(value.square().sum())
        mean_norm = math.sqrt(float(squared_norm.item()))
        if not math.isfinite(mean_norm) or mean_norm <= 0.0:
            raise RuntimeError(
                f"Invalid mean reference-gradient norm: {mean_norm}"
            )

        return ReferenceGradient(
            tensors=reference_sum,
            example_count=len(rows),
            loss_mean=statistics.fmean(
                float(item["loss"]) for item in per_example
            ),
            mean_gradient_norm=mean_norm,
            per_example=per_example,
        )

    def score_response(
        self,
        *,
        problem: str,
        response: str,
        reference: ReferenceGradient,
    ) -> CandidateGradientScore:
        if reference.example_count <= 0:
            raise ValueError("Reference gradient has no examples")
        if set(reference.tensors) != {
            name for name, _ in self.trainable
        }:
            raise RuntimeError("Reference/trainable gradient key mismatch")

        feature, metadata = self.build_feature(problem, response)
        loss = self._backward(feature)
        squared_norm = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )
        dot_product = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )
        for name, parameter in self.trainable:
            gradient = parameter.grad.detach().float()
            squared_norm.add_(gradient.square().sum())
            dot_product.add_(
                (gradient * reference.tensors[name]).sum()
            )

        gradient_norm = math.sqrt(float(squared_norm.item()))
        gradient_dot = float(dot_product.item())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"Invalid candidate gradient norm: {gradient_norm}")
        if not math.isfinite(gradient_dot):
            raise RuntimeError(f"Invalid candidate gradient dot: {gradient_dot}")

        cosine = gradient_dot / (
            gradient_norm * reference.mean_gradient_norm
        )
        if not math.isfinite(cosine) or not -1.0001 <= cosine <= 1.0001:
            raise RuntimeError(f"Invalid candidate gradient cosine: {cosine}")

        return CandidateGradientScore(
            loss=loss,
            gradient_norm=gradient_norm,
            gradient_dot=gradient_dot,
            gradient_cosine=cosine,
            predicted_reference_loss_change_per_unit_step=-gradient_dot,
            tokens=metadata,
        )
