"""Run matched Stage 6C.3 students with seeded LoRA initialization.

The frozen Stage 6A training/scoring module is reused without changing its
source. Only its adapter factory is wrapped, within this process, so every
new student starts from the configured seed before LoRA weights are created.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from transformers import set_seed  # noqa: E402

from optimize import score_candidates as students  # noqa: E402


def run(cfg: argparse.Namespace) -> None:
    seed = int(cfg.seed)
    original_factory = students.get_peft_model

    def seeded_adapter_factory(*args: Any, **kwargs: Any) -> Any:
        set_seed(seed)
        print(f"stage6c3_student_lora_initialization_seed={seed}", flush=True)
        return original_factory(*args, **kwargs)

    students.get_peft_model = seeded_adapter_factory
    try:
        students.run_scoring(cfg)
    finally:
        students.get_peft_model = original_factory


if __name__ == "__main__":
    run(students.ConfigManager().get_config())
