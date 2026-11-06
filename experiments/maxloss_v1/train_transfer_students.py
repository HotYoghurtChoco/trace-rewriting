"""Train the transfer pair with both samplers grouped by the target's Clean lengths."""

from datasets import load_from_disk

from experiments.stage6c_cosrewrite_v1.run_stage6c3_students import run as run_seeded_students
from optimize import score_candidates as students


def run(cfg):
    # prepare_students.py aligns both arms to the Clean trainer's seed-137 order.
    clean = load_from_disk(cfg.original_traces_path).shuffle(seed=137)
    clean = clean.select(range(cfg.dataset_size_used_for_optimize))
    target, = cfg.proxy_models
    tokenizer = students._load_training_tokenizer(target["tokenizer"])
    tokens = students._tokenize_completion_only_batch(
        tokenizer, cfg.instruction_generation, clean["problem"], clean["original_trace"], target["name"]
    )
    clean_lengths = [len(ids) for ids in tokens["input_ids"]]

    trainer_factory = students.SFTTrainer

    def matched_trainer(*, train_dataset, args, **kwargs):
        # Match the lengths the Clean sampler sees after the trainer truncates input_ids.
        lengths = clean_lengths if args.max_length is None else [min(n, args.max_length) for n in clean_lengths]
        grouped_dataset = train_dataset.add_column(args.length_column_name, lengths)
        return trainer_factory(train_dataset=grouped_dataset, args=args, **kwargs)

    students.SFTTrainer = matched_trainer
    try:
        run_seeded_students(cfg)
    finally:
        students.SFTTrainer = trainer_factory


if __name__ == "__main__":
    run(students.ConfigManager().get_config())
