import argparse
import importlib.util
import os
import re
from typing import Callable, Optional

import yaml
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from lite_llm.configuration import LiteLlmConfig
from lite_llm.data_utils import (
    DataCollatorForPretraining,
    load_tokenized_dataset,
    split_train_val,
)
from lite_llm.modeling import LiteLlmForCausalLM


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_last_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for entry in os.listdir(output_dir):
        match = CHECKPOINT_PATTERN.fullmatch(entry)
        if not match:
            continue

        checkpoint_dir = os.path.join(output_dir, entry)
        if os.path.isdir(checkpoint_dir):
            candidates.append((int(match.group(1)), checkpoint_dir))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def parse_args(default_train_config: str, default_model_config: str, description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--train-config",
        default=default_train_config,
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--model-config",
        default=default_model_config,
        help="Path to the model YAML config.",
    )
    return parser.parse_args()


def _check_vocab_size(tokenizer, model_cfg: dict, model_config_path: str):
    """Tokenizer length must *fit* inside the model vocab (padding up is allowed
    for tensor-core alignment).
    """
    tok_len = len(tokenizer)
    model_vocab = model_cfg["vocab_size"]
    if tok_len > model_vocab:
        raise ValueError(
            "Tokenizer has more tokens than the model vocab allows: "
            f"tokenizer has {tok_len:,} tokens but "
            f"{model_config_path} sets vocab_size={model_vocab:,}."
        )
    if model_vocab - tok_len > 4096:
        print(
            f"Warning: model vocab_size={model_vocab:,} is much larger than "
            f"tokenizer length={tok_len:,} — the extra embeddings are wasted."
        )


def run_training(
    train_config_path: str,
    model_config_path: str,
    validate_fn: Optional[Callable[[dict, dict], None]] = None,
):
    train_cfg = load_config(train_config_path)
    model_cfg = load_config(model_config_path)

    if validate_fn is not None:
        validate_fn(train_cfg, model_cfg)

    if train_cfg.get("deepspeed") and importlib.util.find_spec("deepspeed") is None:
        raise ModuleNotFoundError(
            "DeepSpeed is not installed in the current environment. "
            "Run `uv sync --extra production --frozen` before using the production flow."
        )

    set_seed(train_cfg.get("seed", 42))

    tokenizer = None
    tokenizer_name = train_cfg.get("tokenizer_name")
    if tokenizer_name:
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        _check_vocab_size(tokenizer, model_cfg, model_config_path)
    else:
        print("Tokenizer: skipped (tokenizer_name not set, assuming pre-tokenized ids match model vocab)")

    config = LiteLlmConfig(**model_cfg)
    model = LiteLlmForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}")

    data_dir = train_cfg["data_dir"]
    max_seq_length = train_cfg["max_seq_length"]
    val_fraction = float(train_cfg.get("val_fraction", 0.0) or 0.0)
    max_val_tokens = train_cfg.get("max_val_tokens")
    print(f"Loading tokenized data from {data_dir} (seq_len={max_seq_length})")

    eval_dataset = None
    if val_fraction > 0:
        train_dataset, eval_dataset = split_train_val(
            data_dir,
            max_seq_length,
            val_fraction=val_fraction,
            max_val_tokens=max_val_tokens,
        )
        val_count = len(eval_dataset) if eval_dataset is not None else 0
        print(
            f"Dataset: {len(train_dataset):,} train sequences, "
            f"{val_count:,} val sequences"
        )
    else:
        train_dataset = load_tokenized_dataset(data_dir, max_seq_length)
        print(f"Dataset: {len(train_dataset):,} sequences")

    data_collator = DataCollatorForPretraining()

    logging_dir = train_cfg.get("logging_dir")
    if logging_dir:
        os.environ["TENSORBOARD_LOGGING_DIR"] = logging_dir

    training_kwargs = {
        "output_dir": train_cfg["output_dir"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "adam_beta1": train_cfg["adam_beta1"],
        "adam_beta2": train_cfg["adam_beta2"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "bf16": train_cfg["bf16"],
        "logging_steps": train_cfg["logging_steps"],
        "report_to": train_cfg.get("report_to", "none"),
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "deepspeed": train_cfg.get("deepspeed"),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", True),
        "max_steps": train_cfg.get("max_steps", -1),
        "use_cpu": train_cfg.get("use_cpu", False),
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 4),
        "dataloader_pin_memory": train_cfg.get(
            "dataloader_pin_memory",
            not train_cfg.get("use_cpu", False),
        ),
        "remove_unused_columns": False,
    }
    if "warmup_steps" in train_cfg:
        training_kwargs["warmup_steps"] = train_cfg["warmup_steps"]
    else:
        training_kwargs["warmup_ratio"] = train_cfg["warmup_ratio"]

    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = train_cfg.get("eval_strategy", "steps")
        training_kwargs["eval_steps"] = train_cfg.get(
            "eval_steps", train_cfg["save_steps"]
        )
        training_kwargs["per_device_eval_batch_size"] = train_cfg.get(
            "per_device_eval_batch_size", train_cfg["per_device_train_batch_size"]
        )

    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    last_checkpoint = None
    if train_cfg.get("resume_from_last_checkpoint", True):
        last_checkpoint = find_last_checkpoint(train_cfg["output_dir"])
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(os.path.join(train_cfg["output_dir"], "final"))
    print("Training complete!")


def main(
    default_train_config: str,
    default_model_config: str,
    description: str,
    validate_fn: Optional[Callable[[dict, dict], None]] = None,
):
    args = parse_args(default_train_config, default_model_config, description)
    # Resolve config paths to absolute paths before running, so we can keep
    # relative paths for output/data dirs without changing the process CWD.
    train_config = os.path.abspath(args.train_config)
    model_config = os.path.abspath(args.model_config)
    run_training(train_config, model_config, validate_fn=validate_fn)
