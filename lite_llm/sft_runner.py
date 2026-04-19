"""SFT training entrypoint.

Mirrors ``train_runner.py`` structure but uses TRL ``SFTTrainer`` with
``DataCollatorForCompletionOnlyLM`` for loss masking on assistant tokens only.

Key differences from pretraining:
- Loads model from a pretrained checkpoint (not random init).
- Uses SFTTrainer instead of Trainer.
- Uses DataCollatorForCompletionOnlyLM for assistant-only loss.
- Loads JSONL conversations (not packed .npy token arrays).
- Always requires a tokenizer (for chat template application).
"""

import os
from typing import Callable, Optional

import torch
from transformers import AutoTokenizer, set_seed

try:
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra sft` first.") from exc

from lite_llm.configuration import LiteLlmConfig
from lite_llm.modeling import LiteLlmForCausalLM
from lite_llm.sft_data_utils import (
    load_sft_dataset,
    split_sft_train_val,
    validate_messages_format,
)
from lite_llm.train_runner import (
    RichLoggingCallback,
    find_last_checkpoint,
    load_config,
    parse_args,
)


def run_sft_training(
    train_config_path: str,
    model_config_path: str,
    validate_fn: Optional[Callable[[dict, dict], None]] = None,
):
    train_cfg = load_config(train_config_path)
    model_cfg = load_config(model_config_path)

    if validate_fn is not None:
        validate_fn(train_cfg, model_cfg)

    set_seed(train_cfg.get("seed", 42))

    # --- Tokenizer (required for SFT chat template) ---
    tokenizer_name = train_cfg["tokenizer_name"]
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model from pretrained checkpoint ---
    pretrained_path = train_cfg["pretrained_model_path"]
    config = LiteLlmConfig(**model_cfg)
    print(f"Loading model from: {pretrained_path}")
    model = LiteLlmForCausalLM.from_pretrained(pretrained_path, config=config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- SFT Dataset ---
    data_dir = train_cfg["data_dir"]
    max_seq_length = train_cfg["max_seq_length"]
    print(f"Loading SFT data from {data_dir}")

    dataset = load_sft_dataset(data_dir)
    validate_messages_format(dataset)

    val_fraction = float(train_cfg.get("val_fraction", 0.0) or 0.0)
    if val_fraction > 0:
        train_dataset, eval_dataset = split_sft_train_val(dataset, val_fraction)
        print(
            f"Dataset: {len(train_dataset):,} train, "
            f"{len(eval_dataset):,} val conversations"
        )
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Dataset: {len(train_dataset):,} conversations")

    # --- Loss masking: only compute loss on assistant responses ---
    # Qwen ChatML: assistant responses start with <|im_start|>assistant\n
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    # --- TrainingArguments (same YAML→kwargs pattern as pretraining) ---
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

    from transformers import TrainingArguments

    training_args = TrainingArguments(**training_kwargs)

    # Tokens consumed per optimizer step
    tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max_seq_length
        * training_args.world_size
    )

    logging_dir = train_cfg.get("logging_dir")
    if logging_dir:
        os.environ["TENSORBOARD_LOGGING_DIR"] = logging_dir

    # --- SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        max_seq_length=max_seq_length,
        callbacks=[RichLoggingCallback(tokens_per_step)],
    )

    print("Starting SFT training...")
    last_checkpoint = None
    if train_cfg.get("resume_from_last_checkpoint", True):
        last_checkpoint = find_last_checkpoint(train_cfg["output_dir"])
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(os.path.join(train_cfg["output_dir"], "final"))
    print("SFT training complete!")


def main(
    default_train_config: str,
    default_model_config: str,
    description: str,
    validate_fn: Optional[Callable[[dict, dict], None]] = None,
):
    args = parse_args(default_train_config, default_model_config, description)
    train_config = os.path.abspath(args.train_config)
    model_config = os.path.abspath(args.model_config)
    run_sft_training(train_config, model_config, validate_fn=validate_fn)
