"""SFT training runner kept separate from the pretraining runner."""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import re
import time
from typing import Callable, Optional

import torch
import yaml
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed

from lite_llm.configuration import LiteLlmConfig
from lite_llm.modeling import LiteLlmForCausalLM
from sft.collator import DataCollatorForChatSFT
from sft.data_utils import SftJsonlDataset, load_sft_records, split_sft_records, validate_messages_format
from sft.toy_tokenizer import ToyChatTokenizer


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


class SftLoggingCallback(TrainerCallback):
    """Print token throughput, ETA, elapsed time, and GPU memory on log steps."""

    def __init__(self, tokens_per_step: int):
        self._tokens_per_step = tokens_per_step
        self._train_start: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_step = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        del args, control, kwargs
        self._train_start = time.monotonic()
        self._last_time = self._train_start
        self._last_step = state.global_step

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        del args, control, kwargs
        if not state.is_world_process_zero or logs is None:
            return

        now = time.monotonic()
        step = state.global_step
        elapsed = now - self._train_start if self._train_start is not None else 0.0
        delta_steps = step - self._last_step
        delta_t = (now - self._last_time) if self._last_time is not None else 0.0
        tok_per_sec = None
        if delta_steps > 0 and delta_t > 0:
            tok_per_sec = delta_steps * self._tokens_per_step / delta_t

        max_steps = state.max_steps
        if max_steps and max_steps > 0 and step > 0:
            avg_sec_per_step = elapsed / step
            eta_sec = avg_sec_per_step * (max_steps - step)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            pct = 100.0 * step / max_steps
        else:
            eta_str = "N/A"
            pct = None

        gpu_info = ""
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_reserved() / 1024**3
            gpu_info = f"  gpu_mem={mem_gb:.1f}GB"

        parts = []
        if tok_per_sec is not None:
            parts.append(f"tok/s={tok_per_sec / 1e3:.2f}k")
        if pct is not None:
            parts.append(f"progress={pct:.1f}%  ({step}/{max_steps})")
        parts.append(f"ETA={eta_str}")
        parts.append(f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))}")
        print(f"  [sft] {'  '.join(parts)}{gpu_info}", flush=True)

        self._last_time = now
        self._last_step = step


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_last_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for entry in os.listdir(output_dir):
        match = CHECKPOINT_PATTERN.fullmatch(entry)
        if match:
            checkpoint_dir = os.path.join(output_dir, entry)
            if os.path.isdir(checkpoint_dir):
                candidates.append((int(match.group(1)), checkpoint_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _check_vocab_size(tokenizer, model_cfg: dict, model_config_path: str) -> None:
    tok_len = len(tokenizer)
    model_vocab = int(model_cfg["vocab_size"])
    if tok_len > model_vocab:
        raise ValueError(
            "Tokenizer has more tokens than the model vocab allows: "
            f"tokenizer has {tok_len:,} tokens but {model_config_path} "
            f"sets vocab_size={model_vocab:,}."
        )


def _load_tokenizer(train_cfg: dict, model_cfg: dict):
    if train_cfg.get("tokenizer_type") == "toy_chatml":
        return ToyChatTokenizer(vocab_size=int(model_cfg["vocab_size"]))

    tokenizer_name = train_cfg["tokenizer_name"]
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_training_args(train_cfg: dict, eval_dataset) -> TrainingArguments:
    kwargs = {
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
        kwargs["warmup_steps"] = train_cfg["warmup_steps"]
    else:
        kwargs["warmup_ratio"] = train_cfg["warmup_ratio"]

    if eval_dataset is not None:
        kwargs["eval_strategy"] = train_cfg.get("eval_strategy", "steps")
        kwargs["eval_steps"] = train_cfg.get("eval_steps", train_cfg["save_steps"])
        kwargs["per_device_eval_batch_size"] = train_cfg.get(
            "per_device_eval_batch_size",
            train_cfg["per_device_train_batch_size"],
        )
    return TrainingArguments(**kwargs)


def run_sft_training(
    train_config_path: str,
    model_config_path: str,
    validate_fn: Optional[Callable[[dict, dict], None]] = None,
) -> None:
    train_cfg = load_config(train_config_path)
    model_cfg = load_config(model_config_path)

    if validate_fn is not None:
        validate_fn(train_cfg, model_cfg)
    if train_cfg.get("deepspeed") and importlib.util.find_spec("deepspeed") is None:
        raise ModuleNotFoundError(
            "DeepSpeed is not installed. Run `uv sync --extra production --frozen` "
            "before using production SFT."
        )

    set_seed(train_cfg.get("seed", 42))

    tokenizer = _load_tokenizer(train_cfg, model_cfg)
    _check_vocab_size(tokenizer, model_cfg, model_config_path)

    config = LiteLlmConfig(**model_cfg)
    pretrained_path = train_cfg["pretrained_model_path"]
    print(f"Loading SFT base checkpoint: {pretrained_path}")
    model = LiteLlmForCausalLM.from_pretrained(pretrained_path, config=config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}")

    records = load_sft_records(train_cfg["data_dir"])
    validate_messages_format(records)
    train_records, val_records = split_sft_records(
        records,
        val_fraction=float(train_cfg.get("val_fraction", 0.0) or 0.0),
        seed=train_cfg.get("seed", 42),
    )
    train_dataset = SftJsonlDataset(train_records)
    eval_dataset = SftJsonlDataset(val_records) if val_records is not None else None
    if eval_dataset is None:
        print(f"Dataset: {len(train_dataset):,} train conversations")
    else:
        print(
            f"Dataset: {len(train_dataset):,} train, "
            f"{len(eval_dataset):,} val conversations"
        )

    max_seq_length = int(train_cfg["max_seq_length"])
    collator = DataCollatorForChatSFT(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        train_on_assistant_end=train_cfg.get("train_on_assistant_end", True),
    )

    logging_dir = train_cfg.get("logging_dir")
    if logging_dir:
        os.environ["TENSORBOARD_LOGGING_DIR"] = logging_dir

    training_args = _build_training_args(train_cfg, eval_dataset)
    tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max_seq_length
        * training_args.world_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[SftLoggingCallback(tokens_per_step)],
    )

    last_checkpoint = None
    if train_cfg.get("resume_from_last_checkpoint", True):
        last_checkpoint = find_last_checkpoint(train_cfg["output_dir"])
        if last_checkpoint is not None:
            print(f"Resuming SFT from checkpoint: {last_checkpoint}")

    print("Starting SFT training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = os.path.join(train_cfg["output_dir"], "final")
    trainer.save_model(final_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(final_dir)
    print("SFT training complete!")


def parse_args(default_train_config: str, default_model_config: str, description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-config", default=default_train_config)
    parser.add_argument("--model-config", default=default_model_config)
    return parser.parse_args()


def main(
    default_train_config: str,
    default_model_config: str,
    description: str,
    validate_fn: Optional[Callable[[dict, dict], None]] = None,
):
    args = parse_args(default_train_config, default_model_config, description)
    run_sft_training(
        os.path.abspath(args.train_config),
        os.path.abspath(args.model_config),
        validate_fn=validate_fn,
    )

