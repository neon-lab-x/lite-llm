"""SFT training entrypoint.

Mirrors ``train_runner.py`` structure but uses TRL ``SFTTrainer`` with
assistant-only loss masking (compatible with old/new TRL APIs).

Key differences from pretraining:
- Loads model from a pretrained checkpoint (not random init).
- Uses SFTTrainer instead of Trainer.
- Uses completion-only masking for assistant responses.
- Loads JSONL conversations (not packed .npy token arrays).
- Always requires a tokenizer (for chat template application).
"""

import importlib.util
import inspect
import os
from typing import Callable, Optional

from transformers import AutoTokenizer, set_seed

try:
    from trl import SFTTrainer
except ModuleNotFoundError as exc:
    raise SystemExit("Run `uv sync --extra sft` first.") from exc

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    DataCollatorForCompletionOnlyLM = None

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

from lite_llm.configuration import LiteLlmConfig
from lite_llm.modeling import LiteLlmForCausalLM
from lite_llm.sft_data_utils import (
    load_sft_dataset,
    split_sft_train_val,
    validate_messages_format,
)
from lite_llm.train_runner import (
    RichLoggingCallback,
    _check_vocab_size,
    find_last_checkpoint,
    load_config,
    parse_args,
)


def _trl_supports_legacy_completion_collator() -> bool:
    return DataCollatorForCompletionOnlyLM is not None


def _find_subsequence_positions(sequence: list[int], pattern: list[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    positions = []
    max_start = len(sequence) - len(pattern) + 1
    for start in range(max_start):
        if sequence[start : start + len(pattern)] == pattern:
            positions.append(start)
    return positions


def _mask_assistant_spans(
    input_ids: list[int],
    response_token_ids: list[int],
    end_token_ids: list[int],
    attention_mask: Optional[list[int]] = None,
) -> list[int]:
    """Mask non-assistant tokens with -100 for completion-only loss."""
    labels = [-100] * len(input_ids)
    if attention_mask is not None and len(attention_mask) == len(input_ids):
        valid_length = int(sum(1 for x in attention_mask if int(x) != 0))
    else:
        valid_length = len(input_ids)
    valid_input_ids = input_ids[:valid_length]

    starts = _find_subsequence_positions(valid_input_ids, response_token_ids)
    if not starts:
        return labels

    for start in starts:
        assistant_start = start + len(response_token_ids)
        if assistant_start >= valid_length:
            continue
        end_positions = _find_subsequence_positions(
            valid_input_ids[assistant_start:],
            end_token_ids,
        )
        if end_positions:
            assistant_end = assistant_start + end_positions[0]
        else:
            # If assistant segment is truncated, stop at non-padding tokens only.
            assistant_end = valid_length
        for idx in range(assistant_start, assistant_end):
            labels[idx] = input_ids[idx]

    return labels


def _build_response_template_ids(tokenizer, response_template: str) -> list[int]:
    """Build response template token IDs with verification against a real conversation.

    Standalone ``tokenizer.encode(template)`` might produce different token IDs than
    when the same text appears inside a full conversation (BPE merge rules can be
    context-sensitive).  We verify by tokenizing a sample conversation and confirming
    the standalone IDs appear as a contiguous subsequence.
    """
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    try:
        sample_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "_"}, {"role": "assistant", "content": "_"}],
            tokenize=True,
            add_special_tokens=False,
        )
        found = any(
            sample_ids[i : i + len(template_ids)] == template_ids
            for i in range(max(len(sample_ids) - len(template_ids) + 1, 0))
        )
        if not found:
            raise ValueError(
                f"Response template {response_template!r} encodes to {template_ids}, "
                f"but this sequence was not found in a tokenized sample conversation. "
                f"Standalone encoding doesn't match in-context encoding — loss masking "
                f"will not work correctly."
            )
    except (AttributeError, TypeError):
        pass  # apply_chat_template not available for this tokenizer

    return template_ids


class CompletionOnlyMaskingCollator:
    """TRL-version-agnostic assistant-only loss masking collator."""

    def __init__(self, tokenizer, response_template: str):
        from transformers import DataCollatorForLanguageModeling

        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.response_token_ids = _build_response_template_ids(
            tokenizer, response_template
        )
        # ChatML conversation boundary marker.
        self.end_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def __call__(self, features):
        filtered = []
        for feature in features:
            filtered.append(
                {
                    k: v
                    for k, v in feature.items()
                    if k in {"input_ids", "attention_mask", "labels", "token_type_ids"}
                }
            )

        batch = self.base_collator(filtered)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        for row in range(input_ids.shape[0]):
            row_ids = input_ids[row].tolist()
            row_attention_mask = (
                attention_mask[row].tolist() if attention_mask is not None else None
            )
            masked = _mask_assistant_spans(
                row_ids,
                self.response_token_ids,
                self.end_token_ids,
                attention_mask=row_attention_mask,
            )
            labels[row] = labels[row].new_tensor(masked)
        batch["labels"] = labels
        return batch


def _build_sft_processing_kwargs(tokenizer, sft_init_params: dict) -> dict:
    """Build tokenizer-related kwargs across TRL API variants."""
    if "processing_class" in sft_init_params:
        return {"processing_class": tokenizer}
    if "tokenizer" in sft_init_params:
        return {"tokenizer": tokenizer}
    raise TypeError(
        "Unsupported TRL SFTTrainer signature: missing both "
        "'processing_class' and 'tokenizer' parameters."
    )


def _build_sft_args(train_cfg: dict, eval_dataset, max_seq_length: int):
    """Build training args compatible with old/new TRL versions."""
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

    # New TRL versions prefer SFTConfig and replace max_seq_length with max_length.
    if not _trl_supports_legacy_completion_collator() and SFTConfig is not None:
        training_kwargs["max_length"] = max_seq_length
        training_kwargs["packing"] = False
        return SFTConfig(**training_kwargs)

    from transformers import TrainingArguments

    return TrainingArguments(**training_kwargs)


def run_sft_training(
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

    # --- Tokenizer (required for SFT chat template) ---
    tokenizer_name = train_cfg["tokenizer_name"]
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _check_vocab_size(tokenizer, model_cfg, model_config_path)

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

    # --- Loss masking setup: support both old/new TRL APIs ---
    trainer_kwargs = {}
    if _trl_supports_legacy_completion_collator():
        # Legacy API: explicit completion-only collator.
        trainer_kwargs["data_collator"] = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n",
            tokenizer=tokenizer,
        )
    else:
        # New TRL API removed DataCollatorForCompletionOnlyLM; keep
        # assistant-only masking behavior with a local collator.
        trainer_kwargs["data_collator"] = CompletionOnlyMaskingCollator(
            tokenizer=tokenizer,
            response_template="<|im_start|>assistant\n",
        )

    training_args = _build_sft_args(train_cfg, eval_dataset, max_seq_length)

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
    sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "max_seq_length" in sft_init_params:
        trainer_kwargs["max_seq_length"] = max_seq_length

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **_build_sft_processing_kwargs(tokenizer, sft_init_params),
        callbacks=[RichLoggingCallback(tokens_per_step)],
        **trainer_kwargs,
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
