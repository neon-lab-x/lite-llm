"""Isolation checks for the parallel SFT flow."""

from __future__ import annotations

import os


def _require_prefix(path: str, prefix: str, label: str) -> None:
    resolved = os.path.realpath(path)
    prefix_resolved = os.path.realpath(prefix)
    try:
        common = os.path.commonpath([resolved, prefix_resolved])
    except ValueError:
        common = ""
    if common != prefix_resolved:
        raise ValueError(f"{label} must stay under {prefix}, got {path}")


def _require_one_prefix(path: str, prefixes: list[str], label: str) -> None:
    errors = []
    for prefix in prefixes:
        try:
            _require_prefix(path, prefix, label)
            return
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(f"{label} must stay under one of {prefixes}, got {path}")


def _require_sft_common(train_cfg: dict) -> None:
    if not train_cfg.get("pretrained_model_path"):
        raise ValueError("SFT flow requires pretrained_model_path.")
    if not train_cfg.get("tokenizer_name") and train_cfg.get("tokenizer_type") != "toy_chatml":
        raise ValueError("SFT flow requires tokenizer_name or tokenizer_type=toy_chatml.")


def validate_local_sft_config(train_cfg: dict, model_cfg: dict) -> None:
    del model_cfg
    _require_sft_common(train_cfg)
    if train_cfg.get("use_cpu") is not True:
        raise ValueError("Local SFT flow must run with use_cpu=true.")
    if train_cfg.get("deepspeed"):
        raise ValueError("Local SFT flow must not enable DeepSpeed.")
    if train_cfg.get("resume_from_last_checkpoint") not in (False, None):
        raise ValueError("Local SFT flow should start clean and not auto-resume checkpoints.")

    _require_prefix(train_cfg["data_dir"], "./data/local_smoke/sft/", "Local SFT data_dir")
    _require_prefix(train_cfg["output_dir"], "./artifacts/local/sft/", "Local SFT output_dir")
    _require_prefix(
        train_cfg["pretrained_model_path"],
        "./artifacts/local/sft/",
        "Local SFT pretrained_model_path",
    )
    if "logging_dir" in train_cfg:
        _require_prefix(train_cfg["logging_dir"], "./artifacts/local/sft/", "Local SFT logging_dir")


def validate_production_sft_config(train_cfg: dict, model_cfg: dict) -> None:
    _require_sft_common(train_cfg)
    if train_cfg.get("use_cpu", False):
        raise ValueError("Production SFT flow must not be pinned to CPU.")
    if not train_cfg.get("tokenizer_name"):
        raise ValueError("Production SFT flow requires tokenizer_name.")
    if not train_cfg.get("deepspeed"):
        raise ValueError("Production SFT flow requires a DeepSpeed config.")
    if train_cfg.get("resume_from_last_checkpoint") is not True:
        raise ValueError("Production SFT flow should auto-resume from the latest checkpoint.")

    _require_one_prefix(
        train_cfg["data_dir"],
        ["./data/production/sft/", "/root/autodl-fs/sft/"],
        "Production SFT data_dir",
    )
    _require_prefix(
        train_cfg["output_dir"],
        "./artifacts/production/sft/",
        "Production SFT output_dir",
    )
    _require_prefix(
        train_cfg["pretrained_model_path"],
        "./artifacts/production/",
        "Production SFT pretrained_model_path",
    )
    if "logging_dir" in train_cfg:
        _require_prefix(
            train_cfg["logging_dir"],
            "./artifacts/production/sft/",
            "Production SFT logging_dir",
        )
    if model_cfg["vocab_size"] < 10000:
        raise ValueError("Production SFT flow should use the full tokenizer/model vocab.")

