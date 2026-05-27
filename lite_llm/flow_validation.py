import os


def _require_prefix(path: str, prefix: str, label: str):
    resolved = os.path.realpath(path)
    prefix_resolved = os.path.realpath(prefix)
    try:
        common = os.path.commonpath([resolved, prefix_resolved])
    except ValueError:
        common = ""
    if common != prefix_resolved:
        raise ValueError(f"{label} must stay under {prefix}, got {path}")



def validate_local_train_config(train_cfg: dict, model_cfg: dict):
    if train_cfg.get("use_cpu") is not True:
        raise ValueError("Local flow must run with use_cpu=true.")
    if train_cfg.get("tokenizer_name"):
        raise ValueError("Local flow must not require a remote tokenizer.")
    if train_cfg.get("deepspeed"):
        raise ValueError("Local flow must not enable DeepSpeed.")
    if train_cfg.get("resume_from_last_checkpoint") not in (False, None):
        raise ValueError("Local flow should start clean and not auto-resume checkpoints.")
    _require_prefix(train_cfg["data_dir"], "./data/local_smoke/", "Local data_dir")
    _require_prefix(train_cfg["output_dir"], "./artifacts/local/", "Local output_dir")
    if "logging_dir" in train_cfg:
        _require_prefix(train_cfg["logging_dir"], "./artifacts/local/", "Local logging_dir")
    if model_cfg["vocab_size"] > 4096:
        raise ValueError("Local flow should keep a tiny vocab for smoke testing.")


def validate_production_train_config(train_cfg: dict, model_cfg: dict):
    if train_cfg.get("use_cpu", False):
        raise ValueError("Production flow must not be pinned to CPU.")
    if not train_cfg.get("tokenizer_name"):
        raise ValueError("Production flow requires tokenizer_name.")
    if not train_cfg.get("deepspeed"):
        raise ValueError("Production flow requires a DeepSpeed config.")
    if train_cfg.get("resume_from_last_checkpoint") is not True:
        raise ValueError("Production flow should auto-resume from the latest checkpoint.")
    _require_prefix(
        train_cfg["data_dir"],
        "./data/production/",
        "Production data_dir",
    )
    _require_prefix(train_cfg["output_dir"], "./artifacts/production/", "Production output_dir")
    if "logging_dir" in train_cfg:
        _require_prefix(
            train_cfg["logging_dir"],
            "./artifacts/production/",
            "Production logging_dir",
        )
    if model_cfg["vocab_size"] < 10000:
        raise ValueError("Production flow should use the full tokenizer/model vocab.")
