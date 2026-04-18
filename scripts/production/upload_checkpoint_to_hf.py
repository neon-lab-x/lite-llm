#!/usr/bin/env python3
"""Periodically upload the latest training checkpoint to a HuggingFace Model repo.

Run this script alongside training in a separate terminal:

    python scripts/production/upload_checkpoint_to_hf.py \
        --repo username/my-model-checkpoints \
        --interval 3600

The script polls the checkpoint output directory once per interval, finds the
latest complete checkpoint, and uploads it to the specified HF model repo.
Already-uploaded checkpoints are skipped so repeated polls are idempotent.

Authentication: set the HF_TOKEN environment variable or pass --token.

Options
-------
--repo          HF model repo id (required), e.g. "NoBey/lite-llm-ckpt"
--output-dir    Local checkpoint directory (default: read from train.yaml)
--config        Path to train.yaml (default: configs/production/train.yaml)
--token         HF write token (default: $HF_TOKEN env var)
--interval      Seconds between upload attempts (default: 3600)
--once          Upload once and exit instead of looping
--private       Create the repo as private if it doesn't exist yet
"""

import argparse
import datetime
import os
import re
import sys
import time

import yaml
from huggingface_hub import HfApi


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
DEFAULT_CONFIG = "configs/production/train.yaml"
DEFAULT_INTERVAL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_output_dir_from_config(config_path: str) -> str:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["output_dir"]


def _find_latest_checkpoint(output_dir: str):
    """Return (step, abs_path) for the highest-numbered complete checkpoint, or None."""
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for entry in os.listdir(output_dir):
        m = CHECKPOINT_PATTERN.fullmatch(entry)
        if not m:
            continue
        ckpt_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(ckpt_dir):
            continue
        # A checkpoint is considered complete when trainer_state.json exists.
        if not os.path.exists(os.path.join(ckpt_dir, "trainer_state.json")):
            continue
        candidates.append((int(m.group(1)), ckpt_dir))

    if not candidates:
        return None

    return max(candidates, key=lambda t: t[0])


def _ensure_repo(api: HfApi, repo_id: str, private: bool):
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"[upload] Repo {repo_id!r} found.")
    except Exception:
        print(f"[upload] Creating {'private' if private else 'public'} model repo {repo_id!r} ...")
        api.create_repo(repo_id=repo_id, repo_type="model", private=private)
        print("[upload] Repo created.")


def _upload(api: HfApi, repo_id: str, checkpoint_path: str, step: int) -> bool:
    path_in_repo = f"checkpoint-{step}"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[upload] {ts}  uploading step {step:,}  →  {repo_id}/{path_in_repo}")
    try:
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=f"checkpoint step {step:,}",
        )
        print(f"[upload] Done  (step {step:,})")
        return True
    except Exception as exc:
        print(f"[upload] Upload failed: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Periodically upload training checkpoints to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="USER/REPO",
        help="HuggingFace model repo id to upload into.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local checkpoint directory (overrides train.yaml output_dir).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to train.yaml (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace write token (default: $HF_TOKEN env var).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        metavar="SECONDS",
        help=f"Seconds between upload attempts (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Upload once and exit instead of looping.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    args = parser.parse_args()

    # Resolve output_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _load_output_dir_from_config(args.config)
    output_dir = os.path.abspath(output_dir)
    print(f"[upload] Watching checkpoint dir: {output_dir}")
    print(f"[upload] Target repo:             {args.repo}")
    print(f"[upload] Upload interval:         {args.interval}s ({args.interval / 3600:.2f}h)")

    # Auth
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[upload] Error: provide --token or set the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)
    _ensure_repo(api, args.repo, private=args.private)

    last_uploaded_step: int = -1

    while True:
        result = _find_latest_checkpoint(output_dir)

        if result is None:
            print(f"[upload] No complete checkpoint found in {output_dir}, will retry.")
        else:
            step, ckpt_path = result
            if step == last_uploaded_step:
                print(f"[upload] Step {step:,} already uploaded, skipping.")
            else:
                success = _upload(api, args.repo, ckpt_path, step)
                if success:
                    last_uploaded_step = step

        if args.once:
            break

        next_at = datetime.datetime.now() + datetime.timedelta(seconds=args.interval)
        print(f"[upload] Next attempt at {next_at.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
