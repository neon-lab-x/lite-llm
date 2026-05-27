#!/usr/bin/env python3
"""Run the isolated local SFT smoke flow."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sft.flow_validation import validate_local_sft_config
from sft.runner import main


if __name__ == "__main__":
    main(
        default_train_config="sft/configs/local/train.yaml",
        default_model_config="sft/configs/local/model.yaml",
        description="Run Lite-LLM local SFT smoke training",
        validate_fn=validate_local_sft_config,
    )

