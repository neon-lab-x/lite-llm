#!/usr/bin/env python3
"""Run the isolated production SFT flow."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sft.flow_validation import validate_production_sft_config
from sft.runner import main


if __name__ == "__main__":
    main(
        default_train_config="sft/configs/production/train_zh_first_3b.yaml",
        default_model_config="configs/production/model.yaml",
        description="Run Lite-LLM production SFT training",
        validate_fn=validate_production_sft_config,
    )

