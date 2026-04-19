#!/usr/bin/env python3
"""Run the isolated production SFT training flow."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.flow_validation import validate_production_sft_config
from lite_llm.sft_runner import main


if __name__ == "__main__":
    main(
        default_train_config="configs/production/sft_train.yaml",
        default_model_config="configs/production/model.yaml",
        description="Run Lite-LLM production SFT training",
        validate_fn=validate_production_sft_config,
    )
