#!/usr/bin/env python3
"""Run the isolated local SFT smoke-test training flow."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.flow_validation import validate_local_sft_config
from lite_llm.sft_runner import main


if __name__ == "__main__":
    main(
        default_train_config="configs/local/sft_train.yaml",
        default_model_config="configs/local/sft_model.yaml",
        description="Run Lite-LLM local SFT smoke-test training",
        validate_fn=validate_local_sft_config,
    )
