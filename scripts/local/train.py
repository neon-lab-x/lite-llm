#!/usr/bin/env python3
"""Run the isolated local smoke-test training flow."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from lite_llm.flow_validation import validate_local_train_config
from lite_llm.train_runner import main


if __name__ == "__main__":
    main(
        default_train_config="configs/local/train.yaml",
        default_model_config="configs/local/model.yaml",
        description="Run Lite-LLM local smoke-test training",
        validate_fn=validate_local_train_config,
    )
