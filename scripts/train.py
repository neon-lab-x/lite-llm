#!/usr/bin/env python3
"""Legacy entrypoint kept only to redirect to explicit flows."""


if __name__ == "__main__":
    raise SystemExit(
        "Training flow is now split. Use `scripts/local/train.py` for local smoke tests "
        "or `scripts/production/train.py` for server training."
    )
