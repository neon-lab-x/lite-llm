#!/usr/bin/env python3
"""Legacy entrypoint kept only to redirect to explicit flows."""


if __name__ == "__main__":
    raise SystemExit(
        "Data preparation flow is now split. Use `scripts/local/prepare_data.py` "
        "for local smoke data or `scripts/production/prepare_data.py` for server-scale data."
    )
