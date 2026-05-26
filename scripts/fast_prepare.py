#!/usr/bin/env python3
"""Deprecated data-prep entrypoint.

The old fast_prepare.py used streaming dataset order and category-level targets,
which made partial runs prone to source-order bias. Use the production data
preparation script instead; it now has deterministic shuffled sampling, file
level resume, and disk budget guards.
"""

raise SystemExit(
    "scripts/fast_prepare.py is deprecated. Use "
    "`uv run python scripts/production/prepare_data.py --local-only` "
    "for the default zh_first_v1_3b recipe, or pass "
    "`--datasets-config configs/production/datasets_zh_first_20b.yaml` "
    "for the formal run."
)
