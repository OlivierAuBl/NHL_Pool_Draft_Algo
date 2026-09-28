from __future__ import annotations

import argparse
from pathlib import Path

from nhl_draft_lab.historical_projection_master import build_historical_projection_master


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a source-by-source 2024-2025 projection master with realised scoring."
    )
    parser.add_argument(
        "--normalized-db",
        type=Path,
        default=Path("../NHL_Source_Analysis/normalisation/nhl_pool.sqlite"),
    )
    parser.add_argument(
        "--history-db",
        type=Path,
        default=Path("data/nhl_history.sqlite"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/projection_source_master_20242025"),
    )
    args = parser.parse_args()
    master, audit = build_historical_projection_master(
        normalized_db=args.normalized_db.resolve(),
        history_db=args.history_db.resolve(),
        season=20242025,
        output_dir=args.output_dir.resolve(),
    )
    print(f"Master rows: {len(master)}")
    print(master.groupby("category").size().to_string())
    print(f"Identity audit rows: {len(audit)}")
    print(f"Output -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
