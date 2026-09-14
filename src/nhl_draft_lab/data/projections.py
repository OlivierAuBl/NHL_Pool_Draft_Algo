from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nhl_draft_lab.models import DraftAsset


@dataclass(frozen=True)
class Projection:
    asset: DraftAsset
    mean: float
    stddev: float


def load_projection_csv(path: Path) -> list[Projection]:
    frame = pd.read_csv(path)
    required = {"entity_id", "name", "category", "projected_points", "stddev_points"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Projection CSV is missing columns: {sorted(missing)}")

    projections: list[Projection] = []
    for row in frame.to_dict("records"):
        mean = float(row["projected_points"])
        stddev = max(0.0, float(row["stddev_points"]))
        asset = DraftAsset(
            entity_id=str(row["entity_id"]),
            name=str(row["name"]),
            category=str(row["category"]).upper(),
            value=mean,
            nhl_team=str(row.get("nhl_team") or ""),
            metadata=row,
        )
        projections.append(Projection(asset=asset, mean=mean, stddev=stddev))
    return projections
