from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from nhl_draft_lab.models import DraftAsset


def write_rankings(db_path: Path, rankings: pd.DataFrame, replace_season: bool = True) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        if replace_season and not rankings.empty:
            seasons = sorted({int(x) for x in rankings["season_id"].unique()})
            table_exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rankings'"
            ).fetchone()
            if table_exists:
                for season in seasons:
                    con.execute("DELETE FROM rankings WHERE season_id = ?", (season,))
        rankings.to_sql("rankings", con, if_exists="append", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_rankings_season_category ON rankings(season_id, category)")


def available_seasons(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT DISTINCT season_id FROM rankings ORDER BY season_id").fetchall()
    return [int(row[0]) for row in rows]


def load_assets(db_path: Path, season_id: int, value_column: str = "pool_points_84") -> list[DraftAsset]:
    with sqlite3.connect(db_path) as con:
        frame = pd.read_sql_query(
            f"SELECT * FROM rankings WHERE season_id = ? ORDER BY category, rank",
            con,
            params=(season_id,),
        )
    if frame.empty:
        raise ValueError(f"No rankings found for season {season_id} in {db_path}")
    if value_column not in frame.columns:
        raise ValueError(f"Missing value column {value_column!r}; available: {sorted(frame.columns)}")

    assets: list[DraftAsset] = []
    for row in frame.to_dict("records"):
        assets.append(DraftAsset(
            entity_id=str(row["entity_id"]),
            name=str(row["name"]),
            category=str(row["category"]),
            value=float(row[value_column]),
            nhl_team=str(row.get("nhl_team") or ""),
            metadata=row,
        ))
    return assets
