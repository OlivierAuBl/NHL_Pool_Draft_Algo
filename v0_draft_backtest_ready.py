from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nhl_draft_lab.draft.engine import run_draft
from nhl_draft_lab.models import DraftAsset, RosterConfig
from nhl_draft_lab.strategies.tier_vorp import TierVorpStrategy
from nhl_draft_lab.strategies.vorp import VorpStrategy
from nhl_draft_lab.tiering import TierConfig

SEASON = 20252026
GM_COUNT = 15
ROSTER = RosterConfig({"F": 10, "D": 3, "G": 2, "T": 1})
TIER_CONFIG = TierConfig(
    relative_width=0.05,
    max_size=5,
    superstar_max_size=3,
    superstar_tiers=2,
)

# NHL rankings tables generally omit players with 0 GP.
# Barkov is a known 0-GP result in 2025-26 and must stay in the evaluation.
ZERO_GAME_ACTUALS = {
    8477493: 0.0,  # Aleksander Barkov
}

TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Carolina Hurricanes": "CAR",
    "Columbus Blue Jackets": "CBJ",
    "Calgary Flames": "CGY",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "New Jersey Devils": "NJD",
    "Nashville Predators": "NSH",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "Seattle Kraken": "SEA",
    "San Jose Sharks": "SJS",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Winnipeg Jets": "WPG",
    "Washington Capitals": "WSH",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    decision_source: str  # projection | actual_same_universe | actual_full_universe
    focal_strategy: str
    opponent_strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 V0: draft using preseason projections, then score the drafted "
            "rosters with actual 2025-26 NHL results."
        )
    )
    parser.add_argument("--db", type=Path, default=Path("data/nhl_history.sqlite"))
    parser.add_argument(
        "--players",
        type=Path,
        default=Path("output/v0_projection_tiers/v0_players_20252026.csv"),
    )
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/v0_draft_backtest"),
    )
    return parser.parse_args()


def strategy(name: str):
    if name == "tier_vorp":
        return TierVorpStrategy(TIER_CONFIG)
    if name == "vorp":
        return VorpStrategy()
    raise ValueError(name)


def load_actual(db_path: Path, season: int) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as con:
        actual = pd.read_sql_query(
            """
            SELECT season_id, category, rank, entity_id, name, nhl_team,
                   games_played, pool_points_raw
            FROM rankings
            WHERE season_id = ?
            ORDER BY category, rank
            """,
            con,
            params=(season,),
        )

    if actual.empty:
        raise ValueError(f"No data for season {season} in {db_path}")

    actual["entity_id"] = pd.to_numeric(actual["entity_id"], errors="coerce").astype("Int64")
    actual["pool_points_raw"] = pd.to_numeric(actual["pool_points_raw"], errors="coerce")
    actual["team_abbrev"] = actual["nhl_team"].fillna("").astype(str).str.strip()

    mask = actual["category"].eq("T") & actual["team_abbrev"].eq("")
    actual.loc[mask, "team_abbrev"] = (
        actual.loc[mask, "name"].map(TEAM_NAME_TO_ABBREV).fillna("")
    )
    return actual


def load_projection_assets(players_path: Path, actual: pd.DataFrame) -> tuple[list[DraftAsset], dict[str, float], pd.DataFrame]:
    if not players_path.exists():
        raise FileNotFoundError(f"Projection file not found: {players_path}")

    df = pd.read_csv(players_path)
    df["projection_median"] = pd.to_numeric(df["projection_median"], errors="coerce")
    df["NHLID"] = pd.to_numeric(df.get("NHLID"), errors="coerce").astype("Int64")

    player_actual = {
        (str(int(r.entity_id)), r.category): float(r.pool_points_raw)
        for r in actual.itertuples()
        if r.category in {"F", "D", "G"} and pd.notna(r.entity_id)
    }
    team_actual = {
        str(r.team_abbrev): float(r.pool_points_raw)
        for r in actual.itertuples()
        if r.category == "T" and str(r.team_abbrev)
    }

    assets: list[DraftAsset] = []
    actual_by_asset_id: dict[str, float] = {}
    audit_rows = []

    for row in df.itertuples(index=False):
        category = str(row.category)
        projection = float(row.projection_median)

        if category == "T":
            team = str(row.Team)
            asset_id = f"T:{team}"
            actual_points = team_actual.get(team)
            name = team
            nhl_team = team
        else:
            if pd.isna(row.NHLID):
                continue
            nhlid = int(row.NHLID)
            asset_id = f"P:{nhlid}"
            actual_points = player_actual.get((str(nhlid), category))
            if actual_points is None and nhlid in ZERO_GAME_ACTUALS:
                actual_points = ZERO_GAME_ACTUALS[nhlid]
            name = str(row.FullName)
            nhl_team = str(row.Team)

        if actual_points is None:
            audit_rows.append({
                "asset_id": asset_id,
                "name": name,
                "category": category,
                "projection": projection,
                "actual_points": np.nan,
                "status": "UNMATCHED",
            })
            continue

        assets.append(
            DraftAsset(
                entity_id=asset_id,
                name=name,
                category=category,
                value=projection,
                nhl_team=nhl_team,
                metadata={"actual_points": float(actual_points)},
            )
        )
        actual_by_asset_id[asset_id] = float(actual_points)
        audit_rows.append({
            "asset_id": asset_id,
            "name": name,
            "category": category,
            "projection": projection,
            "actual_points": float(actual_points),
            "status": "MATCHED",
        })

    audit = pd.DataFrame(audit_rows)
    return assets, actual_by_asset_id, audit


def actual_same_universe_assets(projected_assets: list[DraftAsset], actual_by_asset_id: dict[str, float]) -> list[DraftAsset]:
    return [
        DraftAsset(
            entity_id=a.entity_id,
            name=a.name,
            category=a.category,
            value=actual_by_asset_id[a.entity_id],
            nhl_team=a.nhl_team,
        )
        for a in projected_assets
    ]


def actual_full_universe_assets(actual: pd.DataFrame) -> list[DraftAsset]:
    assets = []
    for row in actual.itertuples(index=False):
        category = str(row.category)
        if category == "T":
            asset_id = f"T:{row.team_abbrev}"
            name = str(row.team_abbrev) if row.team_abbrev else str(row.name)
            team = str(row.team_abbrev)
        else:
            if pd.isna(row.entity_id):
                continue
            asset_id = f"P:{int(row.entity_id)}"
            name = str(row.name)
            team = str(row.nhl_team)

        assets.append(
            DraftAsset(
                entity_id=asset_id,
                name=name,
                category=category,
                value=float(row.pool_points_raw),
                nhl_team=team,
            )
        )
    return assets


def actual_score_rosters(result, actual_by_asset_id: dict[str, float] | None = None) -> dict[int, float]:
    totals = {}
    for gm_id, roster in result.rosters.items():
        if actual_by_asset_id is None:
            totals[gm_id] = float(sum(asset.value for asset in roster))
        else:
            totals[gm_id] = float(sum(actual_by_asset_id[asset.entity_id] for asset in roster))
    return totals


def competition_rank(totals: dict[int, float], gm_id: int) -> int:
    focal = totals[gm_id]
    return 1 + sum(score > focal for other, score in totals.items() if other != gm_id)


def run_scenario(
    scenario: Scenario,
    projected_assets: list[DraftAsset],
    actual_same_assets: list[DraftAsset],
    actual_full_assets: list[DraftAsset],
    projected_actual_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario.decision_source == "projection":
        assets = projected_assets
        actual_map = projected_actual_map
    elif scenario.decision_source == "actual_same_universe":
        assets = actual_same_assets
        actual_map = None
    elif scenario.decision_source == "actual_full_universe":
        assets = actual_full_assets
        actual_map = None
    else:
        raise ValueError(scenario.decision_source)

    rows = []
    pick_rows = []

    for focal_slot in range(1, GM_COUNT + 1):
        strategies = {
            gm_id: strategy(
                scenario.focal_strategy if gm_id == focal_slot else scenario.opponent_strategy
            )
            for gm_id in range(1, GM_COUNT + 1)
        }
        result = run_draft(assets, GM_COUNT, ROSTER, strategies)
        totals = actual_score_rosters(result, actual_map)
        focal_total = totals[focal_slot]
        winner = max(totals.values())
        field = [v for gm, v in totals.items() if gm != focal_slot]

        rows.append({
            "scenario": scenario.name,
            "decision_source": scenario.decision_source,
            "focal_strategy": scenario.focal_strategy,
            "opponent_strategy": scenario.opponent_strategy,
            "draft_slot": focal_slot,
            "actual_points": focal_total,
            "actual_rank": competition_rank(totals, focal_slot),
            "winner_points": winner,
            "gap_to_winner": winner - focal_total,
            "margin_vs_field_mean": focal_total - float(np.mean(field)),
        })

        for pick in result.picks:
            if pick.gm_id != focal_slot:
                continue
            if actual_map is None:
                actual_points = pick.asset.value
            else:
                actual_points = actual_map[pick.asset.entity_id]
            pick_rows.append({
                "scenario": scenario.name,
                "draft_slot": focal_slot,
                "overall_pick": pick.overall_pick,
                "round_no": pick.round_no,
                "category": pick.asset.category,
                "asset_id": pick.asset.entity_id,
                "name": pick.asset.name,
                "decision_value": pick.asset.value,
                "actual_points": actual_points,
            })

    return pd.DataFrame(rows), pd.DataFrame(pick_rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, part in details.groupby("scenario", sort=False):
        rows.append({
            "scenario": scenario,
            "trials": len(part),
            "mean_actual_points": part["actual_points"].mean(),
            "median_actual_points": part["actual_points"].median(),
            "mean_rank": part["actual_rank"].mean(),
            "win_rate": (part["actual_rank"] == 1).mean(),
            "top3_rate": (part["actual_rank"] <= 3).mean(),
            "mean_gap_to_winner": part["gap_to_winner"].mean(),
            "mean_margin_vs_field": part["margin_vs_field_mean"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual = load_actual(args.db, args.season)
    projected_assets, actual_map, audit = load_projection_assets(args.players, actual)
    actual_same = actual_same_universe_assets(projected_assets, actual_map)
    actual_full = actual_full_universe_assets(actual)

    counts = pd.Series([a.category for a in projected_assets]).value_counts().to_dict()
    for category, required in {k: GM_COUNT * v for k, v in ROSTER.counts.items()}.items():
        if counts.get(category, 0) < required:
            raise ValueError(
                f"Projection universe has only {counts.get(category, 0)} {category}; need {required}"
            )

    scenarios = [
        Scenario("v0_tier_vorp", "projection", "tier_vorp", "tier_vorp"),
        Scenario("v0_vorp_vs_tier_opponents", "projection", "vorp", "tier_vorp"),
        Scenario("perfect_info_tier_vorp_same_universe", "actual_same_universe", "tier_vorp", "tier_vorp"),
        Scenario("perfect_info_vorp_vs_tier_same_universe", "actual_same_universe", "vorp", "tier_vorp"),
        Scenario("perfect_info_tier_vorp_full_universe", "actual_full_universe", "tier_vorp", "tier_vorp"),
        Scenario("perfect_info_vorp_vs_tier_full_universe", "actual_full_universe", "vorp", "tier_vorp"),
    ]

    detail_frames = []
    pick_frames = []
    for scenario in scenarios:
        details, picks = run_scenario(
            scenario,
            projected_assets,
            actual_same,
            actual_full,
            actual_map,
        )
        detail_frames.append(details)
        pick_frames.append(picks)

    details = pd.concat(detail_frames, ignore_index=True)
    picks = pd.concat(pick_frames, ignore_index=True)
    summary = summarize(details)

    summary_path = args.output_dir / "v0_draft_backtest_summary.csv"
    details_path = args.output_dir / "v0_draft_backtest_by_slot.csv"
    picks_path = args.output_dir / "v0_draft_backtest_picks.csv"
    audit_path = args.output_dir / "v0_projection_actual_match_audit.csv"

    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    picks.to_csv(picks_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("Stage 2 V0 — projection-only draft backtest")
    print(f"Season: {args.season}")
    print(f"GM: {GM_COUNT} | roster: {dict(ROSTER.counts)}")
    print(f"Projection universe: {counts}")
    print(f"Matched assets: {(audit['status'] == 'MATCHED').sum()} | unmatched: {(audit['status'] == 'UNMATCHED').sum()}")
    print()
    print("Summary (actual 2025-26 points)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    compare = details.loc[
        details["scenario"].isin(["v0_tier_vorp", "v0_vorp_vs_tier_opponents"]),
        [
            "scenario", "draft_slot", "actual_points", "actual_rank",
            "gap_to_winner", "margin_vs_field_mean",
        ],
    ]
    print("V0 by draft slot")
    print(compare.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print("Files created:")
    for path in [summary_path, details_path, picks_path, audit_path]:
        print(path)


if __name__ == "__main__":
    main()
