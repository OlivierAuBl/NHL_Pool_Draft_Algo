from __future__ import annotations

from pathlib import Path
from statistics import fmean

import pandas as pd

from nhl_draft_lab.draft.engine import run_draft
from nhl_draft_lab.models import RosterConfig
from nhl_draft_lab.projection_backtest import _ProjectionViewStrategy
from nhl_draft_lab.projection_source_experiment import (
    DEFAULT_TRIM_MIN_SOURCES,
    _assets_for_experiment,
    _competition_rank,
    prepare_projection_experiment,
)
from nhl_draft_lab.strategies.factory import build_strategy
from nhl_draft_lab.strategies.team_vorp import TeamAwareVorpStrategy
from nhl_draft_lab.team_tiebreak_experiment import (
    _selected_methods,
    _team_concentration,
)


ALL_RAW = "all_raw"
FOCAL_STACKS = "focal_stack_vs_raw_field"
FIELD_STACKS = "focal_raw_vs_stack_field"
FIELD_REGIMES = (ALL_RAW, FOCAL_STACKS, FIELD_STACKS)


def run_team_field_drafts(
    prepared,
    *,
    gm_count: int,
    roster_config: RosterConfig,
    tolerance: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = _assets_for_experiment(prepared)
    details: list[dict[str, object]] = []
    picks: list[dict[str, object]] = []

    for method in _selected_methods(prepared):
        for regime in FIELD_REGIMES:
            for focal_slot in range(1, gm_count + 1):
                if regime == FIELD_STACKS:
                    strategies = {
                        gm_id: TeamAwareVorpStrategy("stack", tolerance)
                        for gm_id in range(1, gm_count + 1)
                    }
                    focal_base = build_strategy("vorp")
                else:
                    strategies = {
                        gm_id: build_strategy("vorp")
                        for gm_id in range(1, gm_count + 1)
                    }
                    focal_base = (
                        TeamAwareVorpStrategy("stack", tolerance)
                        if regime == FOCAL_STACKS
                        else build_strategy("vorp")
                    )
                strategies[focal_slot] = _ProjectionViewStrategy(
                    focal_base,
                    value_column=method.value_column,
                    gm_count=gm_count,
                )
                result = run_draft(assets, gm_count, roster_config, strategies)
                actual_totals = {
                    gm_id: float(
                        sum(asset.metadata["actual_points"] for asset in roster)
                    )
                    for gm_id, roster in result.rosters.items()
                }
                focal_roster = result.rosters[focal_slot]
                field = [
                    points
                    for gm_id, points in actual_totals.items()
                    if gm_id != focal_slot
                ]
                field_concentration = [
                    _team_concentration(roster)
                    for gm_id, roster in result.rosters.items()
                    if gm_id != focal_slot
                ]
                focal_points = actual_totals[focal_slot]
                details.append(
                    {
                        "projection_method": method.name,
                        "projection_type": method.projection_type,
                        "draft_slot": focal_slot,
                        "field_regime": regime,
                        "vor_tolerance": tolerance,
                        "final_points": focal_points,
                        "final_rank": _competition_rank(actual_totals, focal_slot),
                        "gap_to_winner": max(actual_totals.values()) - focal_points,
                        "margin_vs_field": focal_points - fmean(field),
                        **_team_concentration(focal_roster),
                        "field_mean_max_players_same_team": fmean(
                            item["max_players_same_team"] for item in field_concentration
                        ),
                        "field_mean_unique_player_teams": fmean(
                            item["unique_player_teams"] for item in field_concentration
                        ),
                    }
                )
                for pick in result.picks:
                    if pick.gm_id != focal_slot:
                        continue
                    picks.append(
                        {
                            "projection_method": method.name,
                            "projection_type": method.projection_type,
                            "draft_slot": focal_slot,
                            "field_regime": regime,
                            "overall_pick": pick.overall_pick,
                            "round_no": pick.round_no,
                            "category": pick.asset.category,
                            "entity_id": pick.asset.entity_id,
                            "name": pick.asset.name,
                            "nhl_team": pick.asset.nhl_team,
                            "actual_points": pick.asset.metadata["actual_points"],
                        }
                    )
    return (
        pd.DataFrame(details).sort_values(
            ["projection_method", "field_regime", "draft_slot"]
        ).reset_index(drop=True),
        pd.DataFrame(picks).sort_values(
            ["projection_method", "field_regime", "draft_slot", "overall_pick"]
        ).reset_index(drop=True),
    )


def summarize_team_field(details: pd.DataFrame) -> pd.DataFrame:
    work = details.assign(
        win=details["final_rank"].eq(1).astype(float),
        top_3=details["final_rank"].le(3).astype(float),
    )
    return (
        work.groupby(
            ["projection_method", "projection_type", "field_regime"],
            as_index=False,
        )
        .agg(
            simulations=("draft_slot", "size"),
            mean_final_points=("final_points", "mean"),
            mean_rank=("final_rank", "mean"),
            win_rate=("win", "mean"),
            top_3_rate=("top_3", "mean"),
            mean_gap_to_winner=("gap_to_winner", "mean"),
            mean_margin_vs_field=("margin_vs_field", "mean"),
            focal_mean_max_team=("max_players_same_team", "mean"),
            focal_mean_unique_teams=("unique_player_teams", "mean"),
            field_mean_max_team=("field_mean_max_players_same_team", "mean"),
            field_mean_unique_teams=("field_mean_unique_player_teams", "mean"),
        )
        .sort_values(["projection_method", "mean_rank", "mean_final_points"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def paired_field_deltas(details: pd.DataFrame) -> pd.DataFrame:
    keys = ["projection_method", "projection_type", "draft_slot"]
    metrics = [
        "final_points",
        "final_rank",
        "margin_vs_field",
        "max_players_same_team",
        "unique_player_teams",
        "field_mean_max_players_same_team",
        "field_mean_unique_player_teams",
    ]
    baseline = details[details["field_regime"].eq(ALL_RAW)][keys + metrics].rename(
        columns={metric: f"raw_{metric}" for metric in metrics}
    )
    variants = details[~details["field_regime"].eq(ALL_RAW)].copy()
    paired = variants.merge(baseline, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        paired[f"{metric}_delta_vs_all_raw"] = paired[metric] - paired[f"raw_{metric}"]
    return paired.sort_values(
        ["field_regime", "projection_method", "draft_slot"]
    ).reset_index(drop=True)


def summarize_paired_field(paired: pd.DataFrame) -> pd.DataFrame:
    work = paired.assign(
        points_improved=paired["final_points_delta_vs_all_raw"].gt(0).astype(int),
        points_worsened=paired["final_points_delta_vs_all_raw"].lt(0).astype(int),
        rank_improved=paired["final_rank_delta_vs_all_raw"].lt(0).astype(int),
        rank_worsened=paired["final_rank_delta_vs_all_raw"].gt(0).astype(int),
    )
    return (
        work.groupby(
            ["projection_method", "projection_type", "field_regime"],
            as_index=False,
        )
        .agg(
            simulations=("draft_slot", "size"),
            mean_points_delta=("final_points_delta_vs_all_raw", "mean"),
            mean_rank_delta=("final_rank_delta_vs_all_raw", "mean"),
            mean_margin_delta=("margin_vs_field_delta_vs_all_raw", "mean"),
            points_improved=("points_improved", "sum"),
            points_worsened=("points_worsened", "sum"),
            rank_improved=("rank_improved", "sum"),
            rank_worsened=("rank_worsened", "sum"),
            focal_max_team_delta=("max_players_same_team_delta_vs_all_raw", "mean"),
            field_max_team_delta=("field_mean_max_players_same_team_delta_vs_all_raw", "mean"),
        )
        .sort_values(["field_regime", "mean_rank_delta"])
        .reset_index(drop=True)
    )


def _report(
    season_label: str,
    tolerance: float,
    summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> str:
    table = paired_summary[[
        "projection_method",
        "projection_type",
        "field_regime",
        "mean_points_delta",
        "mean_rank_delta",
        "mean_margin_delta",
        "points_improved",
        "points_worsened",
        "focal_max_team_delta",
        "field_max_team_delta",
    ]].copy()
    for column in [
        "mean_points_delta",
        "mean_rank_delta",
        "mean_margin_delta",
        "focal_max_team_delta",
        "field_max_team_delta",
    ]:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    aggregate = (
        paired_summary.groupby("field_regime", as_index=False)
        .agg(
            mean_points_delta=("mean_points_delta", "mean"),
            mean_rank_delta=("mean_rank_delta", "mean"),
            mean_margin_delta=("mean_margin_delta", "mean"),
            focal_max_team_delta=("focal_max_team_delta", "mean"),
            field_max_team_delta=("field_max_team_delta", "mean"),
        )
    )
    conclusions = [
        f"- `{row.field_regime}`: {row.mean_points_delta:+.3f} points, "
        f"{row.mean_rank_delta:+.3f} rang et {row.mean_margin_delta:+.3f} de marge "
        f"versus le scénario où tous utilisent le VOR brut."
        for row in aggregate.itertuples(index=False)
    ]
    return "\n".join(
        [
            f"# Inversion du terrain concentré — {season_label}",
            "",
            f"Fenêtre de départage VOR: {tolerance:g} points.",
            "",
            "- `all_raw`: tous les GMs utilisent le VOR brut.",
            "- `focal_stack_vs_raw_field`: seul le GM testé concentre ses équipes.",
            "- `focal_raw_vs_stack_field`: le GM testé reste normal et les 15 adversaires concentrent.",
            "",
            "## Deltas appariés contre `all_raw`",
            "",
            table.to_markdown(index=False),
            "",
            "## Lecture globale",
            "",
            *conclusions,
            "",
            "Un delta de rang négatif est une amélioration. Les résultats sont appariés par méthode de projection et slot de draft.",
        ]
    )


def run_team_field_experiment(
    *,
    master_path: Path,
    output_dir: Path,
    season_label: str,
    gm_count: int = 16,
    roster_config: RosterConfig = RosterConfig({"F": 10, "D": 3, "G": 2, "T": 1}),
    tolerance: float = 3.0,
    trim_min_sources: int = DEFAULT_TRIM_MIN_SOURCES,
) -> dict[str, pd.DataFrame]:
    master = pd.read_csv(master_path, low_memory=False)
    prepared = prepare_projection_experiment(
        master,
        gm_count=gm_count,
        roster_config=roster_config,
        opponent_source="PoolPro",
        trim_min_sources=trim_min_sources,
    )
    details, picks = run_team_field_drafts(
        prepared,
        gm_count=gm_count,
        roster_config=roster_config,
        tolerance=tolerance,
    )
    summary = summarize_team_field(details)
    paired = paired_field_deltas(details)
    paired_summary = summarize_paired_field(paired)
    outputs = {
        "details": details,
        "summary": summary,
        "paired": paired,
        "paired_summary": paired_summary,
        "picks": picks,
    }
    filenames = {
        "details": "01_team_field_details.csv",
        "summary": "02_team_field_summary.csv",
        "paired": "03_team_field_vs_all_raw_by_slot.csv",
        "paired_summary": "04_team_field_vs_all_raw_summary.csv",
        "picks": "05_team_field_focal_picks.csv",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in outputs.items():
        frame.to_csv(output_dir / filenames[key], index=False)
    (output_dir / "06_team_field_report.md").write_text(
        _report(season_label, tolerance, summary, paired_summary), encoding="utf-8"
    )
    return outputs
