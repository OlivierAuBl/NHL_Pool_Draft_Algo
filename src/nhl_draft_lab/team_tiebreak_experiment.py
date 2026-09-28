from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean

import pandas as pd

from nhl_draft_lab.draft.engine import run_draft
from nhl_draft_lab.models import DraftAsset, RosterConfig
from nhl_draft_lab.projection_backtest import _ProjectionViewStrategy
from nhl_draft_lab.projection_source_experiment import (
    DEFAULT_TRIM_MIN_SOURCES,
    PreparedExperiment,
    ProjectionMethod,
    _assets_for_experiment,
    _competition_rank,
    prepare_projection_experiment,
)
from nhl_draft_lab.strategies.factory import build_strategy
from nhl_draft_lab.strategies.team_vorp import TeamAwareVorpStrategy


RAW_VOR = "raw_vor"
TEAM_STACK = "team_stack_within_3"
TEAM_DIVERSIFY = "team_diversify_within_3"
SELECTION_SCENARIOS = (RAW_VOR, TEAM_STACK, TEAM_DIVERSIFY)


def _selected_methods(prepared: PreparedExperiment) -> tuple[ProjectionMethod, ...]:
    return tuple(
        method
        for method in prepared.methods
        if method.projection_type == "single_source"
        or method.name == "trimmed_mean_minmax"
    )


def _focal_strategy(scenario: str, tolerance: float):
    if scenario == RAW_VOR:
        return build_strategy("vorp")
    if scenario == TEAM_STACK:
        return TeamAwareVorpStrategy("stack", tolerance)
    if scenario == TEAM_DIVERSIFY:
        return TeamAwareVorpStrategy("diversify", tolerance)
    raise ValueError(f"Unknown selection scenario {scenario!r}")


def _team_concentration(roster: list[DraftAsset]) -> dict[str, float | int]:
    counts = Counter(
        asset.nhl_team
        for asset in roster
        if asset.category != "T" and asset.nhl_team
    )
    total = sum(counts.values())
    return {
        "unique_player_teams": len(counts),
        "max_players_same_team": max(counts.values(), default=0),
        "same_team_pairs": sum(count * (count - 1) // 2 for count in counts.values()),
        "team_concentration_hhi": (
            sum((count / total) ** 2 for count in counts.values()) if total else 0.0
        ),
    }


def run_team_tiebreak_drafts(
    prepared: PreparedExperiment,
    *,
    gm_count: int,
    roster_config: RosterConfig,
    tolerance: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    assets = _assets_for_experiment(prepared)
    methods = _selected_methods(prepared)
    detail_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []

    for method in methods:
        for scenario in SELECTION_SCENARIOS:
            for focal_slot in range(1, gm_count + 1):
                strategies = {
                    gm_id: build_strategy("vorp")
                    for gm_id in range(1, gm_count + 1)
                }
                strategies[focal_slot] = _ProjectionViewStrategy(
                    _focal_strategy(scenario, tolerance),
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
                focal_points = actual_totals[focal_slot]
                field = [
                    points
                    for gm_id, points in actual_totals.items()
                    if gm_id != focal_slot
                ]
                detail_rows.append(
                    {
                        "draft_slot": focal_slot,
                        "projection_method": method.name,
                        "projection_type": method.projection_type,
                        "selection_scenario": scenario,
                        "vor_tolerance": tolerance,
                        "opponent_projection_method": "PoolPro",
                        "opponent_selection_scenario": RAW_VOR,
                        "final_points": focal_points,
                        "final_rank": _competition_rank(actual_totals, focal_slot),
                        "winner_points": max(actual_totals.values()),
                        "gap_to_winner": max(actual_totals.values()) - focal_points,
                        "field_mean_points": fmean(field),
                        "margin_vs_field": focal_points - fmean(field),
                        "projected_roster_points": float(
                            sum(
                                asset.metadata[method.value_column]
                                for asset in focal_roster
                            )
                        ),
                        **_team_concentration(focal_roster),
                    }
                )
                for pick in result.picks:
                    if pick.gm_id != focal_slot:
                        continue
                    pick_rows.append(
                        {
                            "draft_slot": focal_slot,
                            "projection_method": method.name,
                            "projection_type": method.projection_type,
                            "selection_scenario": scenario,
                            "overall_pick": pick.overall_pick,
                            "round_no": pick.round_no,
                            "category": pick.asset.category,
                            "entity_id": pick.asset.entity_id,
                            "name": pick.asset.name,
                            "nhl_team": pick.asset.nhl_team,
                            "decision_value": pick.asset.metadata[method.value_column],
                            "actual_points": pick.asset.metadata["actual_points"],
                        }
                    )

    details = pd.DataFrame(detail_rows).sort_values(
        ["projection_method", "selection_scenario", "draft_slot"]
    ).reset_index(drop=True)
    picks = pd.DataFrame(pick_rows).sort_values(
        ["projection_method", "selection_scenario", "draft_slot", "overall_pick"]
    ).reset_index(drop=True)
    return details, picks


def summarize_team_tiebreaks(details: pd.DataFrame) -> pd.DataFrame:
    work = details.assign(
        win=details["final_rank"].eq(1).astype(float),
        top_3=details["final_rank"].le(3).astype(float),
    )
    return (
        work.groupby(
            ["projection_method", "projection_type", "selection_scenario"],
            as_index=False,
        )
        .agg(
            simulations=("draft_slot", "size"),
            mean_final_points=("final_points", "mean"),
            median_final_points=("final_points", "median"),
            mean_rank=("final_rank", "mean"),
            median_rank=("final_rank", "median"),
            win_rate=("win", "mean"),
            top_3_rate=("top_3", "mean"),
            mean_gap_to_winner=("gap_to_winner", "mean"),
            mean_margin_vs_field=("margin_vs_field", "mean"),
            mean_unique_player_teams=("unique_player_teams", "mean"),
            mean_max_players_same_team=("max_players_same_team", "mean"),
            mean_same_team_pairs=("same_team_pairs", "mean"),
            mean_team_concentration_hhi=("team_concentration_hhi", "mean"),
        )
        .sort_values(
            ["projection_method", "mean_rank", "mean_final_points"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )


def paired_scenario_deltas(details: pd.DataFrame) -> pd.DataFrame:
    baseline_columns = [
        "projection_method",
        "projection_type",
        "draft_slot",
        "final_points",
        "final_rank",
        "max_players_same_team",
        "unique_player_teams",
        "same_team_pairs",
        "team_concentration_hhi",
    ]
    baseline = details[details["selection_scenario"].eq(RAW_VOR)][
        baseline_columns
    ].rename(
        columns={
            column: f"raw_{column}"
            for column in baseline_columns
            if column not in {"projection_method", "projection_type", "draft_slot"}
        }
    )
    variants = details[~details["selection_scenario"].eq(RAW_VOR)].copy()
    paired = variants.merge(
        baseline,
        on=["projection_method", "projection_type", "draft_slot"],
        how="left",
        validate="many_to_one",
    )
    for column in [
        "final_points",
        "final_rank",
        "max_players_same_team",
        "unique_player_teams",
        "same_team_pairs",
        "team_concentration_hhi",
    ]:
        paired[f"{column}_delta_vs_raw"] = paired[column] - paired[f"raw_{column}"]
    return paired.sort_values(
        ["selection_scenario", "projection_method", "draft_slot"]
    ).reset_index(drop=True)


def summarize_paired_scenarios(paired: pd.DataFrame) -> pd.DataFrame:
    work = paired.assign(
        points_improved=paired["final_points_delta_vs_raw"].gt(0).astype(int),
        points_tied=paired["final_points_delta_vs_raw"].eq(0).astype(int),
        points_worsened=paired["final_points_delta_vs_raw"].lt(0).astype(int),
        rank_improved=paired["final_rank_delta_vs_raw"].lt(0).astype(int),
        rank_worsened=paired["final_rank_delta_vs_raw"].gt(0).astype(int),
    )
    return (
        work.groupby(
            ["projection_method", "projection_type", "selection_scenario"],
            as_index=False,
        )
        .agg(
            simulations=("draft_slot", "size"),
            mean_points_delta_vs_raw=("final_points_delta_vs_raw", "mean"),
            median_points_delta_vs_raw=("final_points_delta_vs_raw", "median"),
            mean_rank_delta_vs_raw=("final_rank_delta_vs_raw", "mean"),
            points_improved=("points_improved", "sum"),
            points_tied=("points_tied", "sum"),
            points_worsened=("points_worsened", "sum"),
            rank_improved=("rank_improved", "sum"),
            rank_worsened=("rank_worsened", "sum"),
            mean_max_team_delta=("max_players_same_team_delta_vs_raw", "mean"),
            mean_unique_teams_delta=("unique_player_teams_delta_vs_raw", "mean"),
            mean_same_team_pairs_delta=("same_team_pairs_delta_vs_raw", "mean"),
            mean_hhi_delta=("team_concentration_hhi_delta_vs_raw", "mean"),
        )
        .sort_values(["selection_scenario", "mean_rank_delta_vs_raw"])
        .reset_index(drop=True)
    )


def _by_slot(details: pd.DataFrame) -> pd.DataFrame:
    work = details.copy()
    work["method_scenario"] = (
        work["projection_method"] + "__" + work["selection_scenario"]
    )
    return (
        work.pivot(index="draft_slot", columns="method_scenario", values="final_rank")
        .reset_index()
        .rename_axis(columns=None)
        .sort_values("draft_slot")
    )


def _report(
    *,
    season_label: str,
    master_path: Path,
    tolerance: float,
    summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> str:
    display = summary[[
        "projection_method",
        "projection_type",
        "selection_scenario",
        "mean_final_points",
        "mean_rank",
        "win_rate",
        "top_3_rate",
        "mean_max_players_same_team",
        "mean_unique_player_teams",
    ]].copy()
    for column in display.columns[3:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    paired_display = paired_summary[[
        "projection_method",
        "selection_scenario",
        "mean_points_delta_vs_raw",
        "mean_rank_delta_vs_raw",
        "points_improved",
        "points_worsened",
        "mean_max_team_delta",
        "mean_unique_teams_delta",
    ]].copy()
    for column in [
        "mean_points_delta_vs_raw",
        "mean_rank_delta_vs_raw",
        "mean_max_team_delta",
        "mean_unique_teams_delta",
    ]:
        paired_display[column] = paired_display[column].map(lambda value: f"{value:.3f}")

    aggregate = (
        paired_summary.groupby("selection_scenario", as_index=False)
        .agg(
            mean_points_delta_vs_raw=("mean_points_delta_vs_raw", "mean"),
            mean_rank_delta_vs_raw=("mean_rank_delta_vs_raw", "mean"),
            mean_max_team_delta=("mean_max_team_delta", "mean"),
            mean_unique_teams_delta=("mean_unique_teams_delta", "mean"),
        )
    )
    conclusions = []
    for row in aggregate.itertuples(index=False):
        conclusions.append(
            f"- `{row.selection_scenario}`: {row.mean_points_delta_vs_raw:+.3f} points, "
            f"{row.mean_rank_delta_vs_raw:+.3f} rang, "
            f"{row.mean_max_team_delta:+.3f} joueur au maximum d'une même équipe et "
            f"{row.mean_unique_teams_delta:+.3f} équipe unique, en moyenne versus VOR brut."
        )

    return "\n".join(
        [
            f"# Départage VOR par équipe — {season_label}",
            "",
            f"- Master: `{master_path.as_posix()}`",
            "- Méthodes de projection: toutes les sources individuelles et `trimmed_mean_minmax`.",
            f"- Fenêtre: candidats dont le VOR est à au plus {tolerance:g} points du meilleur VOR disponible.",
            "- `team_stack_within_3`: privilégier l'équipe NHL la plus représentée dans le roster focal.",
            "- `team_diversify_within_3`: privilégier l'équipe NHL la moins représentée dans le roster focal.",
            "- Les joueurs `T` ne comptent pas dans la mesure de concentration finale. Les adversaires restent Pool Pro + VOR brut.",
            "",
            "## Résultats absolus",
            "",
            display.to_markdown(index=False),
            "",
            "## Effet apparié versus VOR brut",
            "",
            paired_display.to_markdown(index=False),
            "",
            "## Lecture globale",
            "",
            *conclusions,
            "",
            "Un delta de rang négatif est une amélioration. Les résultats couvrent 16 slots déterministes par méthode et une seule saison; le taux de victoire ne doit pas être interprété isolément.",
        ]
    )


def run_team_tiebreak_experiment(
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
    details, picks = run_team_tiebreak_drafts(
        prepared,
        gm_count=gm_count,
        roster_config=roster_config,
        tolerance=tolerance,
    )
    summary = summarize_team_tiebreaks(details)
    paired = paired_scenario_deltas(details)
    paired_summary = summarize_paired_scenarios(paired)
    by_slot = _by_slot(details)
    outputs = {
        "details": details,
        "summary": summary,
        "by_slot": by_slot,
        "paired": paired,
        "paired_summary": paired_summary,
        "picks": picks,
        "coverage": prepared.coverage,
        "exclusions": prepared.exclusions,
    }
    filenames = {
        "details": "01_team_tiebreak_details.csv",
        "summary": "02_team_tiebreak_summary.csv",
        "by_slot": "03_team_tiebreak_by_slot.csv",
        "paired": "04_team_tiebreak_vs_raw_by_slot.csv",
        "paired_summary": "05_team_tiebreak_vs_raw_summary.csv",
        "picks": "06_team_tiebreak_picks.csv",
        "coverage": "07_projection_coverage.csv",
        "exclusions": "08_projection_exclusions.csv",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in outputs.items():
        frame.to_csv(output_dir / filenames[key], index=False)
    report = _report(
        season_label=season_label,
        master_path=master_path,
        tolerance=tolerance,
        summary=summary,
        paired_summary=paired_summary,
    )
    (output_dir / "09_team_tiebreak_report.md").write_text(report, encoding="utf-8")
    return outputs
