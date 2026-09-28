from __future__ import annotations

from pathlib import Path

import pandas as pd


def _quality_bands(values: pd.DataFrame, value_column: str, rank_column: str) -> pd.DataFrame:
    ranked = values.sort_values(value_column, ascending=False).reset_index(drop=True)
    ranked[rank_column] = ranked.index + 1
    ranked[f"{rank_column}_band"] = pd.cut(
        ranked[rank_column],
        bins=[0, 8, 16, 24, 32],
        labels=["top_8", "rank_9_16", "rank_17_24", "bottom_8"],
    )
    return ranked


def analyze_team_stack_quality(
    *,
    master_path: Path,
    experiment_dir: Path,
    output_dir: Path,
    season_label: str,
) -> dict[str, pd.DataFrame]:
    master = pd.read_csv(master_path, low_memory=False)
    picks = pd.read_csv(experiment_dir / "06_team_tiebreak_picks.csv")
    paired = pd.read_csv(experiment_dir / "04_team_tiebreak_vs_raw_by_slot.csv")

    teams = master[master["category"].eq("T")][
        ["team_key", "actual_points"]
    ].rename(columns={"actual_points": "team_actual_points"})
    teams = _quality_bands(teams, "team_actual_points", "team_standings_rank")

    players = master[
        master["category"].ne("T") & master["team_key"].notna()
    ].copy()
    fantasy_strength = (
        players.sort_values(["team_key", "actual_points"], ascending=[True, False])
        .groupby("team_key")
        .head(5)
        .groupby("team_key", as_index=False)["actual_points"]
        .sum()
        .rename(columns={"actual_points": "top5_actual_player_points"})
    )
    fantasy_strength = _quality_bands(
        fantasy_strength, "top5_actual_player_points", "fantasy_strength_rank"
    )
    team_quality = teams.merge(fantasy_strength, on="team_key", how="outer")

    stack_picks = picks[
        picks["selection_scenario"].eq("team_stack_within_3")
        & picks["category"].ne("T")
    ]
    dominant = (
        stack_picks.groupby(
            ["projection_method", "projection_type", "draft_slot", "nhl_team"],
            as_index=False,
        )
        .agg(
            stacked_players=("entity_id", "size"),
            stacked_player_actual_points=("actual_points", "sum"),
        )
        .sort_values(
            [
                "projection_method",
                "draft_slot",
                "stacked_players",
                "stacked_player_actual_points",
            ],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(["projection_method", "draft_slot"])
    )
    delta_columns = [
        "projection_method",
        "draft_slot",
        "final_points_delta_vs_raw",
        "final_rank_delta_vs_raw",
    ]
    stack_deltas = paired[
        paired["selection_scenario"].eq("team_stack_within_3")
    ][delta_columns]
    dominant = (
        dominant.merge(
            stack_deltas,
            on=["projection_method", "draft_slot"],
            how="left",
            validate="one_to_one",
        )
        .merge(team_quality, left_on="nhl_team", right_on="team_key", how="left")
        .drop(columns="team_key")
        .sort_values(["projection_method", "draft_slot"])
        .reset_index(drop=True)
    )

    raw = picks[picks["selection_scenario"].eq("raw_vor")].copy()
    stack = picks[picks["selection_scenario"].eq("team_stack_within_3")].copy()
    keys = ["projection_method", "projection_type", "draft_slot", "entity_id"]
    raw_keys = raw[keys]
    stack_keys = stack[keys]
    removed = raw.merge(stack_keys, on=keys, how="left", indicator=True)
    removed = removed[removed["_merge"].eq("left_only")].drop(columns="_merge")
    removed["swap_action"] = "removed"
    removed["signed_actual_contribution"] = -removed["actual_points"]
    added = stack.merge(raw_keys, on=keys, how="left", indicator=True)
    added = added[added["_merge"].eq("left_only")].drop(columns="_merge")
    added["swap_action"] = "added"
    added["signed_actual_contribution"] = added["actual_points"]
    swaps = pd.concat([added, removed], ignore_index=True).sort_values(
        ["projection_method", "draft_slot", "swap_action", "overall_pick"]
    )

    team_summary = (
        dominant.groupby("nhl_team", as_index=False)
        .agg(
            cases=("draft_slot", "size"),
            team_standings_rank=("team_standings_rank", "first"),
            team_actual_points=("team_actual_points", "first"),
            fantasy_strength_rank=("fantasy_strength_rank", "first"),
            top5_actual_player_points=("top5_actual_player_points", "first"),
            mean_stacked_players=("stacked_players", "mean"),
            mean_points_delta=("final_points_delta_vs_raw", "mean"),
            median_points_delta=("final_points_delta_vs_raw", "median"),
            mean_rank_delta=("final_rank_delta_vs_raw", "mean"),
        )
        .sort_values(["cases", "mean_points_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )

    band_rows: list[dict[str, object]] = []
    for rank_column in ["team_standings_rank", "fantasy_strength_rank"]:
        band_column = f"{rank_column}_band"
        for band, group in dominant.groupby(band_column, observed=True):
            band_rows.append(
                {
                    "quality_measure": rank_column,
                    "quality_band": str(band),
                    "cases": len(group),
                    "mean_points_delta": group["final_points_delta_vs_raw"].mean(),
                    "median_points_delta": group["final_points_delta_vs_raw"].median(),
                    "mean_rank_delta": group["final_rank_delta_vs_raw"].mean(),
                }
            )
    band_summary = pd.DataFrame(band_rows)

    standings_corr = dominant["team_actual_points"].corr(
        dominant["final_points_delta_vs_raw"]
    )
    fantasy_corr = dominant["top5_actual_player_points"].corr(
        dominant["final_points_delta_vs_raw"]
    )
    best_cases = dominant.nlargest(8, "final_points_delta_vs_raw")[
        [
            "projection_method",
            "draft_slot",
            "nhl_team",
            "stacked_players",
            "team_standings_rank",
            "fantasy_strength_rank",
            "final_points_delta_vs_raw",
        ]
    ]
    worst_cases = dominant.nsmallest(8, "final_points_delta_vs_raw")[
        best_cases.columns
    ]
    report = "\n".join(
        [
            f"# Qualité de l'équipe dominante et stacking — {season_label}",
            "",
            f"- Corrélation points réels de l'équipe / delta du stacking: **{standings_corr:.3f}**.",
            f"- Corrélation force fantasy réelle des 5 meilleurs joueurs / delta du stacking: **{fantasy_corr:.3f}**.",
            "- Ces corrélations sont descriptives: les 16 slots d'une même méthode ne sont pas des observations indépendantes.",
            "",
            "## Résultats par bande de qualité",
            "",
            band_summary.to_markdown(index=False),
            "",
            "## Équipes dominantes les plus fréquentes",
            "",
            team_summary.head(20).to_markdown(index=False),
            "",
            "## Meilleurs cas de stacking",
            "",
            best_cases.to_markdown(index=False),
            "",
            "## Pires cas de stacking",
            "",
            worst_cases.to_markdown(index=False),
            "",
            "Les contributions exactes des joueurs ajoutés et retirés sont dans `04_stack_roster_swaps.csv`.",
        ]
    )

    outputs = {
        "dominant_cases": dominant,
        "team_summary": team_summary,
        "band_summary": band_summary,
        "swaps": swaps,
    }
    filenames = {
        "dominant_cases": "01_dominant_team_cases.csv",
        "team_summary": "02_team_quality_summary.csv",
        "band_summary": "03_quality_band_summary.csv",
        "swaps": "04_stack_roster_swaps.csv",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in outputs.items():
        frame.to_csv(output_dir / filenames[key], index=False)
    (output_dir / "05_team_stack_quality_report.md").write_text(report, encoding="utf-8")
    return outputs
