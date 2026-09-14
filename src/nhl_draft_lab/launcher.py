from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from nhl_draft_lab.backtest import (
    run_backtest_grid,
    summarize_backtests,
    summarize_backtests_by_slot,
)
from nhl_draft_lab.data.nhl_api import (
    build_session,
    export_excel,
    fetch_season,
    last_completed_season_ids,
)
from nhl_draft_lab.data.repository import available_seasons, load_assets, load_history, write_rankings
from nhl_draft_lab.draft.engine import replacement_levels, run_draft
from nhl_draft_lab.evaluation import (
    active_universe_projections,
    build_v1_comparison,
    coverage_summary,
    draft_zone_metrics,
    projection_metrics,
    skater_component_errors,
)
from nhl_draft_lab.forecasting import ForecastConfig, project_v1
from nhl_draft_lab.models import RosterConfig
from nhl_draft_lab.pressure import PressureConfig, build_tier_pressure_book
from nhl_draft_lab.strategies.factory import STRATEGY_NAMES, build_strategy
from nhl_draft_lab.strategies.manual import ManualStrategy
from nhl_draft_lab.tiering import TierConfig, build_tier_book


def parse_roster(text: str) -> RosterConfig:
    counts: dict[str, int] = {}
    for part in text.split(","):
        category, value = part.split("=", 1)
        category = category.strip().upper()
        if category not in {"F", "D", "G", "T"}:
            raise argparse.ArgumentTypeError(f"Unknown category {category!r}")
        counts[category] = int(value)
    return RosterConfig(counts)


def _tier_config_from_args(args: argparse.Namespace) -> TierConfig:
    return TierConfig(
        relative_width=args.tier_relative_width,
        max_size=args.tier_max_size,
        superstar_max_size=args.tier_superstar_max_size,
        superstar_tiers=args.tier_superstar_tiers,
    )


def _strategy_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "plateau_window": args.plateau_window,
        "plateau_weight": args.plateau_weight,
        "lookahead_candidates_per_category": args.lookahead_candidates_per_category,
        "lookahead2_focal_picks": args.lookahead2_focal_picks,
        "tier_relative_width": args.tier_relative_width,
        "tier_max_size": args.tier_max_size,
        "tier_superstar_max_size": args.tier_superstar_max_size,
        "tier_superstar_tiers": args.tier_superstar_tiers,
        "tier_lookahead_focal_picks": args.tier_lookahead_focal_picks,
        "tier_pressure_short_horizon": args.tier_pressure_short_horizon,
        "tier_pressure_mid_horizon": args.tier_pressure_mid_horizon,
        "tier_pressure_long_horizon": args.tier_pressure_long_horizon,
        "tier_pressure_temperature": args.tier_pressure_temperature,
        "tier_pressure_cap": args.tier_pressure_cap,
    }


def strategies_for_gms(args: argparse.Namespace):
    kwargs = _strategy_kwargs(args)
    strategies = {
        gm: build_strategy(args.strategy, **kwargs)
        for gm in range(1, args.gms + 1)
    }
    manual_slot = getattr(args, "manual_slot", None)
    if manual_slot is not None:
        if not 1 <= manual_slot <= args.gms:
            raise ValueError("manual-slot must be between 1 and number of GMs")
        strategies[manual_slot] = ManualStrategy()
    return strategies


def cmd_fetch(args: argparse.Namespace) -> None:
    seasons = args.seasons or last_completed_season_ids(4)
    session = build_session()
    for season in seasons:
        rankings = fetch_season(season, session=session)
        write_rankings(args.db, rankings)
        print(f"{season}: {len(rankings)} rows -> {args.db}")
        if args.excel_dir:
            path = args.excel_dir / f"nhl_rankings_{season}.xlsx"
            export_excel(rankings, season, path)
            print(f"  Excel -> {path}")


def cmd_project_v1(args: argparse.Namespace) -> None:
    history = load_history(args.db, args.history_seasons)
    manual_context = pd.read_csv(args.manual_context) if args.manual_context else None
    config = ForecastConfig(
        season_games=args.season_games,
        recency_weights=tuple(args.recency_weights),
    )
    projections = project_v1(history, manual_context=manual_context, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(args.output, index=False)

    print("Stage 2 — V1.0 component projections")
    print(f"History seasons: {', '.join(str(x) for x in sorted(history['season_id'].unique()))}")
    print(f"Rows: {len(projections)}")
    print(projections.groupby("category").size().rename("assets").to_string())
    print(f"\nProjections -> {args.output}")


def cmd_evaluate_v1(args: argparse.Namespace) -> None:
    v0_master = pd.read_csv(args.v0_master)
    v1 = pd.read_csv(args.v1)
    comparison = build_v1_comparison(v0_master, v1)
    coverage = coverage_summary(comparison)
    metrics = projection_metrics(comparison)
    skater_components = skater_component_errors(comparison)
    active_projections = active_universe_projections(comparison)
    draft_counts = {
        category: args.draft_counts.required(category)
        for category in args.draft_counts.counts
        if args.draft_counts.required(category) > 0
    }
    relevant_metrics = draft_zone_metrics(
        comparison,
        draft_counts=draft_counts,
        defense_focus_ranks=tuple(args.defense_focus_ranks),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "01_v1_universe_comparison.csv", index=False)
    coverage.to_csv(args.output_dir / "02_v1_coverage.csv", index=False)
    metrics.to_csv(args.output_dir / "03_v1_projection_metrics.csv", index=False)
    skater_components.to_csv(args.output_dir / "04_v1_skater_component_errors.csv", index=False)
    active_projections.to_csv(args.output_dir / "05_v1_active_universe_projections.csv", index=False)
    relevant_metrics.to_csv(args.output_dir / "06_v1_draft_zone_metrics.csv", index=False)

    print("Stage 2 — V1.0 evaluation against frozen V0")
    print("\nCoverage")
    print(coverage.to_string(index=False, formatters={"v1_coverage_rate": "{:.1%}".format}))
    print("\nOverall metrics")
    overall = metrics.loc[metrics["category"] == "ALL", [
        "model", "n", "mae", "rmse", "bias_actual_minus_projection",
        "pearson", "spearman", "mae_delta_vs_v0_same_coverage",
    ]]
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nDraft-relevant metrics")
    relevant_display = relevant_metrics.loc[
        relevant_metrics["model"].isin({"V0_same_V1_coverage", "V1_history_components"}),
        ["segment", "category", "rank_cutoff", "model", "n", "mae", "rmse", "spearman"],
    ]
    print(relevant_display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nEvaluation -> {args.output_dir}")


def cmd_draft(args: argparse.Namespace) -> None:
    assets = load_assets(args.db, args.season, args.value_column)
    result = run_draft(assets, args.gms, args.roster, strategies_for_gms(args))

    rows = []
    for pick in result.picks:
        rows.append({
            "overall_pick": pick.overall_pick,
            "round": pick.round_no,
            "gm": pick.gm_id,
            "strategy": pick.strategy_name,
            "category": pick.asset.category,
            "name": pick.asset.name,
            "value": pick.asset.value,
        })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print("\nFinal totals")
    for gm, total in sorted(result.totals().items(), key=lambda kv: kv[1], reverse=True):
        print(f"GM {gm:2d}: {total:8.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"\nDraft log -> {args.output}")


def cmd_tiers(args: argparse.Namespace) -> None:
    assets = load_assets(args.db, args.season, args.value_column)
    levels = replacement_levels(assets, args.gms, args.roster)
    book = build_tier_book(assets, levels, _tier_config_from_args(args))

    rows: list[dict[str, object]] = []
    for category, tiers in book.tiers_by_category.items():
        for index, tier in enumerate(tiers):
            next_value = tiers[index + 1].tier_value if index + 1 < len(tiers) else None
            rows.append({
                "category": category,
                "tier": tier.tier_no,
                "size": tier.size,
                "tier_value_median_vorp": tier.tier_value,
                "leader_vorp": tier.leader_vorp,
                "min_vorp": tier.min_vorp,
                "gap_to_next_tier": (
                    tier.tier_value - next_value if next_value is not None else None
                ),
                "members": " | ".join(tier.asset_names),
            })

    frame = pd.DataFrame(rows)
    print(
        f"Fixed tiers — season {args.season} | GM {args.gms} | "
        f"roster {dict(args.roster.counts)}"
    )
    print(
        f"relative_width={args.tier_relative_width:.3f} | "
        f"max_size={args.tier_max_size} | "
        f"superstar_max_size={args.tier_superstar_max_size} | "
        f"superstar_tiers={args.tier_superstar_tiers}"
    )
    print(frame.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"\nTier report -> {args.output}")



def cmd_tier_scores(args: argparse.Namespace) -> None:
    assets = load_assets(args.db, args.season, args.value_column)
    levels = replacement_levels(assets, args.gms, args.roster)
    tier_book = build_tier_book(assets, levels, _tier_config_from_args(args))
    pressure_config = PressureConfig(
        short_horizon=args.tier_pressure_short_horizon,
        mid_horizon=args.tier_pressure_mid_horizon,
        long_horizon=args.tier_pressure_long_horizon,
        temperature=args.tier_pressure_temperature,
        cap=args.tier_pressure_cap,
    )
    pressure_book = build_tier_pressure_book(tier_book, args.roster, pressure_config)

    rows: list[dict[str, object]] = []
    for category, tiers in tier_book.tiers_by_category.items():
        for tier in tiers:
            score = pressure_book.for_tier(tier)
            rows.append({
                "category": category,
                "tier": tier.tier_no,
                "players": " | ".join(tier.asset_names),
                "tier_value": tier.tier_value,
                "size": tier.size,
                "gap_to_next": score.gap_to_next_tier,
                "SHORT": score.short_score,
                "MID": score.mid_score,
                "LONG": score.long_score,
                "estimated_pick_share": score.estimated_pick_share,
                "market_pick": score.market_pick,
            })

    frame = pd.DataFrame(rows)
    printable = [
        "category", "tier", "players", "tier_value", "size",
        "gap_to_next", "SHORT", "MID", "LONG",
    ]
    print(
        f"Printable tier pressure scores — season {args.season} | GM {args.gms} | "
        f"roster {dict(args.roster.counts)}"
    )
    print(
        f"horizons={pressure_config.short_horizon}/"
        f"{pressure_config.mid_horizon}/{pressure_config.long_horizon} | "
        f"temperature={pressure_config.temperature:.2f} | "
        f"cap={pressure_config.cap:.1%}"
    )
    print(
        frame[printable].to_string(
            index=False,
            formatters={
                "tier_value": "{:.2f}".format,
                "gap_to_next": "{:.2f}".format,
                "SHORT": "{:.2f}".format,
                "MID": "{:.2f}".format,
                "LONG": "{:.2f}".format,
            },
        )
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"\nTier score sheet -> {args.output}")

def _summary_formatters() -> dict[str, object]:
    return {
        "mean_points": "{:.2f}".format,
        "mean_rank": "{:.2f}".format,
        "win_rate": "{:.1%}".format,
        "top3_rate": "{:.1%}".format,
        "mean_gap_to_winner": "{:.2f}".format,
        "mean_margin_vs_field": "{:+.2f}".format,
        "divergence_rate": "{:.1%}".format,
        "mean_direct_points_delta_when_diverging": "{:+.2f}".format,
        "mean_direct_vorp_delta_when_diverging": "{:+.2f}".format,
    }


def _display_columns(frame: pd.DataFrame) -> list[str]:
    wanted = [
        "focal_strategy",
        "opponent_strategy",
        "diagnostic_baseline",
        "trials",
        "mean_points",
        "mean_rank",
        "win_rate",
        "top3_rate",
        "mean_gap_to_winner",
        "mean_margin_vs_field",
        "divergence_rate",
        "mean_direct_points_delta_when_diverging",
    ]
    return [column for column in wanted if column in frame.columns]


def cmd_backtest(args: argparse.Namespace) -> None:
    seasons = args.seasons or available_seasons(args.db)
    frame = run_backtest_grid(
        db_path=args.db,
        seasons=seasons,
        gm_count=args.gms,
        roster_config=args.roster,
        focal_strategies=args.focal_strategies,
        opponent_strategies=args.opponent_strategies,
        value_column=args.value_column,
        plateau_window=args.plateau_window,
        plateau_weight=args.plateau_weight,
        lookahead_candidates_per_category=args.lookahead_candidates_per_category,
        lookahead2_focal_picks=args.lookahead2_focal_picks,
        tier_relative_width=args.tier_relative_width,
        tier_max_size=args.tier_max_size,
        tier_superstar_max_size=args.tier_superstar_max_size,
        tier_superstar_tiers=args.tier_superstar_tiers,
        tier_lookahead_focal_picks=args.tier_lookahead_focal_picks,
        tier_pressure_short_horizon=args.tier_pressure_short_horizon,
        tier_pressure_mid_horizon=args.tier_pressure_mid_horizon,
        tier_pressure_long_horizon=args.tier_pressure_long_horizon,
        tier_pressure_temperature=args.tier_pressure_temperature,
        tier_pressure_cap=args.tier_pressure_cap,
    )
    summary = summarize_backtests(frame)
    by_slot = summarize_backtests_by_slot(frame)

    print("Stage 1 — perfect-information strategy backtest")
    print(f"Seasons: {', '.join(str(s) for s in seasons)}")
    print(f"GM: {args.gms} | roster: {dict(args.roster.counts)}")
    print("\nSummary")
    print(
        summary[_display_columns(summary)].to_string(
            index=False,
            formatters=_summary_formatters(),
        )
    )

    if args.diagnostics:
        print("\nDiagnostics by draft slot")
        slot_columns = [
            "focal_strategy",
            "opponent_strategy",
            "diagnostic_baseline",
            "draft_slot",
            "mean_points",
            "mean_rank",
            "win_rate",
            "top3_rate",
            "mean_margin_vs_field",
            "divergence_rate",
            "mean_direct_points_delta_when_diverging",
        ]
        slot_columns = [column for column in slot_columns if column in by_slot.columns]
        print(
            by_slot[slot_columns].to_string(
                index=False,
                formatters=_summary_formatters(),
            )
        )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        details_path = args.output_dir / "stage1_backtest_details.csv"
        summary_path = args.output_dir / "stage1_backtest_summary.csv"
        by_slot_path = args.output_dir / "stage1_backtest_by_slot.csv"
        frame.to_csv(details_path, index=False)
        summary.to_csv(summary_path, index=False)
        by_slot.to_csv(by_slot_path, index=False)
        print(f"\nDetails -> {details_path}")
        print(f"Summary -> {summary_path}")
        print(f"By slot -> {by_slot_path}")


def add_tier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tier-relative-width",
        type=float,
        default=0.05,
        help="Max anchored VORP gap / tier-leader raw points (default: 0.05)",
    )
    parser.add_argument("--tier-max-size", type=int, default=5)
    parser.add_argument("--tier-superstar-max-size", type=int, default=3)
    parser.add_argument(
        "--tier-superstar-tiers",
        type=int,
        default=2,
        help="Number of top tiers using the smaller superstar cap (0 disables)",
    )
    parser.add_argument(
        "--tier-lookahead-focal-picks",
        type=int,
        default=3,
        help="Current + future focal picks optimized by tier_lookahead (default: 3)",
    )
    parser.add_argument("--tier-pressure-short-horizon", type=int, default=4)
    parser.add_argument("--tier-pressure-mid-horizon", type=int, default=14)
    parser.add_argument("--tier-pressure-long-horizon", type=int, default=24)
    parser.add_argument(
        "--tier-pressure-temperature",
        type=float,
        default=10.0,
        help=(
            "VORP points controlling how strongly better tiers attract picks; "
            "10 points multiplies demand weight by e at the default temperature"
        ),
    )
    parser.add_argument(
        "--tier-pressure-cap",
        type=float,
        default=0.03,
        help=(
            "Maximum pressure bonus as a fraction of tier median VORP "
            "(default: 0.03 = 3%%)"
        ),
    )


def add_strategy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="vorp")
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--plateau-weight", type=float, default=1.0)
    parser.add_argument("--lookahead-candidates-per-category", type=int, default=5)
    parser.add_argument(
        "--lookahead2-focal-picks",
        type=int,
        default=3,
        help="Total focal picks optimised by lookahead2 (default: current + next 2 = 3)",
    )
    add_tier_args(parser)


def _add_default_pool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=Path("data/nhl_history.sqlite"))
    parser.add_argument("--gms", type=int, default=15)
    parser.add_argument(
        "--roster",
        type=parse_roster,
        default=RosterConfig({"F": 10, "D": 3, "G": 2, "T": 1}),
    )
    parser.add_argument("--value-column", default="pool_points_84")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NHL Draft Lab — Stage 1: perfect-information draft strategy backtests"
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch historical NHL final-season data")
    fetch.add_argument("--db", type=Path, default=Path("data/nhl_history.sqlite"))
    fetch.add_argument("--seasons", type=int, nargs="*")
    fetch.add_argument("--excel-dir", type=Path)
    fetch.set_defaults(func=cmd_fetch)

    project_v1_parser = sub.add_parser(
        "project-v1",
        help="Build transparent PPG/GP and goalie-component projections",
    )
    project_v1_parser.add_argument("--db", type=Path, default=Path("data/nhl_history.sqlite"))
    project_v1_parser.add_argument(
        "--history-seasons",
        type=int,
        nargs="*",
        help="Optional subset of seasons; default uses every season in the database",
    )
    project_v1_parser.add_argument("--manual-context", type=Path)
    project_v1_parser.add_argument("--season-games", type=int, default=82)
    project_v1_parser.add_argument(
        "--recency-weights",
        type=float,
        nargs="+",
        default=[0.60, 0.30, 0.10],
        help="Weights from most recent to oldest season",
    )
    project_v1_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/v1/projections.csv"),
    )
    project_v1_parser.set_defaults(func=cmd_project_v1)

    evaluate_v1 = sub.add_parser(
        "evaluate-v1",
        help="Compare V1 component projections with the frozen V0 universe",
    )
    evaluate_v1.add_argument(
        "--v0-master",
        type=Path,
        required=True,
        help="02_projection_master.csv generated by v0_full_analysis_fixed.py",
    )
    evaluate_v1.add_argument(
        "--v1",
        type=Path,
        default=Path("output/v1/projections.csv"),
    )
    evaluate_v1.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/v1/evaluation"),
    )
    evaluate_v1.add_argument(
        "--draft-counts",
        type=parse_roster,
        default=RosterConfig({"F": 200, "D": 75, "G": 45, "T": 25}),
        help="Total category cutoffs for the draft-relevant evaluation zone",
    )
    evaluate_v1.add_argument(
        "--defense-focus-ranks",
        type=int,
        nargs="+",
        default=[6, 9],
        help="Elite-defense rank windows reported separately (default: 6 9)",
    )
    evaluate_v1.set_defaults(func=cmd_evaluate_v1)

    draft = sub.add_parser("draft", help="Run one omniscient historical snake draft")
    _add_default_pool_args(draft)
    draft.add_argument("--season", type=int, required=True)
    draft.add_argument("--manual-slot", type=int)
    draft.add_argument("--output", type=Path)
    add_strategy_args(draft)
    draft.set_defaults(func=cmd_draft)

    tiers = sub.add_parser("tiers", help="Inspect fixed pre-draft VORP tiers")
    _add_default_pool_args(tiers)
    tiers.add_argument("--season", type=int, required=True)
    tiers.add_argument("--output", type=Path)
    add_tier_args(tiers)
    tiers.set_defaults(func=cmd_tiers)

    tier_scores = sub.add_parser(
        "tier-scores",
        help="Export fixed printable SHORT / MID / LONG scores for every tier",
    )
    _add_default_pool_args(tier_scores)
    tier_scores.add_argument("--season", type=int, required=True)
    tier_scores.add_argument("--output", type=Path)
    add_tier_args(tier_scores)
    tier_scores.set_defaults(func=cmd_tier_scores)

    backtest = sub.add_parser(
        "backtest",
        help="Compare Stage-1 strategies with perfect knowledge of final season values",
    )
    _add_default_pool_args(backtest)
    backtest.add_argument("--seasons", type=int, nargs="*")
    backtest.add_argument(
        "--focal-strategies",
        nargs="+",
        choices=STRATEGY_NAMES,
        default=list(STRATEGY_NAMES),
    )
    backtest.add_argument(
        "--opponent-strategies",
        nargs="+",
        choices=STRATEGY_NAMES,
        default=["vorp"],
    )
    backtest.add_argument("--plateau-window", type=int, default=5)
    backtest.add_argument("--plateau-weight", type=float, default=1.0)
    backtest.add_argument("--lookahead-candidates-per-category", type=int, default=5)
    backtest.add_argument(
        "--lookahead2-focal-picks",
        type=int,
        default=3,
        help="Total focal picks optimised by lookahead2 (default: 3)",
    )
    add_tier_args(backtest)
    backtest.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print performance and divergence metrics by draft slot",
    )
    backtest.add_argument("--output-dir", type=Path)
    backtest.set_defaults(func=cmd_backtest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
