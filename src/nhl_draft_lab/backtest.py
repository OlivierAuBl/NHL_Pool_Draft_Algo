from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

import pandas as pd

from nhl_draft_lab.data.repository import load_assets
from nhl_draft_lab.draft.engine import run_draft
from nhl_draft_lab.models import DraftAsset, RosterConfig
from nhl_draft_lab.strategies.diagnostic import DiagnosticStrategy
from nhl_draft_lab.strategies.factory import STRATEGY_NAMES, build_strategy


@dataclass(frozen=True)
class BacktestResult:
    season_id: int
    gm_count: int
    draft_slot: int
    focal_strategy: str
    opponent_strategy: str
    total_points: float
    rank: int
    winner_points: float
    gap_to_winner: float
    margin_vs_field_mean: float
    focal_picks: int
    divergence_count: int
    direct_points_delta_sum: float
    direct_vorp_delta_sum: float
    diagnostic_baseline: str

    @property
    def divergence_rate(self) -> float:
        return self.divergence_count / self.focal_picks if self.focal_picks else 0.0


def _competition_rank(totals: dict[int, float], gm_id: int) -> int:
    """1 = best. Equal totals receive the same competition rank."""
    focal = totals[gm_id]
    return 1 + sum(total > focal for other, total in totals.items() if other != gm_id)


def run_one_focal_backtest(
    assets: list[DraftAsset],
    *,
    season_id: int,
    gm_count: int,
    roster_config: RosterConfig,
    draft_slot: int,
    focal_strategy: str,
    opponent_strategy: str,
    plateau_window: int = 5,
    plateau_weight: float = 1.0,
    lookahead_candidates_per_category: int = 5,
    lookahead2_focal_picks: int = 3,
    tier_relative_width: float = 0.05,
    tier_max_size: int = 5,
    tier_superstar_max_size: int = 3,
    tier_superstar_tiers: int = 2,
    tier_lookahead_focal_picks: int = 3,
    tier_pressure_short_horizon: int = 4,
    tier_pressure_mid_horizon: int = 14,
    tier_pressure_long_horizon: int = 24,
    tier_pressure_temperature: float = 10.0,
    tier_pressure_cap: float = 0.03,
) -> BacktestResult:
    """Run one deterministic perfect-information draft.

    The focal strategy is wrapped with a local VORP comparator. This gives us
    diagnostic information about *when* the strategy differs from VORP while
    the final points/rank measure whether those local deviations paid off.
    """
    if not 1 <= draft_slot <= gm_count:
        raise ValueError("draft_slot must be between 1 and gm_count")

    diagnostic_focal: DiagnosticStrategy | None = None
    strategies = {}
    for gm_id in range(1, gm_count + 1):
        name = focal_strategy if gm_id == draft_slot else opponent_strategy
        strategy = build_strategy(
            name,
            plateau_window=plateau_window,
            plateau_weight=plateau_weight,
            lookahead_candidates_per_category=lookahead_candidates_per_category,
            lookahead2_focal_picks=lookahead2_focal_picks,
            tier_relative_width=tier_relative_width,
            tier_max_size=tier_max_size,
            tier_superstar_max_size=tier_superstar_max_size,
            tier_superstar_tiers=tier_superstar_tiers,
            tier_lookahead_focal_picks=tier_lookahead_focal_picks,
            tier_pressure_short_horizon=tier_pressure_short_horizon,
            tier_pressure_mid_horizon=tier_pressure_mid_horizon,
            tier_pressure_long_horizon=tier_pressure_long_horizon,
            tier_pressure_temperature=tier_pressure_temperature,
            tier_pressure_cap=tier_pressure_cap,
        )
        if gm_id == draft_slot:
            baseline_name = "tier_vorp" if focal_strategy in {"tier_lookahead", "tier_pressure"} else "vorp"
            baseline = build_strategy(
                baseline_name,
                tier_relative_width=tier_relative_width,
                tier_max_size=tier_max_size,
                tier_superstar_max_size=tier_superstar_max_size,
                tier_superstar_tiers=tier_superstar_tiers,
                tier_lookahead_focal_picks=tier_lookahead_focal_picks,
                tier_pressure_short_horizon=tier_pressure_short_horizon,
                tier_pressure_mid_horizon=tier_pressure_mid_horizon,
                tier_pressure_long_horizon=tier_pressure_long_horizon,
                tier_pressure_temperature=tier_pressure_temperature,
                tier_pressure_cap=tier_pressure_cap,
            )
            diagnostic_focal = DiagnosticStrategy(strategy, baseline)
            strategies[gm_id] = diagnostic_focal
        else:
            strategies[gm_id] = strategy

    result = run_draft(assets, gm_count, roster_config, strategies)
    totals = result.totals()
    focal_total = totals[draft_slot]
    winner_points = max(totals.values())
    field = [points for gm, points in totals.items() if gm != draft_slot]

    assert diagnostic_focal is not None
    divergences = [record for record in diagnostic_focal.records if record.diverged]

    return BacktestResult(
        season_id=season_id,
        gm_count=gm_count,
        draft_slot=draft_slot,
        focal_strategy=focal_strategy,
        opponent_strategy=opponent_strategy,
        total_points=focal_total,
        rank=_competition_rank(totals, draft_slot),
        winner_points=winner_points,
        gap_to_winner=winner_points - focal_total,
        margin_vs_field_mean=focal_total - fmean(field),
        focal_picks=len(diagnostic_focal.records),
        divergence_count=len(divergences),
        direct_points_delta_sum=sum(r.direct_points_delta for r in divergences),
        direct_vorp_delta_sum=sum(r.direct_vorp_delta for r in divergences),
        diagnostic_baseline=("tier_vorp" if focal_strategy in {"tier_lookahead", "tier_pressure"} else "vorp"),
    )


def run_backtest_grid(
    *,
    db_path: Path,
    seasons: list[int],
    gm_count: int,
    roster_config: RosterConfig,
    focal_strategies: list[str] | None = None,
    opponent_strategies: list[str] | None = None,
    value_column: str = "pool_points_84",
    plateau_window: int = 5,
    plateau_weight: float = 1.0,
    lookahead_candidates_per_category: int = 5,
    lookahead2_focal_picks: int = 3,
    tier_relative_width: float = 0.05,
    tier_max_size: int = 5,
    tier_superstar_max_size: int = 3,
    tier_superstar_tiers: int = 2,
    tier_lookahead_focal_picks: int = 3,
    tier_pressure_short_horizon: int = 4,
    tier_pressure_mid_horizon: int = 14,
    tier_pressure_long_horizon: int = 24,
    tier_pressure_temperature: float = 10.0,
    tier_pressure_cap: float = 0.03,
) -> pd.DataFrame:
    """Backtest strategies for every season, opponent model and draft slot."""
    focal_strategies = focal_strategies or list(STRATEGY_NAMES)
    opponent_strategies = opponent_strategies or list(STRATEGY_NAMES)

    rows: list[dict[str, object]] = []
    for season_id in seasons:
        assets = load_assets(db_path, season_id, value_column=value_column)
        for opponent in opponent_strategies:
            for focal in focal_strategies:
                for draft_slot in range(1, gm_count + 1):
                    result = run_one_focal_backtest(
                        assets,
                        season_id=season_id,
                        gm_count=gm_count,
                        roster_config=roster_config,
                        draft_slot=draft_slot,
                        focal_strategy=focal,
                        opponent_strategy=opponent,
                        plateau_window=plateau_window,
                        plateau_weight=plateau_weight,
                        lookahead_candidates_per_category=lookahead_candidates_per_category,
                        lookahead2_focal_picks=lookahead2_focal_picks,
                        tier_relative_width=tier_relative_width,
                        tier_max_size=tier_max_size,
                        tier_superstar_max_size=tier_superstar_max_size,
                        tier_superstar_tiers=tier_superstar_tiers,
                        tier_lookahead_focal_picks=tier_lookahead_focal_picks,
                        tier_pressure_short_horizon=tier_pressure_short_horizon,
                        tier_pressure_mid_horizon=tier_pressure_mid_horizon,
                        tier_pressure_long_horizon=tier_pressure_long_horizon,
                        tier_pressure_temperature=tier_pressure_temperature,
                        tier_pressure_cap=tier_pressure_cap,
                    )
                    row = asdict(result)
                    row["divergence_rate"] = result.divergence_rate
                    rows.append(row)

    return pd.DataFrame(rows)


def _ensure_diagnostic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    defaults = {
        "focal_picks": 0,
        "divergence_count": 0,
        "direct_points_delta_sum": 0.0,
        "direct_vorp_delta_sum": 0.0,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def _summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    frame = _ensure_diagnostic_columns(frame)
    if "diagnostic_baseline" not in frame.columns:
        frame["diagnostic_baseline"] = "vorp"
    work = frame.assign(
        win=(frame["rank"] == 1).astype(float),
        top3=(frame["rank"] <= 3).astype(float),
    )

    summary = (
        work.groupby(group_columns, as_index=False)
        .agg(
            trials=("rank", "size"),
            mean_points=("total_points", "mean"),
            mean_rank=("rank", "mean"),
            win_rate=("win", "mean"),
            top3_rate=("top3", "mean"),
            mean_gap_to_winner=("gap_to_winner", "mean"),
            mean_margin_vs_field=("margin_vs_field_mean", "mean"),
            focal_picks=("focal_picks", "sum"),
            divergence_count=("divergence_count", "sum"),
            direct_points_delta_sum=("direct_points_delta_sum", "sum"),
            direct_vorp_delta_sum=("direct_vorp_delta_sum", "sum"),
        )
    )

    summary["divergence_rate"] = summary.apply(
        lambda row: row["divergence_count"] / row["focal_picks"]
        if row["focal_picks"]
        else 0.0,
        axis=1,
    )
    summary["mean_direct_points_delta_when_diverging"] = summary.apply(
        lambda row: row["direct_points_delta_sum"] / row["divergence_count"]
        if row["divergence_count"]
        else 0.0,
        axis=1,
    )
    summary["mean_direct_vorp_delta_when_diverging"] = summary.apply(
        lambda row: row["direct_vorp_delta_sum"] / row["divergence_count"]
        if row["divergence_count"]
        else 0.0,
        axis=1,
    )

    return summary


def summarize_backtests(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Stage-1 performance plus divergence-vs-VORP diagnostics."""
    summary = _summarize(frame, ["focal_strategy", "opponent_strategy", "diagnostic_baseline"])
    if summary.empty:
        return summary
    return summary.sort_values(
        ["mean_rank", "mean_gap_to_winner", "mean_points"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def summarize_backtests_by_slot(frame: pd.DataFrame) -> pd.DataFrame:
    """Show whether a strategy's advantage depends on snake draft position."""
    summary = _summarize(
        frame,
        ["focal_strategy", "opponent_strategy", "diagnostic_baseline", "draft_slot"],
    )
    if summary.empty:
        return summary
    return summary.sort_values(
        ["focal_strategy", "opponent_strategy", "draft_slot"]
    ).reset_index(drop=True)
