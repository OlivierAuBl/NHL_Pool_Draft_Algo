from __future__ import annotations

from statistics import fmean
from typing import Mapping

import pandas as pd

from nhl_draft_lab.draft.engine import DraftResult, replacement_levels, run_draft
from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.strategies.base import Strategy
from nhl_draft_lab.strategies.factory import build_strategy


def _identifier(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value).strip()


def _asset_key(row: pd.Series) -> str:
    if str(row["category"]).upper() == "T":
        team = str(row.get("team_key", "")).strip().upper()
        return f"T:{team}"
    return f"P:{_identifier(row.get('NHLID'))}"


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def build_projection_draft_universe(
    v0_master: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align V0, the candidate and realised values on the frozen V0 universe."""

    v0_required = {
        "category", "projection_median", "actual_points", "NHLID", "team_key",
    }
    candidate_required = {"entity_id", "category", "projected_points"}
    v0_missing = v0_required - set(v0_master.columns)
    candidate_missing = candidate_required - set(candidate.columns)
    if v0_missing:
        raise ValueError(f"V0 master is missing columns: {sorted(v0_missing)}")
    if candidate_missing:
        raise ValueError(f"Candidate is missing columns: {sorted(candidate_missing)}")

    left = v0_master.copy()
    left["category"] = left["category"].astype(str).str.upper()
    left["entity_id"] = left.apply(_asset_key, axis=1)
    left["v0_projected_points"] = pd.to_numeric(
        left["projection_median"], errors="coerce"
    )
    left["actual_points"] = pd.to_numeric(left["actual_points"], errors="coerce")
    names = left.get("FullName", pd.Series(index=left.index, dtype=object)).copy()
    if "Team" in left:
        names = names.fillna(left["Team"])
    left["name"] = names.fillna(left["entity_id"])
    if left["entity_id"].duplicated().any():
        duplicates = sorted(left.loc[left["entity_id"].duplicated(), "entity_id"].unique())
        raise ValueError(f"V0 master contains duplicate asset keys: {duplicates[:10]}")

    right = candidate.copy()
    right["entity_id"] = right["entity_id"].map(str).str.strip()
    right["category"] = right["category"].astype(str).str.upper()
    right["candidate_projected_points"] = pd.to_numeric(
        right["projected_points"], errors="coerce"
    )
    if right["entity_id"].duplicated().any():
        duplicates = sorted(right.loc[right["entity_id"].duplicated(), "entity_id"].unique())
        raise ValueError(f"Candidate contains duplicate asset keys: {duplicates[:10]}")

    universe = left.merge(
        right[["entity_id", "category", "candidate_projected_points"]],
        on=["entity_id", "category"],
        how="left",
        validate="one_to_one",
    )
    universe["match_status"] = "MATCHED"
    universe.loc[
        universe["v0_projected_points"].isna(), "match_status"
    ] = "MISSING_V0"
    universe.loc[universe["actual_points"].isna(), "match_status"] = "MISSING_ACTUAL"
    universe.loc[
        universe["candidate_projected_points"].isna(), "match_status"
    ] = "MISSING_CANDIDATE"
    audit = universe[[
        "entity_id", "name", "category", "v0_projected_points",
        "candidate_projected_points", "actual_points", "match_status",
    ]].copy()
    matched = universe.loc[universe["match_status"].eq("MATCHED")].copy()
    return matched, audit


def _decision_assets(universe: pd.DataFrame, value_column: str) -> list[DraftAsset]:
    return [
        DraftAsset(
            entity_id=str(row.entity_id),
            name=str(row.name),
            category=str(row.category),
            value=float(getattr(row, value_column)),
            nhl_team=_text(getattr(row, "team_key", "")),
            metadata={
                "actual_points": float(row.actual_points),
                "v0_projected_points": float(row.v0_projected_points),
                "candidate_projected_points": float(row.candidate_projected_points),
            },
        )
        for row in universe.itertuples(index=False)
    ]


class _ProjectionViewStrategy:
    """Let one GM value shared draft assets with a different projection model."""

    def __init__(
        self,
        strategy: Strategy,
        *,
        value_column: str,
        gm_count: int,
    ) -> None:
        self.strategy = strategy
        self.value_column = value_column
        self.gm_count = gm_count
        self.name = strategy.name
        self._replacement_levels: dict[str, float] | None = None

    def _view_asset(self, asset: DraftAsset) -> DraftAsset:
        return DraftAsset(
            entity_id=asset.entity_id,
            name=asset.name,
            category=asset.category,
            value=float(asset.metadata[self.value_column]),
            nhl_team=asset.nhl_team,
            metadata=asset.metadata,
        )

    def choose(self, context: DraftContext) -> DraftAsset:
        original_by_id = {
            asset.entity_id: asset for asset in context.available_assets
        }
        initial = tuple(self._view_asset(asset) for asset in context.initial_assets)
        if self._replacement_levels is None:
            self._replacement_levels = replacement_levels(
                list(initial), self.gm_count, context.roster_config
            )
        viewed = DraftContext(
            gm_id=context.gm_id,
            draft_slot=context.draft_slot,
            overall_pick=context.overall_pick,
            round_no=context.round_no,
            picks_until_next=context.picks_until_next,
            roster=tuple(self._view_asset(asset) for asset in context.roster),
            roster_config=context.roster_config,
            available_assets=tuple(
                self._view_asset(asset) for asset in context.available_assets
            ),
            replacement_levels=self._replacement_levels,
            rosters_by_gm={
                gm_id: tuple(self._view_asset(asset) for asset in roster)
                for gm_id, roster in context.rosters_by_gm.items()
            },
            future_order=context.future_order,
            initial_assets=initial,
        )
        chosen = self.strategy.choose(viewed)
        return original_by_id[chosen.entity_id]


def _competition_rank(totals: Mapping[int, float], gm_id: int) -> int:
    focal = totals[gm_id]
    return 1 + sum(value > focal for other, value in totals.items() if other != gm_id)


def run_projection_draft_comparison(
    universe: pd.DataFrame,
    *,
    gm_count: int,
    roster_config: RosterConfig,
    strategy_names: tuple[str, ...] = ("vorp", "tier_vorp"),
    strategy_kwargs: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draft with each projection model, then score every roster with actuals."""

    if gm_count < 2:
        raise ValueError("gm_count must be at least 2")
    if not strategy_names:
        raise ValueError("strategy_names cannot be empty")
    strategy_kwargs = dict(strategy_kwargs or {})

    counts = universe["category"].value_counts().to_dict()
    for category, slots in roster_config.counts.items():
        required = gm_count * int(slots)
        if counts.get(category, 0) < required:
            raise ValueError(
                f"Projection universe has only {counts.get(category, 0)} {category}; "
                f"need {required}"
            )

    model_columns = {
        "V0": "v0_projected_points",
        "V1_WEIGHTED_GP_CANDIDATE": "candidate_projected_points",
    }
    detail_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []
    for model, value_column in model_columns.items():
        assets = _decision_assets(universe, value_column)
        for strategy_name in strategy_names:
            strategies = {
                gm_id: build_strategy(strategy_name, **strategy_kwargs)
                for gm_id in range(1, gm_count + 1)
            }
            result = run_draft(assets, gm_count, roster_config, strategies)
            actual_totals = {
                gm_id: float(sum(asset.metadata["actual_points"] for asset in roster))
                for gm_id, roster in result.rosters.items()
            }
            projected_totals = {
                gm_id: float(sum(asset.value for asset in roster))
                for gm_id, roster in result.rosters.items()
            }
            winner = max(actual_totals.values())
            for gm_id in range(1, gm_count + 1):
                field = [
                    value for other, value in actual_totals.items() if other != gm_id
                ]
                detail_rows.append({
                    "projection_model": model,
                    "strategy": strategy_name,
                    "draft_slot": gm_id,
                    "projected_roster_points": projected_totals[gm_id],
                    "actual_roster_points": actual_totals[gm_id],
                    "actual_rank": _competition_rank(actual_totals, gm_id),
                    "gap_to_winner": winner - actual_totals[gm_id],
                    "margin_vs_field_mean": actual_totals[gm_id] - fmean(field),
                })
            for pick in result.picks:
                pick_rows.append({
                    "projection_model": model,
                    "strategy": strategy_name,
                    "draft_slot": pick.gm_id,
                    "overall_pick": pick.overall_pick,
                    "round_no": pick.round_no,
                    "category": pick.asset.category,
                    "entity_id": pick.asset.entity_id,
                    "name": pick.asset.name,
                    "decision_value": pick.asset.value,
                    "actual_points": pick.asset.metadata["actual_points"],
                })

    return pd.DataFrame(detail_rows), pd.DataFrame(pick_rows)


def run_focal_candidate_comparison(
    universe: pd.DataFrame,
    *,
    gm_count: int,
    roster_config: RosterConfig,
    strategy_names: tuple[str, ...] = ("vorp", "tier_vorp"),
    strategy_kwargs: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test one candidate-informed GM at a time against V0-informed opponents."""

    if gm_count < 2:
        raise ValueError("gm_count must be at least 2")
    if not strategy_names:
        raise ValueError("strategy_names cannot be empty")
    strategy_kwargs = dict(strategy_kwargs or {})

    counts = universe["category"].value_counts().to_dict()
    for category, slots in roster_config.counts.items():
        required = gm_count * int(slots)
        if counts.get(category, 0) < required:
            raise ValueError(
                f"Projection universe has only {counts.get(category, 0)} {category}; "
                f"need {required}"
            )

    assets = _decision_assets(universe, "v0_projected_points")
    detail_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []

    def record_focal(
        *,
        result: DraftResult,
        model: str,
        opponent_model: str,
        strategy_name: str,
        focal_slot: int,
        value_column: str,
    ) -> None:
        actual_totals = {
            gm_id: float(sum(asset.metadata["actual_points"] for asset in roster))
            for gm_id, roster in result.rosters.items()
        }
        focal_actual = actual_totals[focal_slot]
        field = [
            value for gm_id, value in actual_totals.items() if gm_id != focal_slot
        ]
        roster = result.rosters[focal_slot]
        detail_rows.append({
            "projection_model": model,
            "opponent_projection_model": opponent_model,
            "strategy": strategy_name,
            "draft_slot": focal_slot,
            "projected_roster_points": float(
                sum(asset.metadata[value_column] for asset in roster)
            ),
            "actual_roster_points": focal_actual,
            "actual_rank": _competition_rank(actual_totals, focal_slot),
            "gap_to_winner": max(actual_totals.values()) - focal_actual,
            "margin_vs_field_mean": focal_actual - fmean(field),
        })
        focal_picks = [pick for pick in result.picks if pick.gm_id == focal_slot]
        for pick in focal_picks:
            pick_rows.append({
                "projection_model": model,
                "opponent_projection_model": opponent_model,
                "strategy": strategy_name,
                "draft_slot": focal_slot,
                "overall_pick": pick.overall_pick,
                "round_no": pick.round_no,
                "category": pick.asset.category,
                "entity_id": pick.asset.entity_id,
                "name": pick.asset.name,
                "decision_value": pick.asset.metadata[value_column],
                "actual_points": pick.asset.metadata["actual_points"],
            })

    for strategy_name in strategy_names:
        baseline_strategies = {
            gm_id: build_strategy(strategy_name, **strategy_kwargs)
            for gm_id in range(1, gm_count + 1)
        }
        baseline = run_draft(assets, gm_count, roster_config, baseline_strategies)
        for focal_slot in range(1, gm_count + 1):
            record_focal(
                result=baseline,
                model="V0",
                opponent_model="V0",
                strategy_name=strategy_name,
                focal_slot=focal_slot,
                value_column="v0_projected_points",
            )

        for focal_slot in range(1, gm_count + 1):
            mixed_strategies = {
                gm_id: build_strategy(strategy_name, **strategy_kwargs)
                for gm_id in range(1, gm_count + 1)
            }
            mixed_strategies[focal_slot] = _ProjectionViewStrategy(
                mixed_strategies[focal_slot],
                value_column="candidate_projected_points",
                gm_count=gm_count,
            )
            mixed = run_draft(assets, gm_count, roster_config, mixed_strategies)
            record_focal(
                result=mixed,
                model="V1_WEIGHTED_GP_CANDIDATE",
                opponent_model="V0",
                strategy_name=strategy_name,
                focal_slot=focal_slot,
                value_column="candidate_projected_points",
            )

    return pd.DataFrame(detail_rows), pd.DataFrame(pick_rows)


def summarize_projection_drafts(details: pd.DataFrame) -> pd.DataFrame:
    work = details.assign(
        win=details["actual_rank"].eq(1).astype(float),
        top3=details["actual_rank"].le(3).astype(float),
    )
    return (
        work.groupby(["projection_model", "strategy"], as_index=False)
        .agg(
            trials=("draft_slot", "size"),
            mean_actual_points=("actual_roster_points", "mean"),
            median_actual_points=("actual_roster_points", "median"),
            mean_rank=("actual_rank", "mean"),
            win_rate=("win", "mean"),
            top3_rate=("top3", "mean"),
            mean_gap_to_winner=("gap_to_winner", "mean"),
            mean_margin_vs_field=("margin_vs_field_mean", "mean"),
        )
        .sort_values(["strategy", "projection_model"])
        .reset_index(drop=True)
    )


def paired_candidate_deltas(details: pd.DataFrame) -> pd.DataFrame:
    values = details.pivot(
        index=["strategy", "draft_slot"],
        columns="projection_model",
        values=["actual_roster_points", "actual_rank", "gap_to_winner"],
    )
    required = {"V0", "V1_WEIGHTED_GP_CANDIDATE"}
    if not required <= set(values.columns.get_level_values("projection_model")):
        raise ValueError("details must contain V0 and V1_WEIGHTED_GP_CANDIDATE")
    candidate = "V1_WEIGHTED_GP_CANDIDATE"
    output = pd.DataFrame({
        "strategy": values.index.get_level_values("strategy"),
        "draft_slot": values.index.get_level_values("draft_slot"),
        "v0_actual_points": values[("actual_roster_points", "V0")].to_numpy(),
        "candidate_actual_points": values[("actual_roster_points", candidate)].to_numpy(),
        "v0_actual_rank": values[("actual_rank", "V0")].to_numpy(),
        "candidate_actual_rank": values[("actual_rank", candidate)].to_numpy(),
        "v0_gap_to_winner": values[("gap_to_winner", "V0")].to_numpy(),
        "candidate_gap_to_winner": values[("gap_to_winner", candidate)].to_numpy(),
    })
    output["actual_points_delta"] = (
        output["candidate_actual_points"] - output["v0_actual_points"]
    )
    output["actual_rank_delta"] = (
        output["candidate_actual_rank"] - output["v0_actual_rank"]
    )
    output["gap_to_winner_delta"] = (
        output["candidate_gap_to_winner"] - output["v0_gap_to_winner"]
    )
    return output.sort_values(["strategy", "draft_slot"]).reset_index(drop=True)


def summarize_paired_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby("strategy", as_index=False)
        .agg(
            trials=("draft_slot", "size"),
            mean_actual_points_delta=("actual_points_delta", "mean"),
            median_actual_points_delta=("actual_points_delta", "median"),
            slots_improved=("actual_points_delta", lambda values: int((values > 0).sum())),
            slots_tied=("actual_points_delta", lambda values: int((values == 0).sum())),
            slots_worsened=("actual_points_delta", lambda values: int((values < 0).sum())),
            mean_actual_rank_delta=("actual_rank_delta", "mean"),
            mean_gap_to_winner_delta=("gap_to_winner_delta", "mean"),
        )
        .sort_values("strategy")
        .reset_index(drop=True)
    )


def draft_category_totals(picks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate projected and realised roster value by asset category."""

    return (
        picks.groupby(
            ["projection_model", "strategy", "draft_slot", "category"],
            as_index=False,
        )
        .agg(
            assets_drafted=("entity_id", "size"),
            decision_value=("decision_value", "sum"),
            actual_points=("actual_points", "sum"),
        )
        .sort_values(["strategy", "draft_slot", "category", "projection_model"])
        .reset_index(drop=True)
    )


def paired_category_deltas(category_totals: pd.DataFrame) -> pd.DataFrame:
    """Pair candidate and V0 category results for every strategy and slot."""

    values = category_totals.pivot(
        index=["strategy", "draft_slot", "category"],
        columns="projection_model",
        values="actual_points",
    )
    required = {"V0", "V1_WEIGHTED_GP_CANDIDATE"}
    if not required <= set(values.columns):
        raise ValueError("category totals must contain V0 and V1_WEIGHTED_GP_CANDIDATE")
    output = values.reset_index().rename(columns={
        "V0": "v0_actual_points",
        "V1_WEIGHTED_GP_CANDIDATE": "candidate_actual_points",
    })
    output["actual_points_delta"] = (
        output["candidate_actual_points"] - output["v0_actual_points"]
    )
    return output.sort_values(
        ["strategy", "category", "draft_slot"]
    ).reset_index(drop=True)


def summarize_category_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["strategy", "category"], as_index=False)
        .agg(
            slots=("draft_slot", "size"),
            mean_actual_points_delta=("actual_points_delta", "mean"),
            median_actual_points_delta=("actual_points_delta", "median"),
            total_actual_points_delta=("actual_points_delta", "sum"),
            slots_improved=("actual_points_delta", lambda values: int((values > 0).sum())),
            slots_tied=("actual_points_delta", lambda values: int((values == 0).sum())),
            slots_worsened=("actual_points_delta", lambda values: int((values < 0).sum())),
        )
        .sort_values(["strategy", "category"])
        .reset_index(drop=True)
    )
