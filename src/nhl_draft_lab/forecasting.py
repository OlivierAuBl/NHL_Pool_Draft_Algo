from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from nhl_draft_lab.config import (
    GOALIE_ASSIST_POINTS,
    GOALIE_GOAL_POINTS,
    GOALIE_OT_LOSS_POINTS,
    GOALIE_SHUTOUT_BONUS,
    GOALIE_WIN_POINTS,
    TEAM_OT_LOSS_POINTS,
    TEAM_WIN_POINTS,
)


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the transparent V1.0 component baseline.

    Weights are ordered from most recent season to oldest season.  They are
    normalized for players with fewer seasons of history.
    """

    season_games: int = 82
    recency_weights: tuple[float, ...] = (0.60, 0.30, 0.10)

    def __post_init__(self) -> None:
        if self.season_games <= 0:
            raise ValueError("season_games must be positive")
        if not self.recency_weights or any(weight < 0 for weight in self.recency_weights):
            raise ValueError("recency_weights must contain non-negative values")
        if sum(self.recency_weights) <= 0:
            raise ValueError("at least one recency weight must be positive")


MANUAL_OVERRIDE_COLUMNS = {
    "projected_gp_override",
    "projected_ppg_override",
    "availability",
    "projected_start_share_override",
    "projected_win_rate_override",
    "projected_shutout_rate_override",
    "projected_otl_rate_override",
}


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _weighted(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = [(float(value), float(weight)) for value, weight in zip(values, weights)]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total_weight


def _history_with_weights(group: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    recent = group.sort_values("season_id", ascending=False).head(len(config.recency_weights)).copy()
    recent["_weight"] = config.recency_weights[: len(recent)]
    return recent


def _history_stddev(group: pd.DataFrame) -> float:
    if "pool_points_raw" not in group:
        return 0.0
    values = pd.to_numeric(group["pool_points_raw"], errors="coerce").dropna()
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def _entity_id(value: object) -> str:
    """Keep identifiers stable across SQLite and CSV numeric inference."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value).strip()


def _name_key(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().casefold()


def _identity(group: pd.DataFrame) -> dict[str, object]:
    latest = group.sort_values("season_id", ascending=False).iloc[0]
    return {
        "entity_id": _entity_id(latest["entity_id"]),
        "name": str(latest["name"]),
        "category": str(latest["category"]).upper(),
        "nhl_team": "" if pd.isna(latest.get("nhl_team")) else str(latest.get("nhl_team")),
        "history_seasons": int(group["season_id"].nunique()),
        "latest_history_season": int(group["season_id"].max()),
        "career_gp_observed": float(pd.to_numeric(group["games_played"], errors="coerce").fillna(0).sum()),
    }


def _project_skater(group: pd.DataFrame, config: ForecastConfig) -> dict[str, object]:
    recent = _history_with_weights(group, config)
    gp = pd.to_numeric(recent["games_played"], errors="coerce").fillna(0).clip(lower=0)
    historical_schedule = _series(recent, "_season_games_observed", config.season_games)
    points = _series(recent, "goals") + _series(recent, "assists")
    ppg = points.div(gp.where(gp > 0)).fillna(0)
    weights = recent["_weight"]
    projected_ppg = _weighted(ppg, weights)
    availability_history = gp.div(historical_schedule.where(historical_schedule > 0)).clip(upper=1.0)
    projected_availability = _weighted(availability_history, weights)
    projected_gp = config.season_games * projected_availability

    pp_points = _series(recent, "power_play_points")
    shots = _series(recent, "shots")
    goals = _series(recent, "goals")

    result = _identity(group)
    result.update({
        "projection_model": "v1_recency_components",
        "projected_ppg": projected_ppg,
        "projected_gp": projected_gp,
        "projected_points": projected_ppg * projected_gp,
        "historical_pp_points_per_game": _weighted(pp_points.div(gp.where(gp > 0)).fillna(0), weights),
        "historical_shooting_pct": _weighted(goals.div(shots.where(shots > 0)).fillna(0), weights),
        "projected_starts": pd.NA,
        "projected_start_share": pd.NA,
        "availability": projected_availability,
        "projected_win_rate": pd.NA,
        "projected_shutout_rate": pd.NA,
        "projected_otl_rate": pd.NA,
        "projected_goal_rate": pd.NA,
        "projected_assist_rate": pd.NA,
        "starts_source": pd.NA,
        "stddev_points": _history_stddev(group),
    })
    return result


def _project_goalie(group: pd.DataFrame, config: ForecastConfig) -> dict[str, object]:
    recent = _history_with_weights(group, config)
    gp = pd.to_numeric(recent["games_played"], errors="coerce").fillna(0).clip(lower=0)
    if "games_started" in recent and pd.to_numeric(recent["games_started"], errors="coerce").notna().any():
        raw_starts = pd.to_numeric(recent["games_started"], errors="coerce")
        starts = raw_starts.fillna(gp).clip(lower=0)
        starts_source = "games_started" if raw_starts.notna().all() else "mixed_games_started_and_gp"
    else:
        starts = gp
        starts_source = "games_played_fallback"

    weights = recent["_weight"]
    historical_schedule = _series(recent, "_season_games_observed", config.season_games)
    start_share = min(
        1.0,
        _weighted(starts.div(historical_schedule.where(historical_schedule > 0)).fillna(0), weights),
    )
    projected_starts = config.season_games * start_share

    def rate(column: str) -> float:
        values = _series(recent, column)
        return _weighted(values.div(starts.where(starts > 0)).fillna(0), weights)

    win_rate = rate("wins")
    shutout_rate = rate("shutouts")
    otl_rate = rate("ot_losses")
    goal_rate = rate("goals")
    assist_rate = rate("assists")
    projected_points = projected_starts * (
        GOALIE_WIN_POINTS * win_rate
        + GOALIE_SHUTOUT_BONUS * shutout_rate
        + GOALIE_OT_LOSS_POINTS * otl_rate
        + GOALIE_GOAL_POINTS * goal_rate
        + GOALIE_ASSIST_POINTS * assist_rate
    )

    result = _identity(group)
    result.update({
        "projection_model": "v1_recency_components",
        "projected_ppg": pd.NA,
        "projected_gp": projected_starts,
        "projected_points": projected_points,
        "historical_pp_points_per_game": pd.NA,
        "historical_shooting_pct": pd.NA,
        "projected_starts": projected_starts,
        "projected_start_share": start_share,
        # V1.0 does not pretend that injuries can be identified from starts.
        "availability": 1.0,
        "projected_win_rate": win_rate,
        "projected_shutout_rate": shutout_rate,
        "projected_otl_rate": otl_rate,
        "projected_goal_rate": goal_rate,
        "projected_assist_rate": assist_rate,
        "starts_source": starts_source,
        "stddev_points": _history_stddev(group),
    })
    return result


def _project_team(group: pd.DataFrame, config: ForecastConfig) -> dict[str, object]:
    recent = _history_with_weights(group, config)
    gp = pd.to_numeric(recent["games_played"], errors="coerce").fillna(0).clip(lower=0)
    weights = recent["_weight"]

    def rate(column: str) -> float:
        values = _series(recent, column)
        return _weighted(values.div(gp.where(gp > 0)).fillna(0), weights)

    win_rate = rate("wins")
    otl_rate = rate("ot_losses")
    result = _identity(group)
    result.update({
        "projection_model": "v1_recency_components",
        "projected_ppg": pd.NA,
        "projected_gp": float(config.season_games),
        "projected_points": config.season_games * (
            TEAM_WIN_POINTS * win_rate + TEAM_OT_LOSS_POINTS * otl_rate
        ),
        "historical_pp_points_per_game": pd.NA,
        "historical_shooting_pct": pd.NA,
        "projected_starts": pd.NA,
        "projected_start_share": pd.NA,
        "availability": 1.0,
        "projected_win_rate": win_rate,
        "projected_shutout_rate": pd.NA,
        "projected_otl_rate": otl_rate,
        "projected_goal_rate": pd.NA,
        "projected_assist_rate": pd.NA,
        "starts_source": pd.NA,
        "stddev_points": _history_stddev(group),
    })
    return result


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"season_id", "category", "entity_id", "name", "games_played"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")
    clean = history.copy()
    clean["category"] = clean["category"].astype(str).str.upper()
    clean["entity_id"] = clean["entity_id"].map(_entity_id)
    clean = clean.loc[clean["category"].isin({"F", "D", "G", "T"})]
    if clean.empty:
        raise ValueError("History contains no supported F, D, G or T rows")
    return clean


def _apply_manual_context(projections: pd.DataFrame, manual: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    if "entity_id" not in manual and "name" not in manual:
        raise ValueError("Manual context must contain entity_id or name")
    context = manual.copy()
    if "entity_id" not in context:
        context["entity_id"] = pd.NA

    raw_ids = context["entity_id"]
    has_id = raw_ids.notna() & raw_ids.astype(str).str.strip().ne("")
    context.loc[has_id, "entity_id"] = raw_ids.loc[has_id].map(_entity_id)

    unresolved = ~has_id
    if unresolved.any():
        if "name" not in context:
            raise ValueError("Manual context rows without entity_id must contain name")
        projection_names = projections.assign(_name_key=projections["name"].map(_name_key))
        ambiguous = projection_names.loc[
            projection_names["_name_key"].duplicated(keep=False), "_name_key"
        ]
        requested = context.loc[unresolved, "name"].map(_name_key)
        if requested.isin(set(ambiguous)).any():
            names = sorted(context.loc[unresolved & requested.isin(set(ambiguous)), "name"].astype(str))
            raise ValueError(f"Manual context names are ambiguous: {names}")
        name_to_id = projection_names.set_index("_name_key")["entity_id"]
        resolved = requested.map(name_to_id)
        if resolved.isna().any():
            names = sorted(context.loc[resolved.index[resolved.isna()], "name"].astype(str))
            raise ValueError(f"Manual context contains unknown names: {names}")
        context.loc[unresolved, "entity_id"] = resolved.map(_entity_id)

    if context["entity_id"].duplicated().any():
        duplicated = sorted(context.loc[context["entity_id"].duplicated(), "entity_id"].unique())
        raise ValueError(f"Manual context contains duplicate entity_id values: {duplicated}")

    unknown_overrides = [column for column in context if column.endswith("_override") and column not in MANUAL_OVERRIDE_COLUMNS]
    if unknown_overrides:
        raise ValueError(f"Unknown manual override columns: {sorted(unknown_overrides)}")

    output = projections.merge(context, on="entity_id", how="left", suffixes=("", "_manual"))
    bounded = [
        "availability",
        "projected_start_share_override",
        "projected_win_rate_override",
        "projected_shutout_rate_override",
        "projected_otl_rate_override",
    ]
    for column in bounded:
        if column not in context:
            continue
        numeric = pd.to_numeric(context[column], errors="coerce")
        invalid = numeric.notna() & ~numeric.between(0, 1)
        if invalid.any():
            raise ValueError(f"{column} must be between 0 and 1")

    for index, row in output.iterrows():
        category = row["category"]
        if category in {"F", "D"}:
            ppg = pd.to_numeric(pd.Series([row.get("projected_ppg_override")]), errors="coerce").iloc[0]
            gp_override = pd.to_numeric(pd.Series([row.get("projected_gp_override")]), errors="coerce").iloc[0]
            availability = pd.to_numeric(
                pd.Series([row.get("availability_manual")]), errors="coerce"
            ).iloc[0]
            if not pd.isna(ppg):
                output.at[index, "projected_ppg"] = max(0.0, float(ppg))
            if not pd.isna(gp_override):
                output.at[index, "projected_gp"] = min(config.season_games, max(0.0, float(gp_override)))
                output.at[index, "availability"] = float(output.at[index, "projected_gp"]) / config.season_games
            elif not pd.isna(availability):
                output.at[index, "availability"] = float(availability)
                output.at[index, "projected_gp"] = config.season_games * float(availability)
            output.at[index, "projected_points"] = (
                float(output.at[index, "projected_ppg"]) * float(output.at[index, "projected_gp"])
            )
        elif category == "G":
            availability = pd.to_numeric(pd.Series([row.get("availability_manual", row.get("availability"))]), errors="coerce").iloc[0]
            availability = 1.0 if pd.isna(availability) else float(availability)
            share_override = pd.to_numeric(pd.Series([row.get("projected_start_share_override")]), errors="coerce").iloc[0]
            share = float(row["projected_start_share"]) if pd.isna(share_override) else float(share_override)
            win_rate = _override_or_current(row, "projected_win_rate")
            shutout_rate = _override_or_current(row, "projected_shutout_rate")
            otl_rate = _override_or_current(row, "projected_otl_rate")
            goal_rate = float(row["projected_goal_rate"])
            assist_rate = float(row["projected_assist_rate"])
            starts = config.season_games * share * availability
            output.at[index, "availability"] = availability
            output.at[index, "projected_start_share"] = share
            output.at[index, "projected_starts"] = starts
            output.at[index, "projected_gp"] = starts
            output.at[index, "projected_win_rate"] = win_rate
            output.at[index, "projected_shutout_rate"] = shutout_rate
            output.at[index, "projected_otl_rate"] = otl_rate
            output.at[index, "projected_points"] = starts * (
                GOALIE_WIN_POINTS * win_rate
                + GOALIE_SHUTOUT_BONUS * shutout_rate
                + GOALIE_OT_LOSS_POINTS * otl_rate
                + GOALIE_GOAL_POINTS * goal_rate
                + GOALIE_ASSIST_POINTS * assist_rate
            )
    return output


def _override_or_current(row: pd.Series, column: str) -> float:
    override = pd.to_numeric(pd.Series([row.get(f"{column}_override")]), errors="coerce").iloc[0]
    return float(row[column]) if pd.isna(override) else float(override)


def project_v1(
    history: pd.DataFrame,
    manual_context: pd.DataFrame | None = None,
    config: ForecastConfig | None = None,
) -> pd.DataFrame:
    """Build V1.0 projections while keeping production and volume separate."""

    config = config or ForecastConfig()
    clean = _validate_history(history)
    team_rows = clean.loc[clean["category"] == "T", ["season_id", "games_played"]]
    observed_schedule = (
        team_rows.groupby("season_id")["games_played"].max().to_dict()
        if not team_rows.empty
        else {}
    )
    clean["_season_games_observed"] = (
        clean["season_id"].map(observed_schedule).fillna(config.season_games)
    )
    rows: list[dict[str, object]] = []
    for (_, category), group in clean.groupby(["entity_id", "category"], sort=False):
        if category in {"F", "D"}:
            rows.append(_project_skater(group, config))
        elif category == "G":
            rows.append(_project_goalie(group, config))
        else:
            rows.append(_project_team(group, config))

    output = pd.DataFrame(rows)
    if manual_context is not None:
        output = _apply_manual_context(output, manual_context, config)
    return output.sort_values(["category", "projected_points"], ascending=[True, False]).reset_index(drop=True)
