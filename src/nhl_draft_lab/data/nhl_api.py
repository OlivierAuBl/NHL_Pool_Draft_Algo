from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nhl_draft_lab.config import (
    GAME_TYPE_REGULAR_SEASON,
    GOALIE_ASSIST_POINTS,
    GOALIE_GOAL_POINTS,
    GOALIE_OT_LOSS_POINTS,
    GOALIE_SHUTOUT_BONUS,
    GOALIE_WIN_POINTS,
    NHL_STATS_BASE,
    NORMALIZATION_FACTOR,
    SKATER_ASSIST_POINTS,
    SKATER_GOAL_POINTS,
    TEAM_OT_LOSS_POINTS,
    TEAM_WIN_POINTS,
)


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "nhl-draft-lab/0.1", "Accept": "application/json"})
    return session


def fetch_report(session: requests.Session, report_path: str, season_id: int) -> list[dict[str, Any]]:
    response = session.get(
        f"{NHL_STATS_BASE}/{report_path}",
        params={
            "limit": -1,
            "start": 0,
            "cayenneExp": f"seasonId={season_id} and gameTypeId={GAME_TYPE_REGULAR_SEASON}",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected NHL response for {report_path}, season {season_id}")
    return data


def _num(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, default)))


def last_completed_season_ids(n: int = 4, today: date | None = None) -> list[int]:
    today = today or date.today()
    latest_start_year = today.year - 1 if today.month >= 7 else today.year - 2
    return [int(f"{y}{y + 1}") for y in range(latest_start_year, latest_start_year - n, -1)]


def _skaters(raw: Iterable[dict[str, Any]], season_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in raw:
        gp = _int(row.get("gamesPlayed"))
        if gp <= 0:
            continue
        goals = _int(row.get("goals"))
        assists = _int(row.get("assists"))
        shots = _int(row.get("shots"))
        power_play_points = _int(row.get("ppPoints", row.get("powerPlayPoints")))
        position = str(row.get("positionCode") or "").upper()
        category = "D" if position == "D" else "F"
        pool_points = SKATER_GOAL_POINTS * goals + SKATER_ASSIST_POINTS * assists
        rows.append({
            "season_id": season_id,
            "category": category,
            "entity_id": _int(row.get("playerId")),
            "name": row.get("skaterFullName") or row.get("lastName") or "Unknown",
            "nhl_team": row.get("teamAbbrevs") or "",
            "position": position,
            "games_played": gp,
            "games_started": 0,
            "goals": goals,
            "assists": assists,
            "shots": shots,
            "shooting_pct": _num(row.get("shootingPct")),
            "power_play_points": power_play_points,
            "wins": 0,
            "losses": 0,
            "ot_losses": 0,
            "shutouts": 0,
            "pool_points_raw": pool_points,
        })
    return pd.DataFrame(rows)


def _goalies(raw: Iterable[dict[str, Any]], season_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in raw:
        gp = _int(row.get("gamesPlayed"))
        if gp <= 0:
            continue
        goals = _int(row.get("goals"))
        assists = _int(row.get("assists"))
        wins = _int(row.get("wins"))
        losses = _int(row.get("losses"))
        ot_losses = _int(row.get("otLosses"))
        shutouts = _int(row.get("shutouts"))
        games_started = (
            _int(row.get("gamesStarted"))
            if row.get("gamesStarted") not in (None, "")
            else pd.NA
        )
        pool_points = (
            GOALIE_GOAL_POINTS * goals
            + GOALIE_ASSIST_POINTS * assists
            + GOALIE_OT_LOSS_POINTS * ot_losses
            + GOALIE_WIN_POINTS * wins
            + GOALIE_SHUTOUT_BONUS * shutouts
        )
        rows.append({
            "season_id": season_id,
            "category": "G",
            "entity_id": _int(row.get("playerId")),
            "name": row.get("goalieFullName") or row.get("lastName") or "Unknown",
            "nhl_team": row.get("teamAbbrevs") or "",
            "position": "G",
            "games_played": gp,
            "games_started": games_started,
            "goals": goals,
            "assists": assists,
            "shots": 0,
            "shooting_pct": 0.0,
            "power_play_points": 0,
            "wins": wins,
            "losses": losses,
            "ot_losses": ot_losses,
            "shutouts": shutouts,
            "pool_points_raw": pool_points,
        })
    return pd.DataFrame(rows)


def _teams(raw: Iterable[dict[str, Any]], season_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in raw:
        gp = _int(row.get("gamesPlayed"))
        if gp <= 0:
            continue
        wins = _int(row.get("wins"))
        losses = _int(row.get("losses"))
        ot_losses = _int(row.get("otLosses"))
        pool_points = TEAM_WIN_POINTS * wins + TEAM_OT_LOSS_POINTS * ot_losses
        rows.append({
            "season_id": season_id,
            "category": "T",
            "entity_id": _int(row.get("teamId")),
            "name": row.get("teamFullName") or row.get("teamAbbrev") or "Unknown",
            "nhl_team": row.get("teamAbbrev") or row.get("teamAbbrevs") or "",
            "position": "TEAM",
            "games_played": gp,
            "games_started": 0,
            "goals": 0,
            "assists": 0,
            "shots": 0,
            "shooting_pct": 0.0,
            "power_play_points": 0,
            "wins": wins,
            "losses": losses,
            "ot_losses": ot_losses,
            "shutouts": 0,
            "pool_points_raw": pool_points,
        })
    return pd.DataFrame(rows)


def rank_season(df: pd.DataFrame) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for category, part in df.groupby("category", sort=False):
        part = part.copy()
        if category in {"F", "D"}:
            by, asc = ["pool_points_raw", "goals", "games_played", "name"], [False, False, True, True]
        elif category == "G":
            by, asc = ["pool_points_raw", "wins", "shutouts", "games_played", "name"], [False, False, False, True, True]
        else:
            by, asc = ["pool_points_raw", "wins", "games_played", "name"], [False, False, True, True]
        part = part.sort_values(by, ascending=asc, kind="mergesort").reset_index(drop=True)
        part.insert(2, "rank", range(1, len(part) + 1))
        out.append(part)
    result = pd.concat(out, ignore_index=True)
    result["normalization_factor"] = NORMALIZATION_FACTOR
    result["pool_points_84"] = result["pool_points_raw"].astype(float) * NORMALIZATION_FACTOR
    return result.sort_values(["category", "rank"]).reset_index(drop=True)


def fetch_season(season_id: int, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or build_session()
    logging.info("Fetching NHL season %s", season_id)
    raw_skaters = fetch_report(session, "skater/summary", season_id)
    raw_goalies = fetch_report(session, "goalie/summary", season_id)
    raw_teams = fetch_report(session, "team/summary", season_id)
    combined = pd.concat([
        _skaters(raw_skaters, season_id),
        _goalies(raw_goalies, season_id),
        _teams(raw_teams, season_id),
    ], ignore_index=True)
    if combined.empty:
        raise RuntimeError(f"No rows fetched for season {season_id}")
    return rank_season(combined)


def export_excel(rankings: pd.DataFrame, season_id: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {"F": "Forwards", "D": "Defense", "G": "Goalies", "T": "Teams"}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for category, sheet in names.items():
            rankings.loc[rankings["category"] == category].to_excel(writer, sheet_name=sheet, index=False)
