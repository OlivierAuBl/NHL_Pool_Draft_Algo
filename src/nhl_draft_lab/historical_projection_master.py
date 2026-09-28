from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_LABELS = {
    "espn": "ESPN",
    "fantasy_unknown": "FantasyUnknown",
    "hashtag": "Hashtag",
    "nhl_com": "NHL.com",
    "poolpro": "PoolPro",
    "scott_cullen": "ScottCullen",
}

DERIVED_GOALIE_SOURCES = {
    "local_nhl_com": "nhl_com",
    "local_poolpro": "poolpro",
    "local_scott_cullen": "scott_cullen",
}


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _history_identity_map(history: pd.DataFrame) -> dict[tuple[str, str], int]:
    players = history[history["category"].isin(["F", "D", "G"])].copy()
    players["normalized_name"] = players["name"].map(normalize_name)
    candidates = (
        players.groupby(["category", "normalized_name"])["entity_id"]
        .agg(lambda values: sorted(set(int(value) for value in values)))
        .to_dict()
    )
    return {
        key: ids[0]
        for key, ids in candidates.items()
        if len(ids) == 1 and key[1]
    }


def _resolve_player_id(
    row: pd.Series,
    identity_map: dict[tuple[str, str], int],
) -> tuple[int | None, str]:
    numeric = pd.to_numeric(pd.Series([row.get("nhl_player_id")]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return int(numeric), "nhl_id"
    key = (str(row["category"]), normalize_name(row.get("canonical_name")))
    matched = identity_map.get(key)
    return (matched, "exact_history_name") if matched is not None else (None, "unresolved")


def _read_inputs(
    normalized_db: Path,
    history_db: Path,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(normalized_db) as connection:
        projections = pd.read_sql_query(
            """
            select source, entity_id, nhl_player_id, canonical_name, category,
                   team_code, record_status, points, goalie_components_points
            from v_projections
            where season = ? and variant = 'primary' and category in ('F','D','G')
            """,
            connection,
            params=[str(season)],
        )
        derived_goalies = pd.read_sql_query(
            """
            select s.source, o.entity_id, e.nhl_player_id, e.canonical_name,
                   o.category, coalesce(o.team_code, e.team_code) team_code,
                   sv.value projection_value
            from snapshots s
            join observations o on o.snapshot_id = s.snapshot_id
            join entities e on e.entity_id = o.entity_id
            join stat_values sv on sv.observation_id = o.observation_id
            where s.season = ? and s.kind = 'derived' and o.category = 'G'
              and sv.metric = 'pool_points_reported' and sv.status = 'valid'
            """,
            connection,
            params=[str(season)],
        )
        teams = pd.read_sql_query(
            "select entity_id, canonical_name, team_code from entities where kind = 'team'",
            connection,
        )
    with sqlite3.connect(history_db) as connection:
        history = pd.read_sql_query("select * from rankings", connection)
    return projections, derived_goalies, teams, history


def _projection_rows(
    projections: pd.DataFrame,
    derived_goalies: pd.DataFrame,
) -> pd.DataFrame:
    direct = projections[projections["source"].isin(SOURCE_LABELS)].copy()
    direct["projection_value"] = direct["points"]
    goalie_mask = direct["category"].eq("G")
    direct.loc[goalie_mask, "projection_value"] = direct.loc[
        goalie_mask, "goalie_components_points"
    ]
    direct = direct[[
        "source", "entity_id", "nhl_player_id", "canonical_name", "category",
        "team_code", "projection_value",
    ]]

    derived = derived_goalies[
        derived_goalies["source"].isin(DERIVED_GOALIE_SOURCES)
    ].copy()
    derived["source"] = derived["source"].map(DERIVED_GOALIE_SOURCES)
    combined = pd.concat([direct, derived[direct.columns]], ignore_index=True)
    combined["projection_value"] = pd.to_numeric(
        combined["projection_value"], errors="coerce"
    )
    return combined.dropna(subset=["projection_value"]).copy()


def _team_code_map(teams: pd.DataFrame) -> dict[str, str]:
    result = {
        normalize_name(row.canonical_name): str(row.team_code)
        for row in teams.itertuples(index=False)
    }
    result[normalize_name("Utah Hockey Club")] = "UTA"
    result[normalize_name("Utah Mammoth")] = "UTA"
    return result


def build_historical_projection_master(
    *,
    normalized_db: Path,
    history_db: Path,
    season: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    projections, derived_goalies, teams, history = _read_inputs(
        normalized_db, history_db, season
    )
    history["season_id"] = pd.to_numeric(history["season_id"], errors="coerce")
    history["entity_id"] = pd.to_numeric(history["entity_id"], errors="coerce")
    projection_rows = _projection_rows(projections, derived_goalies)
    identity_map = _history_identity_map(history)

    resolved_ids: list[int | None] = []
    match_methods: list[str] = []
    for _, row in projection_rows.iterrows():
        resolved, method = _resolve_player_id(row, identity_map)
        resolved_ids.append(resolved)
        match_methods.append(method)
    projection_rows["resolved_nhl_id"] = resolved_ids
    projection_rows["match_method"] = match_methods

    identity_audit = projection_rows[
        projection_rows["resolved_nhl_id"].isna()
    ][[
        "source", "category", "canonical_name", "entity_id", "projection_value",
        "match_method",
    ]].copy()
    identity_audit["reason"] = "no NHL ID and no unique exact-name match in history"
    resolved = projection_rows.dropna(subset=["resolved_nhl_id"]).copy()
    resolved["resolved_nhl_id"] = resolved["resolved_nhl_id"].astype(int)
    resolved["source_label"] = resolved["source"].map(SOURCE_LABELS)

    duplicate_counts = (
        resolved.groupby(["resolved_nhl_id", "category", "source_label"])
        .size()
        .rename("duplicate_rows")
        .reset_index()
    )
    conflicts = duplicate_counts[duplicate_counts["duplicate_rows"] > 1]
    if not conflicts.empty:
        conflict_details = resolved.merge(
            conflicts, on=["resolved_nhl_id", "category", "source_label"], how="inner"
        )
        conflict_details = conflict_details.assign(
            reason="multiple rows resolved to the same source/player; median used"
        )
        identity_audit = pd.concat(
            [identity_audit, conflict_details[identity_audit.columns]], ignore_index=True
        )

    wide = resolved.pivot_table(
        index=["resolved_nhl_id", "category"],
        columns="source_label",
        values="projection_value",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None
    for label in SOURCE_LABELS.values():
        if label not in wide:
            wide[label] = pd.NA
        wide = wide.rename(columns={label: f"src__{label}"})
    source_metadata = (
        resolved.sort_values(["resolved_nhl_id", "category", "source_label"])
        .drop_duplicates(["resolved_nhl_id", "category"])
        .set_index(["resolved_nhl_id", "category"])[["canonical_name", "team_code"]]
    )

    current = history[
        history["season_id"].eq(season) & history["category"].isin(["F", "D", "G"])
    ].copy()
    current = current.drop_duplicates(["entity_id", "category"])
    current_lookup = current.set_index(["entity_id", "category"])
    latest_names = (
        history[history["category"].isin(["F", "D", "G"])]
        .sort_values("season_id")
        .drop_duplicates(["entity_id", "category"], keep="last")
        .set_index(["entity_id", "category"])
    )

    player_rows: list[dict[str, object]] = []
    source_columns = [f"src__{label}" for label in SOURCE_LABELS.values()]
    for _, row in wide.iterrows():
        key = (float(row["resolved_nhl_id"]), row["category"])
        actual = current_lookup.loc[key] if key in current_lookup.index else None
        latest = latest_names.loc[key] if key in latest_names.index else None
        source_meta = source_metadata.loc[key]
        source_team = "" if pd.isna(source_meta["team_code"]) else str(source_meta["team_code"])
        values = {
            column: pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            for column in source_columns
        }
        available = [float(value) for value in values.values() if not pd.isna(value)]
        if not available:
            continue
        player_rows.append(
            {
                "season": str(season),
                "category": row["category"],
                "NHLID": int(row["resolved_nhl_id"]),
                "FullName": str(
                    actual["name"]
                    if actual is not None
                    else latest["name"]
                    if latest is not None
                    else source_meta["canonical_name"]
                ),
                "Team": str(
                    actual["nhl_team"]
                    if actual is not None
                    else latest["nhl_team"]
                    if latest is not None
                    else source_team
                ),
                "team_key": str(
                    actual["nhl_team"]
                    if actual is not None
                    else latest["nhl_team"]
                    if latest is not None
                    else source_team
                ),
                "actual_points": float(actual["pool_points_raw"]) if actual is not None else 0.0,
                "projection_median": float(pd.Series(available).median()),
                **values,
            }
        )

    team_codes = _team_code_map(teams)
    actual_teams = history[
        history["season_id"].eq(season) & history["category"].eq("T")
    ].copy()
    previous_teams = history[
        history["season_id"].eq(season - 10001) & history["category"].eq("T")
    ].copy()
    previous_by_code: dict[str, float] = {}
    for row in previous_teams.itertuples(index=False):
        code = team_codes.get(normalize_name(row.name))
        if code:
            previous_by_code[code] = float(row.pool_points_raw)
    if "ARI" in previous_by_code:
        previous_by_code["UTA"] = previous_by_code["ARI"]
    default_team_projection = float(previous_teams["pool_points_raw"].median())

    team_rows: list[dict[str, object]] = []
    for row in actual_teams.itertuples(index=False):
        code = team_codes.get(normalize_name(row.name))
        if not code:
            raise ValueError(f"Cannot map historical team {row.name!r} to a team code")
        baseline = previous_by_code.get(code, default_team_projection)
        team_rows.append(
            {
                "season": str(season),
                "category": "T",
                "NHLID": pd.NA,
                "FullName": code,
                "Team": code,
                "team_key": code,
                "actual_points": float(row.pool_points_raw),
                "projection_median": baseline,
                **{column: pd.NA for column in source_columns},
            }
        )

    master = pd.DataFrame(player_rows + team_rows)
    master = master.sort_values(
        ["category", "projection_median"], ascending=[True, False]
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / f"projection_source_master_{season}.csv"
    audit_path = output_dir / f"projection_source_identity_audit_{season}.csv"
    master.to_csv(master_path, index=False)
    identity_audit.to_csv(audit_path, index=False)

    counts = master.groupby("category").size().rename("assets").reset_index()
    coverage_rows = []
    for source_column in source_columns:
        for category, group in master.groupby("category"):
            coverage_rows.append(
                {
                    "source": source_column.removeprefix("src__"),
                    "category": category,
                    "available": int(group[source_column].notna().sum()),
                    "universe": len(group),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    report = "\n".join(
        [
            f"# Construction du master de projections {season}",
            "",
            f"- Source normalisée: `{normalized_db.as_posix()}`",
            f"- Résultats réels: `{history_db.as_posix()}`",
            f"- Actifs: {len(master)}",
            f"- Identités non résolues ou conflits audités: {len(identity_audit)}",
            "- F/D: points NHL projetés.",
            "- G: `2 × W + OTL + 3 × SO`; les scores dérivés déjà consolidés sont utilisés pour NHL.com, Pool Pro et Scott Cullen.",
            "- Hashtag/G exclu: aucune projection OTL compatible n'est disponible.",
            "- T: aucune source ne contient des points NHL projetés complets; fallback commun sur les points réels de la saison précédente (Arizona transféré à Utah).",
            "- `FantasyUnknown`: fournisseur candidat DtZ dans la base normalisée, édition 2024-2025 non confirmée.",
            "- Les snapshots historiques n'ont pas de date de publication vérifiable (`strict_backtest_eligible = 0`); les résultats doivent donc être lus comme une analyse exploratoire.",
            "",
            "## Actifs par catégorie",
            "",
            counts.to_markdown(index=False),
            "",
            "## Couverture brute par source",
            "",
            coverage.to_markdown(index=False),
            "",
        ]
    )
    (output_dir / f"projection_source_master_report_{season}.md").write_text(
        report, encoding="utf-8"
    )
    return master, identity_audit
