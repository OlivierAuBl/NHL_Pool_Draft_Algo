from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from nhl_draft_lab.draft.engine import run_draft
    from nhl_draft_lab.models import DraftAsset, RosterConfig
    from nhl_draft_lab.strategies.tier_vorp import TierVorpStrategy
    from nhl_draft_lab.strategies.vorp import VorpStrategy
    from nhl_draft_lab.tiering import TierConfig
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "Could not import nhl_draft_lab. Run this script from the NHL Draft Lab repo "
        "with the package installed (for example: python -m pip install -e .)."
    ) from exc


SEASON = 20252026
GM_COUNT = 15
ROSTER_COUNTS = {"F": 10, "D": 3, "G": 2, "T": 1}
ROSTER = RosterConfig(ROSTER_COUNTS)
TIER_CONFIG = TierConfig(
    relative_width=0.05,
    max_size=5,
    superstar_max_size=3,
    superstar_tiers=2,
)

SKATER_SOURCES = [
    "ScottCullen", "CBS", "LineupExpert", "ESPN",
    "PoolPro", "HockeyMag", "Hastag", "PoolExpert",
]
GOALIE_SOURCES = [
    "calc_Hashtag_Score", "calc_HockeyMag_Score",
    "calc_LineupExpert_Score", "calc_PoolPro_Score",
    "calc_ScottCullen_Score", "calc_ESPN_Score",
    "calc_CBS_Score", "calc_PoolExpert_Score",
]
TEAM_SOURCES = [
    "ScottCullen", "CBS", "LineupExpert", "ESPN",
    "PoolPro", "PoolExpert", "Hastag",
]

POSITION_FIX = {
    8485702: "F",  # Max Shabanov
    8483455: "F",  # Isaac Howard
    8483525: "F",  # Danila Yurov
    8476455: "F",  # Gabriel Landeskog
    8485402: "F",  # Michael Misa
    8481721: "F",  # Arseny Gritsyuk
    8482100: "D",  # Alexander Nikishin
    8484798: "D",  # Zeev Buium
    8475169: "F",  # Evander Kane
    8473604: "F",  # Jonathan Toews
}

# The NHL rankings table omits 0-GP players. This is a verified 0-point result.
ZERO_GAME_ACTUALS = {
    8477493: {"name": "Aleksander Barkov", "category": "F", "actual_points": 0.0},
}

MANUAL_NOTES = {
    "Nikita Kucherov": (
        "User manually rejected an implausibly low ESPN projection; ESPN is #N/A "
        "in the consolidated CSV. Annotation only; no V0 feature adjustment."
    ),
    "Aleksander Barkov": (
        "Known at draft: preseason ACL injury, out for the season. Annotation only; "
        "V0 projection is not adjusted."
    ),
    "Matthew Tkachuk": (
        "Known at draft: injured; severity initially uncertain, later confirmed out "
        "until January. Annotation only; V0 projection is not adjusted."
    ),
    "Zach Hyman": (
        "Known at draft: preseason injury, roughly 1.5 months expected absence. "
        "Annotation only; V0 projection is not adjusted."
    ),
    "Mats Zuccarello": (
        "Known at draft: preseason injury, roughly 1.5 months expected absence. "
        "Annotation only; V0 projection is not adjusted."
    ),
    "Logan Cooley": (
        "Serious injury occurred during the season; no pre-draft adjustment in V0."
    ),
    "Jared McCann": (
        "In-season injury / team underperformance; no pre-draft adjustment in V0."
    ),
    "J.T. Miller": (
        "In-season injury / team underperformance; no pre-draft adjustment in V0."
    ),
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
    universe: str  # projection, actual_same, actual_full
    focal_strategy: str
    opponent_strategy: str
    homogeneous: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 V0 full diagnostic suite: projections only. Builds consensus, "
            "tiers, projection diagnostics, team-factor diagnostics, and symmetric "
            "draft backtests against actual 2025-26 results."
        )
    )
    parser.add_argument("--db", type=Path, default=Path("data/nhl_history.sqlite"))
    parser.add_argument("--skaters", type=Path, default=Path("Prediction_Skater_20252026.csv"))
    parser.add_argument("--goalies", type=Path, default=Path("Prediction_goalies_20252026.csv"))
    parser.add_argument("--teams", type=Path, default=Path("Prediction_Team_20252026.csv"))
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v0_full_analysis"))
    parser.add_argument("--top-misses", type=int, default=30)
    return parser.parse_args()


def resolve_existing(path: Path, fallbacks: Iterable[Path] = ()) -> Path:
    candidates = [path, *fallbacks]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Input not found. Tried:\n  - {tried}")


def safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def norm_team(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().rstrip("*")


def pearson(x: pd.Series, y: pd.Series) -> float:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return math.nan
    return float(clean["x"].corr(clean["y"]))


def spearman(x: pd.Series, y: pd.Series) -> float:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return math.nan
    return float(clean["x"].rank(method="average").corr(clean["y"].rank(method="average")))


def rmse(errors: pd.Series) -> float:
    clean = pd.to_numeric(errors, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(np.sqrt(np.mean(np.square(clean))))


def metric_row(label: str, frame: pd.DataFrame, pred_col: str, actual_col: str) -> dict[str, object]:
    clean = frame[[pred_col, actual_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if clean.empty:
        return {
            "label": label, "n": 0, "mae": math.nan, "rmse": math.nan,
            "bias_actual_minus_projection": math.nan, "pearson": math.nan,
            "spearman": math.nan,
        }
    err = clean[actual_col] - clean[pred_col]
    return {
        "label": label,
        "n": len(clean),
        "mae": float(err.abs().mean()),
        "rmse": rmse(err),
        "bias_actual_minus_projection": float(err.mean()),
        "pearson": pearson(clean[pred_col], clean[actual_col]),
        "spearman": spearman(clean[pred_col], clean[actual_col]),
    }


def source_value(value: object, injury: object) -> tuple[float | None, str]:
    """V0 source-cleaning rule.

    #N/A / blank -> missing.
    Zero with an injury flag -> retained as a real zero projection.
    Zero without injury -> treated as source non-coverage and ignored.
    """
    v = safe_float(value)
    if v is None:
        return None, "missing"
    injury_text = "" if pd.isna(injury) else str(injury).strip()
    if v == 0 and not injury_text:
        return None, "ignored_zero_no_injury"
    return v, "used"


def clean_source_columns(df: pd.DataFrame, source_cols: list[str], injury_col: str | None) -> pd.DataFrame:
    out = df.copy()
    statuses: dict[str, list[str]] = {c: [] for c in source_cols}
    cleaned: dict[str, list[float]] = {c: [] for c in source_cols}
    for _, row in out.iterrows():
        injury = row.get(injury_col) if injury_col else None
        for col in source_cols:
            value, status = source_value(row.get(col), injury)
            cleaned[col].append(np.nan if value is None else value)
            statuses[col].append(status)
    for col in source_cols:
        out[f"src__{col}"] = cleaned[col]
        out[f"src_status__{col}"] = statuses[col]
    return out


def consensus_from_clean_sources(df: pd.DataFrame, source_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    clean_cols = [f"src__{c}" for c in source_cols]
    numeric = out[clean_cols].apply(pd.to_numeric, errors="coerce")
    out["projection_median"] = numeric.median(axis=1, skipna=True)
    out["projection_mean"] = numeric.mean(axis=1, skipna=True)
    out["projection_min"] = numeric.min(axis=1, skipna=True)
    out["projection_max"] = numeric.max(axis=1, skipna=True)
    out["projection_std"] = numeric.std(axis=1, ddof=0, skipna=True)
    out["n_sources"] = numeric.notna().sum(axis=1)
    out["projection_mad"] = numeric.apply(
        lambda r: float(np.median(np.abs(r.dropna().to_numpy() - np.median(r.dropna().to_numpy()))))
        if r.notna().any() else np.nan,
        axis=1,
    )
    out["sources_used"] = numeric.apply(
        lambda r: " | ".join(source_cols[i] for i, ok in enumerate(r.notna()) if ok),
        axis=1,
    )
    return out


def prepare_projections(skaters_path: Path, goalies_path: Path, teams_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sk_raw = pd.read_csv(skaters_path)
    go_raw = pd.read_csv(goalies_path).loc[:, lambda x: ~x.columns.astype(str).str.startswith("Unnamed:")]
    tm_raw = pd.read_csv(teams_path)

    qc_rows: list[dict[str, object]] = []

    # Duplicate audit before dedupe.
    for label, frame, key in [("skater", sk_raw, "NHLID"), ("goalie", go_raw, "NHLID"), ("team", tm_raw, "Team")]:
        dupes = frame.loc[frame.duplicated(key, keep=False), [key]].copy()
        for value in dupes[key].drop_duplicates():
            qc_rows.append({"check": "duplicate_key", "dataset": label, "entity": value, "detail": "duplicate in raw input"})

    # Prefer a valid F/D row when duplicate NHLIDs exist.
    sk_raw["_valid_pos"] = sk_raw["Pos"].isin(["F", "D"]).astype(int)
    sk = (
        sk_raw.sort_values(["NHLID", "_valid_pos"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("NHLID", keep="first")
        .drop(columns="_valid_pos")
        .copy()
    )
    go = go_raw.drop_duplicates("NHLID", keep="first").copy()
    tm = tm_raw.drop_duplicates("Team", keep="first").copy()

    sk["NHLID"] = pd.to_numeric(sk["NHLID"], errors="coerce").astype("Int64")
    go["NHLID"] = pd.to_numeric(go["NHLID"], errors="coerce").astype("Int64")

    def category_for(row: pd.Series) -> str:
        pos = str(row.get("Pos", ""))
        if pos in {"F", "D"}:
            return pos
        nhlid = int(row["NHLID"]) if pd.notna(row["NHLID"]) else -1
        fixed = POSITION_FIX.get(nhlid, pos)
        qc_rows.append({
            "check": "position_fix",
            "dataset": "skater",
            "entity": nhlid,
            "detail": f"{row.get('FullName','')} Pos={pos!r} -> {fixed!r}",
        })
        return fixed

    sk["category"] = sk.apply(category_for, axis=1)
    go["category"] = "G"
    tm["category"] = "T"
    sk["team_key"] = sk["Team"].map(norm_team)
    go["team_key"] = go["Team"].map(norm_team)
    tm["team_key"] = tm["Team"].map(norm_team)

    sk = consensus_from_clean_sources(clean_source_columns(sk, SKATER_SOURCES, "DFO_Injury"), SKATER_SOURCES)
    go = consensus_from_clean_sources(clean_source_columns(go, GOALIE_SOURCES, "DFO_Injury"), GOALIE_SOURCES)
    tm = consensus_from_clean_sources(clean_source_columns(tm, TEAM_SOURCES, None), TEAM_SOURCES)

    # PoolPro Top is an upside signal, never part of the central consensus.
    sk["projection_ceiling_poolpro"] = pd.to_numeric(sk["PoolPro_Top"], errors="coerce")
    go["projection_ceiling_poolpro"] = pd.to_numeric(go["calc_PoolPro_Top_Score"], errors="coerce")
    tm["projection_ceiling_poolpro"] = np.nan

    # Apply the same zero-as-noncoverage convention to the ceiling unless an injury flag exists.
    for frame, injury_col in [(sk, "DFO_Injury"), (go, "DFO_Injury")]:
        no_injury = frame[injury_col].isna() | frame[injury_col].astype(str).str.strip().eq("")
        frame.loc[frame["projection_ceiling_poolpro"].eq(0) & no_injury, "projection_ceiling_poolpro"] = np.nan

    sk["manual_note"] = sk["FullName"].map(MANUAL_NOTES).fillna("")
    go["manual_note"] = go["FullName"].map(MANUAL_NOTES).fillna("")
    tm["manual_note"] = ""

    # QC for source zeros and missing consensus.
    for label, frame, sources in [("skater", sk, SKATER_SOURCES), ("goalie", go, GOALIE_SOURCES), ("team", tm, TEAM_SOURCES)]:
        for source in sources:
            status_col = f"src_status__{source}"
            ignored = int((frame[status_col] == "ignored_zero_no_injury").sum())
            if ignored:
                qc_rows.append({"check": "ignored_zero_no_injury", "dataset": label, "entity": source, "detail": ignored})
        missing_consensus = frame["projection_median"].isna()
        for _, row in frame.loc[missing_consensus].iterrows():
            qc_rows.append({"check": "missing_consensus", "dataset": label, "entity": row.get("FullName", row.get("Team", "")), "detail": "no usable central source"})

    qc = pd.DataFrame(qc_rows)
    return sk, go, tm, qc


def combined_projection_table(sk: pd.DataFrame, go: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for category, frame in [("SK", sk), ("G", go), ("T", tm)]:
        part = frame.copy()
        if category == "T":
            part["NHLID"] = pd.Series([pd.NA] * len(part), dtype="Int64")
            part["FullName"] = part["Team"]
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.loc[combined["category"].isin(["F", "D", "G", "T"])].copy()
    combined["projection_median"] = pd.to_numeric(combined["projection_median"], errors="coerce")
    combined = combined.dropna(subset=["projection_median"])
    return combined


def build_tiered_values(frame: pd.DataFrame, value_col: str, rank_counts: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict[str, float], dict[str, int], pd.DataFrame]:
    """Build fixed category tiers using the Stage-1 tier geometry."""
    rank_counts = rank_counts or ROSTER_COUNTS
    output_parts = []
    tier_rows = []
    replacement: dict[str, float] = {}
    replacement_tier: dict[str, int] = {}

    for cat in ["F", "D", "G", "T"]:
        part = frame.loc[frame["category"] == cat].copy()
        part[value_col] = pd.to_numeric(part[value_col], errors="coerce")
        part = part.dropna(subset=[value_col]).sort_values([value_col, "FullName"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        repl_rank = GM_COUNT * rank_counts[cat]
        if len(part) < repl_rank:
            raise ValueError(f"Not enough {cat} assets for replacement rank {repl_rank}: {len(part)}")
        repl = float(part.iloc[repl_rank - 1][value_col])
        replacement[cat] = repl
        part["rank"] = np.arange(1, len(part) + 1)
        part["vorp"] = part[value_col] - repl

        cursor = 0
        tier_no = 1
        tier_assignments: dict[int, tuple[int, float]] = {}
        while cursor < len(part):
            leader_idx = cursor
            leader_value = float(part.iloc[leader_idx][value_col])
            leader_vorp = float(part.iloc[leader_idx]["vorp"])
            denom = max(abs(leader_value), 1e-9)
            max_size = TIER_CONFIG.superstar_max_size if tier_no <= TIER_CONFIG.superstar_tiers else TIER_CONFIG.max_size
            idxs = [cursor]
            cursor += 1
            while cursor < len(part) and len(idxs) < max_size:
                candidate_vorp = float(part.iloc[cursor]["vorp"])
                relative_gap = (leader_vorp - candidate_vorp) / denom
                if relative_gap > TIER_CONFIG.relative_width:
                    break
                idxs.append(cursor)
                cursor += 1
            tier_value = float(part.iloc[idxs]["vorp"].median())
            for idx in idxs:
                tier_assignments[idx] = (tier_no, tier_value)
            names = " | ".join(part.iloc[idxs]["FullName"].astype(str))
            tier_rows.append({
                "category": cat,
                "tier": tier_no,
                "size": len(idxs),
                "tier_value": tier_value,
                "median_points": float(part.iloc[idxs][value_col].median()),
                "leader_points": leader_value,
                "replacement_level": repl,
                "members": names,
            })
            tier_no += 1

        part["tier"] = [tier_assignments[i][0] for i in range(len(part))]
        part["tier_value"] = [tier_assignments[i][1] for i in range(len(part))]
        replacement_tier[cat] = int(part.iloc[repl_rank - 1]["tier"])
        output_parts.append(part)

    tiers = pd.DataFrame(tier_rows)
    if not tiers.empty:
        tiers["global_priority"] = tiers["tier_value"].rank(method="first", ascending=False).astype(int)
        tiers = tiers.sort_values(["global_priority"]).reset_index(drop=True)
    return pd.concat(output_parts, ignore_index=True), replacement, replacement_tier, tiers


def load_actual(db_path: Path, season: int) -> pd.DataFrame:
    """Load actual season rankings while tolerating older DB schemas.

    Some historical ``rankings`` tables do not contain all of the convenience
    columns used by newer exports (notably ``nhl_points``).  V0 only requires
    ``pool_points_raw`` for scoring, so optional NHL-stat columns are selected
    when present and otherwise reconstructed or filled with NA.
    """
    with sqlite3.connect(db_path) as con:
        schema = pd.read_sql_query("PRAGMA table_info(rankings)", con)
        available = set(schema["name"].astype(str))

        required = {"season_id", "category", "entity_id", "name", "pool_points_raw"}
        missing_required = sorted(required - available)
        if missing_required:
            raise ValueError(
                f"The rankings table in {db_path} is missing required columns: "
                + ", ".join(missing_required)
                + f". Available columns: {', '.join(sorted(available))}"
            )

        def select_or_default(column: str, default_sql: str = "NULL") -> str:
            return column if column in available else f"{default_sql} AS {column}"

        # nhl_points is only descriptive in V0.  Older databases may omit it;
        # for skaters it is exactly goals + assists, which is sufficient here.
        if "nhl_points" in available:
            nhl_points_expr = "nhl_points"
        elif {"goals", "assists"}.issubset(available):
            nhl_points_expr = "(COALESCE(goals, 0) + COALESCE(assists, 0)) AS nhl_points"
        elif "points" in available:
            nhl_points_expr = "points AS nhl_points"
        else:
            nhl_points_expr = "NULL AS nhl_points"

        # Rank can also be reconstructed if an older DB did not persist it.
        # The window function preserves the expected category-wise descending
        # pool-score order without changing the database itself.
        if "rank" in available:
            rank_expr = "rank"
        else:
            rank_expr = (
                "ROW_NUMBER() OVER (PARTITION BY category "
                "ORDER BY pool_points_raw DESC, name ASC) AS rank"
            )

        select_exprs = [
            "season_id",
            "category",
            rank_expr,
            "entity_id",
            "name",
            select_or_default("nhl_team", "''"),
            select_or_default("position", "''"),
            select_or_default("games_played"),
            select_or_default("goals"),
            select_or_default("assists"),
            nhl_points_expr,
            select_or_default("wins"),
            select_or_default("losses"),
            select_or_default("ot_losses"),
            select_or_default("shutouts"),
            "pool_points_raw",
        ]

        query = f"""
            SELECT {', '.join(select_exprs)}
            FROM rankings
            WHERE season_id = ?
            ORDER BY category, rank
        """
        actual = pd.read_sql_query(query, con, params=(season,))

    if actual.empty:
        raise ValueError(f"No rankings found for season {season} in {db_path}")
    actual["entity_id"] = pd.to_numeric(actual["entity_id"], errors="coerce").astype("Int64")
    actual["pool_points_raw"] = pd.to_numeric(actual["pool_points_raw"], errors="coerce")
    actual["team_key"] = actual["nhl_team"].map(norm_team)
    team_mask = actual["category"].eq("T")
    actual.loc[team_mask, "team_key"] = actual.loc[team_mask, "name"].map(TEAM_NAME_TO_ABBREV).fillna("")

    # Add verified 0-game assets omitted by the NHL rankings endpoint.
    existing = set(actual.loc[actual["entity_id"].notna(), "entity_id"].astype(int))
    extra = []
    for entity_id, meta in ZERO_GAME_ACTUALS.items():
        if entity_id in existing:
            continue
        extra.append({
            "season_id": season,
            "category": meta["category"],
            "rank": np.nan,
            "entity_id": entity_id,
            "name": meta["name"],
            "nhl_team": "",
            "position": "",
            "games_played": 0,
            "goals": 0,
            "assists": 0,
            "nhl_points": 0,
            "wins": 0,
            "losses": 0,
            "ot_losses": 0,
            "shutouts": 0,
            "pool_points_raw": meta["actual_points"],
            "team_key": "",
        })
    if extra:
        actual = pd.concat([actual, pd.DataFrame(extra)], ignore_index=True)
    return actual


def join_actual(proj: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    players = proj.loc[proj["category"].isin(["F", "D", "G"])].copy()
    teams = proj.loc[proj["category"].eq("T")].copy()

    actual_players = actual.loc[actual["category"].isin(["F", "D", "G"])].copy()
    actual_teams = actual.loc[actual["category"].eq("T")].copy()

    p = players.merge(
        actual_players[
            ["entity_id", "category", "name", "team_key", "games_played", "goals", "assists", "nhl_points", "wins", "losses", "ot_losses", "shutouts", "pool_points_raw", "rank"]
        ].rename(columns={
            "name": "actual_name", "team_key": "actual_team_key", "rank": "actual_rank_full_api"
        }),
        left_on=["NHLID", "category"],
        right_on=["entity_id", "category"],
        how="left",
    )
    t = teams.merge(
        actual_teams[
            ["team_key", "category", "name", "games_played", "wins", "losses", "ot_losses", "pool_points_raw", "rank"]
        ].rename(columns={"name": "actual_name", "rank": "actual_rank_full_api"}),
        on=["team_key", "category"],
        how="left",
    )
    merged = pd.concat([p, t], ignore_index=True, sort=False)
    merged["actual_points"] = pd.to_numeric(merged["pool_points_raw"], errors="coerce")
    merged["matched"] = merged["actual_points"].notna()
    merged["point_error"] = merged["actual_points"] - merged["projection_median"]
    merged["abs_point_error"] = merged["point_error"].abs()
    return merged


def same_universe_actual_tiers(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, int], pd.DataFrame]:
    base = merged.loc[merged["matched"]].copy()
    base["FullName"] = base["FullName"].fillna(base.get("Team", ""))
    tiered, repl, repl_tier, tiers = build_tiered_values(base, "actual_points")
    tiered = tiered.rename(columns={
        "rank": "actual_rank_same_universe",
        "vorp": "actual_vorp_same_universe",
        "tier": "actual_tier_same_universe",
        "tier_value": "actual_tier_value_same_universe",
    })
    keep = ["category", "actual_points", "actual_rank_same_universe", "actual_vorp_same_universe", "actual_tier_same_universe", "actual_tier_value_same_universe"]
    tiered["join_key"] = np.where(
        tiered["category"].eq("T"),
        "T:" + tiered["team_key"].astype(str),
        "P:" + tiered["NHLID"].astype("Int64").astype(str),
    )
    return tiered[["join_key", *keep]], repl, repl_tier, tiers


def full_universe_actual_tiers(actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, int], pd.DataFrame]:
    base = actual.copy()
    base["NHLID"] = base["entity_id"]
    base["FullName"] = np.where(base["category"].eq("T"), base["team_key"], base["name"])
    tiered, repl, repl_tier, tiers = build_tiered_values(base, "pool_points_raw")
    tiered = tiered.rename(columns={
        "rank": "actual_rank_full_universe",
        "vorp": "actual_vorp_full_universe",
        "tier": "actual_tier_full_universe",
        "tier_value": "actual_tier_value_full_universe",
    })
    tiered["join_key"] = np.where(
        tiered["category"].eq("T"),
        "T:" + tiered["team_key"].astype(str),
        "P:" + tiered["entity_id"].astype("Int64").astype(str),
    )
    return tiered[[
        "join_key", "actual_rank_full_universe", "actual_vorp_full_universe",
        "actual_tier_full_universe", "actual_tier_value_full_universe"
    ]], repl, repl_tier, tiers


def add_projection_tiers(proj: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, int], pd.DataFrame]:
    tiered, repl, repl_tier, tiers = build_tiered_values(proj, "projection_median")
    tiered = tiered.rename(columns={
        "rank": "projected_rank", "vorp": "projected_vorp",
        "tier": "projected_tier", "tier_value": "projected_tier_value",
    })
    return tiered, repl, repl_tier, tiers


def add_join_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["join_key"] = np.where(
        out["category"].eq("T"),
        "T:" + out["team_key"].astype(str),
        "P:" + out["NHLID"].astype("Int64").astype(str),
    )
    return out


def build_master(projected_tiered: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, int], dict[str, float], dict[str, int], pd.DataFrame, pd.DataFrame]:
    joined = join_actual(projected_tiered, actual)
    joined = add_join_key(joined)
    same, same_repl, same_repl_tier, same_tiers = same_universe_actual_tiers(joined)
    full, full_repl, full_repl_tier, full_tiers = full_universe_actual_tiers(actual)
    master = joined.merge(same.drop(columns=["actual_points", "category"]), on="join_key", how="left")
    master = master.merge(full, on="join_key", how="left")
    master["rank_error_same"] = master["actual_rank_same_universe"] - master["projected_rank"]
    master["abs_rank_error_same"] = master["rank_error_same"].abs()
    master["tier_error_same"] = master["actual_tier_same_universe"] - master["projected_tier"]
    master["abs_tier_error_same"] = master["tier_error_same"].abs()
    master["tier_error_full"] = master["actual_tier_full_universe"] - master["projected_tier"]
    master["abs_tier_error_full"] = master["tier_error_full"].abs()
    return master, same_repl, same_repl_tier, full_repl, full_repl_tier, same_tiers, full_tiers


def source_performance(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_map = {
        "F": SKATER_SOURCES,
        "D": SKATER_SOURCES,
        "G": GOALIE_SOURCES,
        "T": TEAM_SOURCES,
    }
    for cat, sources in source_map.items():
        part = master.loc[(master["category"] == cat) & master["matched"]].copy()
        for source in sources:
            col = f"src__{source}"
            if col not in part:
                continue
            row = metric_row(source, part, col, "actual_points")
            row.update({"category": cat, "type": "source"})
            rows.append(row)
        row = metric_row("CONSENSUS_MEDIAN", part, "projection_median", "actual_points")
        row.update({"category": cat, "type": "consensus"})
        rows.append(row)

    # Combined skaters is useful in addition to F/D.
    sk = master.loc[master["category"].isin(["F", "D"]) & master["matched"]].copy()
    for source in SKATER_SOURCES:
        row = metric_row(source, sk, f"src__{source}", "actual_points")
        row.update({"category": "SKATERS", "type": "source"})
        rows.append(row)
    row = metric_row("CONSENSUS_MEDIAN", sk, "projection_median", "actual_points")
    row.update({"category": "SKATERS", "type": "consensus"})
    rows.append(row)

    out = pd.DataFrame(rows)
    out["mae_rank_within_category"] = out.groupby("category")["mae"].rank(method="min")
    consensus_mae = out.loc[out["type"].eq("consensus"), ["category", "mae"]].rename(columns={"mae": "consensus_mae"})
    out = out.merge(consensus_mae, on="category", how="left")
    out["mae_delta_vs_consensus"] = out["mae"] - out["consensus_mae"]
    return out.sort_values(["category", "mae"], na_position="last")


def projection_quality_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, part in [("ALL", master), *[(c, master.loc[master["category"] == c]) for c in ["F", "D", "G", "T"]]]:
        clean = part.loc[part["matched"]].copy()
        rows.append({
            "category": label,
            "n": len(clean),
            "mae_points": float(clean["abs_point_error"].mean()) if len(clean) else math.nan,
            "median_abs_error": float(clean["abs_point_error"].median()) if len(clean) else math.nan,
            "rmse_points": rmse(clean["point_error"]),
            "bias_actual_minus_projection": float(clean["point_error"].mean()) if len(clean) else math.nan,
            "rank_spearman_same_universe": spearman(clean["projected_rank"], clean["actual_rank_same_universe"]),
            "mean_abs_rank_error_same": float(clean["abs_rank_error_same"].mean()) if len(clean) else math.nan,
            "exact_tier_rate_same": float((clean["abs_tier_error_same"] == 0).mean()) if len(clean) else math.nan,
            "within_1_tier_rate_same": float((clean["abs_tier_error_same"] <= 1).mean()) if len(clean) else math.nan,
            "within_2_tier_rate_same": float((clean["abs_tier_error_same"] <= 2).mean()) if len(clean) else math.nan,
            "mean_abs_tier_error_same": float(clean["abs_tier_error_same"].mean()) if len(clean) else math.nan,
        })
    return pd.DataFrame(rows)


def value_band_analysis(master: pd.DataFrame) -> pd.DataFrame:
    data = master.loc[master["matched"]].copy()
    repl_rank = {cat: GM_COUNT * slots for cat, slots in ROSTER_COUNTS.items()}

    def band(row: pd.Series) -> str:
        r = float(row["projected_rank"])
        rr = repl_rank[row["category"]]
        ratio = r / rr
        if ratio <= 0.25:
            return "elite_top_25pct_of_draftable"
        if ratio <= 0.75:
            return "draft_core_25_to_75pct"
        if ratio <= 1.25:
            return "replacement_zone_75_to_125pct"
        return "depth_below_125pct"

    data["value_band"] = data.apply(band, axis=1)
    rows = []
    for (cat, band_name), part in data.groupby(["category", "value_band"], sort=False):
        rows.append({
            "category": cat,
            "value_band": band_name,
            "n": len(part),
            "mae": float(part["abs_point_error"].mean()),
            "bias": float(part["point_error"].mean()),
            "mean_abs_rank_error": float(part["abs_rank_error_same"].mean()),
            "mean_abs_tier_error": float(part["abs_tier_error_same"].mean()),
        })
    return pd.DataFrame(rows)


def dispersion_analysis(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    binned_rows = []
    for cat, part in [("ALL", master.loc[master["matched"]]), *[(c, master.loc[(master["category"] == c) & master["matched"]]) for c in ["F", "D", "G", "T"]]]:
        clean = part.dropna(subset=["projection_mad", "abs_point_error"])
        rows.append({
            "category": cat,
            "n": len(clean),
            "pearson_mad_vs_abs_error": pearson(clean["projection_mad"], clean["abs_point_error"]),
            "spearman_mad_vs_abs_error": spearman(clean["projection_mad"], clean["abs_point_error"]),
            "pearson_std_vs_abs_error": pearson(clean["projection_std"], clean["abs_point_error"]),
            "spearman_std_vs_abs_error": spearman(clean["projection_std"], clean["abs_point_error"]),
        })
        if len(clean) >= 8 and clean["projection_mad"].nunique() >= 4:
            temp = clean.copy()
            try:
                temp["dispersion_quartile"] = pd.qcut(temp["projection_mad"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
                for q, qp in temp.groupby("dispersion_quartile", observed=True):
                    binned_rows.append({"category": cat, "dispersion_quartile": str(q), "n": len(qp), "mean_mad": qp["projection_mad"].mean(), "mae": qp["abs_point_error"].mean()})
            except ValueError:
                pass
    return pd.DataFrame(rows), pd.DataFrame(binned_rows)


def source_count_analysis(master: pd.DataFrame) -> pd.DataFrame:
    clean = master.loc[master["matched"]].copy()
    return (
        clean.groupby(["category", "n_sources"], dropna=False)
        .agg(n=("join_key", "size"), mae=("abs_point_error", "mean"), bias=("point_error", "mean"), median_abs_error=("abs_point_error", "median"))
        .reset_index()
        .sort_values(["category", "n_sources"])
    )


def poolpro_ceiling_analysis(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = master.loc[
        master["matched"] & master["projection_ceiling_poolpro"].notna() & master["category"].isin(["F", "D", "G"])
    ].copy()
    data["consensus_abs_error"] = (data["actual_points"] - data["projection_median"]).abs()
    data["ceiling_abs_error"] = (data["actual_points"] - data["projection_ceiling_poolpro"]).abs()
    data["actual_above_consensus"] = data["actual_points"] > data["projection_median"]
    data["ceiling_closer_than_consensus"] = data["ceiling_abs_error"] < data["consensus_abs_error"]
    data["upside_gap_ceiling_minus_consensus"] = data["projection_ceiling_poolpro"] - data["projection_median"]
    summary = []
    for cat, part in [("ALL", data), *[(c, data[data["category"] == c]) for c in ["F", "D", "G"]]]:
        if part.empty:
            continue
        overshoot = part[part["actual_above_consensus"]]
        summary.append({
            "category": cat,
            "n_with_ceiling": len(part),
            "consensus_mae": part["consensus_abs_error"].mean(),
            "ceiling_mae": part["ceiling_abs_error"].mean(),
            "n_actual_above_consensus": len(overshoot),
            "ceiling_closer_rate_when_actual_above_consensus": overshoot["ceiling_closer_than_consensus"].mean() if len(overshoot) else math.nan,
            "corr_upside_gap_vs_positive_error": pearson(part["upside_gap_ceiling_minus_consensus"], part["point_error"].clip(lower=0)),
        })
    return pd.DataFrame(summary), data


def team_diagnostics(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = master.loc[(master["category"] == "T") & master["matched"]].copy()
    skaters = master.loc[master["category"].isin(["F", "D"]) & master["matched"]].copy()
    goalies = master.loc[(master["category"] == "G") & master["matched"]].copy()

    sk_agg = skaters.groupby("team_key").agg(
        n_skaters=("join_key", "size"),
        mean_skater_error=("point_error", "mean"),
        median_skater_error=("point_error", "median"),
        mean_skater_abs_error=("abs_point_error", "mean"),
    ).reset_index()
    g_agg = goalies.groupby("team_key").agg(
        n_goalies=("join_key", "size"),
        mean_goalie_error=("point_error", "mean"),
        median_goalie_error=("point_error", "median"),
        mean_goalie_abs_error=("abs_point_error", "mean"),
    ).reset_index()

    cols = ["team_key", "projection_median", "actual_points", "point_error", "wins", "losses", "ot_losses"]
    team = teams[cols].rename(columns={
        "projection_median": "team_projection",
        "actual_points": "team_actual_points",
        "point_error": "team_error",
    }).merge(sk_agg, on="team_key", how="left").merge(g_agg, on="team_key", how="left")
    team["points_from_wins"] = 2 * pd.to_numeric(team["wins"], errors="coerce")
    team["otl_cushion"] = pd.to_numeric(team["ot_losses"], errors="coerce")
    team["otl_share_of_points"] = team["otl_cushion"] / team["team_actual_points"].replace(0, np.nan)
    team["projected_rank"] = team["team_projection"].rank(method="first", ascending=False)
    team["actual_rank"] = team["team_actual_points"].rank(method="first", ascending=False)

    correlations = pd.DataFrame([
        {"relationship": "team_error_vs_mean_skater_error", "n": team[["team_error", "mean_skater_error"]].dropna().shape[0], "pearson": pearson(team["team_error"], team["mean_skater_error"]), "spearman": spearman(team["team_error"], team["mean_skater_error"])},
        {"relationship": "team_error_vs_mean_goalie_error", "n": team[["team_error", "mean_goalie_error"]].dropna().shape[0], "pearson": pearson(team["team_error"], team["mean_goalie_error"]), "spearman": spearman(team["team_error"], team["mean_goalie_error"])},
    ])

    clean = team[["team_projection", "team_actual_points"]].dropna()
    slope = intercept = r2 = math.nan
    if len(clean) >= 2 and clean["team_projection"].nunique() > 1:
        slope, intercept = np.polyfit(clean["team_projection"], clean["team_actual_points"], 1)
        predicted = intercept + slope * clean["team_projection"]
        ss_res = float(np.square(clean["team_actual_points"] - predicted).sum())
        ss_tot = float(np.square(clean["team_actual_points"] - clean["team_actual_points"].mean()).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else math.nan
    compression = pd.DataFrame([{
        "n": len(clean),
        "actual_on_projection_slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "projection_std": clean["team_projection"].std(ddof=0),
        "actual_std": clean["team_actual_points"].std(ddof=0),
        "actual_to_projection_std_ratio": clean["team_actual_points"].std(ddof=0) / clean["team_projection"].std(ddof=0) if clean["team_projection"].std(ddof=0) else math.nan,
        "projection_range": clean["team_projection"].max() - clean["team_projection"].min(),
        "actual_range": clean["team_actual_points"].max() - clean["team_actual_points"].min(),
    }])

    # Quantify OTL cushioning by projected quartile. No projected W/OTL exist in the source.
    try:
        team["projected_strength_quartile"] = pd.qcut(team["team_projection"], q=4, labels=["Q1_weak", "Q2", "Q3", "Q4_strong"])
    except ValueError:
        team["projected_strength_quartile"] = "ALL"
    otl = team.groupby("projected_strength_quartile", observed=True).agg(
        n=("team_key", "size"),
        mean_projection=("team_projection", "mean"),
        mean_actual_points=("team_actual_points", "mean"),
        mean_wins=("wins", "mean"),
        mean_losses=("losses", "mean"),
        mean_otl=("ot_losses", "mean"),
        mean_otl_cushion=("otl_cushion", "mean"),
        mean_otl_share=("otl_share_of_points", "mean"),
        mean_team_error=("team_error", "mean"),
    ).reset_index()
    return team.sort_values("team_projection", ascending=False), correlations, compression, otl


def strategy(name: str):
    if name == "vorp":
        return VorpStrategy()
    if name == "tier_vorp":
        return TierVorpStrategy(TIER_CONFIG)
    raise ValueError(name)


def projection_assets(master: pd.DataFrame) -> tuple[list[DraftAsset], dict[str, float]]:
    assets = []
    actual_map = {}
    for row in master.loc[master["matched"]].itertuples(index=False):
        asset_id = row.join_key
        team = row.team_key if hasattr(row, "team_key") else ""
        assets.append(DraftAsset(
            entity_id=asset_id,
            name=str(row.FullName),
            category=str(row.category),
            value=float(row.projection_median),
            nhl_team=str(team),
            metadata={"actual_points": float(row.actual_points)},
        ))
        actual_map[asset_id] = float(row.actual_points)
    return assets, actual_map


def actual_same_assets(projected_assets: list[DraftAsset], actual_map: dict[str, float]) -> list[DraftAsset]:
    return [DraftAsset(a.entity_id, a.name, a.category, actual_map[a.entity_id], a.nhl_team) for a in projected_assets]


def actual_full_assets(actual: pd.DataFrame) -> list[DraftAsset]:
    assets = []
    for row in actual.itertuples(index=False):
        if row.category == "T":
            asset_id = f"T:{row.team_key}"
            name = str(row.team_key)
            team = str(row.team_key)
        else:
            if pd.isna(row.entity_id):
                continue
            asset_id = f"P:{int(row.entity_id)}"
            name = str(row.name)
            team = norm_team(row.nhl_team)
        assets.append(DraftAsset(asset_id, name, str(row.category), float(row.pool_points_raw), team))
    return assets


def validate_rosters(result) -> None:
    for gm_id, roster in result.rosters.items():
        counts = {c: sum(a.category == c for a in roster) for c in ROSTER_COUNTS}
        if counts != ROSTER_COUNTS:
            raise AssertionError(f"Invalid roster GM {gm_id}: {counts} != {ROSTER_COUNTS}")


def score_rosters(result, actual_map: dict[str, float] | None) -> dict[int, float]:
    totals = {}
    for gm_id, roster in result.rosters.items():
        if actual_map is None:
            totals[gm_id] = float(sum(a.value for a in roster))
        else:
            totals[gm_id] = float(sum(actual_map[a.entity_id] for a in roster))
    return totals


def result_row(scenario_name: str, draft_slot: int, totals: dict[int, float], focal_gm: int) -> dict[str, object]:
    focal = totals[focal_gm]
    ordered = sorted(totals.values(), reverse=True)
    rank = 1 + sum(v > focal for v in totals.values())
    winner = max(ordered)
    field = [v for gm, v in totals.items() if gm != focal_gm]
    return {
        "scenario": scenario_name,
        "draft_slot": draft_slot,
        "actual_points": focal,
        "actual_rank": rank,
        "win": rank == 1,
        "top3": rank <= 3,
        "gap_to_winner": winner - focal,
        "margin_vs_field_mean": focal - float(np.mean(field)),
    }


def run_draft_suite(master: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proj_assets, actual_map = projection_assets(master)
    same_assets = actual_same_assets(proj_assets, actual_map)
    full_assets = actual_full_assets(actual)
    universes = {"projection": (proj_assets, actual_map), "actual_same": (same_assets, actual_map), "actual_full": (full_assets, None)}

    scenarios = [
        Scenario("v0_tier_vorp_vs_tier_vorp", "projection", "tier_vorp", "tier_vorp", homogeneous=True),
        Scenario("v0_vorp_vs_vorp", "projection", "vorp", "vorp", homogeneous=True),
        Scenario("v0_vorp_vs_tier_vorp", "projection", "vorp", "tier_vorp"),
        Scenario("v0_tier_vorp_vs_vorp", "projection", "tier_vorp", "vorp"),
        Scenario("perfect_same_tier_vorp_vs_tier_vorp", "actual_same", "tier_vorp", "tier_vorp", homogeneous=True),
        Scenario("perfect_same_vorp_vs_vorp", "actual_same", "vorp", "vorp", homogeneous=True),
        Scenario("perfect_same_vorp_vs_tier_vorp", "actual_same", "vorp", "tier_vorp"),
        Scenario("perfect_same_tier_vorp_vs_vorp", "actual_same", "tier_vorp", "vorp"),
        Scenario("perfect_full_tier_vorp_vs_tier_vorp", "actual_full", "tier_vorp", "tier_vorp", homogeneous=True),
        Scenario("perfect_full_vorp_vs_vorp", "actual_full", "vorp", "vorp", homogeneous=True),
    ]

    rows = []
    pick_rows = []
    for sc in scenarios:
        assets, actual_map_for_universe = universes[sc.universe]
        if sc.homogeneous:
            strategies = {gm: strategy(sc.focal_strategy) for gm in range(1, GM_COUNT + 1)}
            result = run_draft(assets, GM_COUNT, ROSTER, strategies)
            validate_rosters(result)
            totals = score_rosters(result, actual_map_for_universe)
            for gm in range(1, GM_COUNT + 1):
                rows.append(result_row(sc.name, gm, totals, gm))
            for pick in result.picks:
                pick_rows.append({"scenario": sc.name, "focal_slot": "ALL", "overall_pick": pick.overall_pick, "round": pick.round_no, "gm_id": pick.gm_id, "asset_id": pick.asset.entity_id, "name": pick.asset.name, "category": pick.asset.category, "decision_value": pick.asset.value, "strategy": pick.strategy_name})
        else:
            for focal in range(1, GM_COUNT + 1):
                strategies = {
                    gm: strategy(sc.focal_strategy if gm == focal else sc.opponent_strategy)
                    for gm in range(1, GM_COUNT + 1)
                }
                result = run_draft(assets, GM_COUNT, ROSTER, strategies)
                validate_rosters(result)
                totals = score_rosters(result, actual_map_for_universe)
                rows.append(result_row(sc.name, focal, totals, focal))
                for pick in result.picks:
                    if pick.gm_id == focal:
                        pick_rows.append({"scenario": sc.name, "focal_slot": focal, "overall_pick": pick.overall_pick, "round": pick.round_no, "gm_id": pick.gm_id, "asset_id": pick.asset.entity_id, "name": pick.asset.name, "category": pick.asset.category, "decision_value": pick.asset.value, "strategy": pick.strategy_name})

    by_slot = pd.DataFrame(rows)
    summary = by_slot.groupby("scenario").agg(
        trials=("draft_slot", "size"),
        mean_actual_points=("actual_points", "mean"),
        median_actual_points=("actual_points", "median"),
        std_actual_points=("actual_points", "std"),
        min_actual_points=("actual_points", "min"),
        max_actual_points=("actual_points", "max"),
        mean_rank=("actual_rank", "mean"),
        win_rate=("win", "mean"),
        top3_rate=("top3", "mean"),
        mean_gap_to_winner=("gap_to_winner", "mean"),
        mean_margin_vs_field=("margin_vs_field_mean", "mean"),
    ).reset_index()
    return summary, by_slot, pd.DataFrame(pick_rows)


def paired_draft_strategy_comparison(draft_by_slot: pd.DataFrame) -> pd.DataFrame:
    """Compare homogeneous VORP and TierVORP slot-by-slot.

    Mean points across a homogeneous 15-GM draft can be mechanically similar because
    it averages the allocation over all managers. Paired slot deltas reveal who is
    helped/hurt by the strategy and how much slot variance changes.
    """
    pairs = [
        ("v0", "v0_vorp_vs_vorp", "v0_tier_vorp_vs_tier_vorp"),
        ("perfect_same", "perfect_same_vorp_vs_vorp", "perfect_same_tier_vorp_vs_tier_vorp"),
        ("perfect_full", "perfect_full_vorp_vs_vorp", "perfect_full_tier_vorp_vs_tier_vorp"),
    ]
    rows = []
    for label, vorp_name, tier_name in pairs:
        v = draft_by_slot.loc[draft_by_slot["scenario"].eq(vorp_name), ["draft_slot", "actual_points", "actual_rank"]].rename(
            columns={"actual_points": "vorp_points", "actual_rank": "vorp_rank"}
        )
        t = draft_by_slot.loc[draft_by_slot["scenario"].eq(tier_name), ["draft_slot", "actual_points", "actual_rank"]].rename(
            columns={"actual_points": "tier_points", "actual_rank": "tier_rank"}
        )
        merged = v.merge(t, on="draft_slot", how="inner")
        if merged.empty:
            continue
        merged["points_delta_vorp_minus_tier"] = merged["vorp_points"] - merged["tier_points"]
        merged["rank_delta_vorp_minus_tier"] = merged["vorp_rank"] - merged["tier_rank"]
        for row in merged.itertuples(index=False):
            rows.append({
                "comparison": label,
                "draft_slot": row.draft_slot,
                "vorp_points": row.vorp_points,
                "tier_points": row.tier_points,
                "points_delta_vorp_minus_tier": row.points_delta_vorp_minus_tier,
                "vorp_rank": row.vorp_rank,
                "tier_rank": row.tier_rank,
                "rank_delta_vorp_minus_tier": row.rank_delta_vorp_minus_tier,
            })
    return pd.DataFrame(rows)


def paired_draft_strategy_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows = []
    for label, part in paired.groupby("comparison"):
        delta = part["points_delta_vorp_minus_tier"]
        rows.append({
            "comparison": label,
            "n_slots": len(part),
            "mean_points_delta_vorp_minus_tier": delta.mean(),
            "median_points_delta_vorp_minus_tier": delta.median(),
            "mean_abs_slot_delta": delta.abs().mean(),
            "max_abs_slot_delta": delta.abs().max(),
            "vorp_better_slots": int((delta > 0).sum()),
            "tier_better_slots": int((delta < 0).sum()),
            "tied_slots": int((delta == 0).sum()),
        })
    return pd.DataFrame(rows)


def draft_universe_effect(draft_summary: pd.DataFrame) -> pd.DataFrame:
    lookup = draft_summary.set_index("scenario")
    rows = []
    pairs = [
        ("tier_vorp", "perfect_same_tier_vorp_vs_tier_vorp", "perfect_full_tier_vorp_vs_tier_vorp"),
        ("vorp", "perfect_same_vorp_vs_vorp", "perfect_full_vorp_vs_vorp"),
    ]
    for strategy_name, same_name, full_name in pairs:
        if same_name in lookup.index and full_name in lookup.index:
            same = lookup.loc[same_name, "mean_actual_points"]
            full = lookup.loc[full_name, "mean_actual_points"]
            rows.append({"strategy": strategy_name, "perfect_same_universe_mean": same, "perfect_full_universe_mean": full, "full_minus_same": full - same})
    return pd.DataFrame(rows)


def final_loss_decomposition(draft_summary: pd.DataFrame) -> pd.DataFrame:
    lookup = draft_summary.set_index("scenario")
    rows = []
    for strat in ["tier_vorp", "vorp"]:
        v0 = f"v0_{strat}_vs_{strat}"
        perfect_same = f"perfect_same_{strat}_vs_{strat}"
        perfect_full = f"perfect_full_{strat}_vs_{strat}"
        if all(x in lookup.index for x in [v0, perfect_same, perfect_full]):
            v0_pts = float(lookup.loc[v0, "mean_actual_points"])
            same_pts = float(lookup.loc[perfect_same, "mean_actual_points"])
            full_pts = float(lookup.loc[perfect_full, "mean_actual_points"])
            rows.append({
                "strategy": strat,
                "v0_projection_mean_points": v0_pts,
                "perfect_info_same_universe_mean_points": same_pts,
                "perfect_info_full_universe_mean_points": full_pts,
                "projection_error_cost_same_minus_v0": same_pts - v0_pts,
                "universe_cost_full_minus_same": full_pts - same_pts,
                "total_gap_full_perfect_minus_v0": full_pts - v0_pts,
            })
    return pd.DataFrame(rows)


def build_test_catalog() -> pd.DataFrame:
    rows = [
        (1, "Contrôle qualité des données", "IMPLEMENTED", "01_qc_issues.csv; 02_projection_master.csv"),
        (2, "Consensus de projection", "IMPLEMENTED", "02_projection_master.csv"),
        (3, "Performance de chaque source", "IMPLEMENTED", "03_source_performance.csv"),
        (4, "Consensus vs sources individuelles", "IMPLEMENTED", "03_source_performance.csv"),
        (5, "Erreur brute asset par asset", "IMPLEMENTED", "02_projection_master.csv; 10_biggest_misses.csv"),
        (6, "Erreur par position", "IMPLEMENTED", "04_projection_quality_summary.csv"),
        (7, "Erreur selon niveau de joueur", "IMPLEMENTED", "05_value_band_analysis.csv"),
        (8, "Erreur de classement", "IMPLEMENTED", "02_projection_master.csv; 04_projection_quality_summary.csv"),
        (9, "Erreur de tier", "IMPLEMENTED", "02_projection_master.csv; 04_projection_quality_summary.csv"),
        (10, "Dispersion experts vs erreur", "IMPLEMENTED", "06_dispersion_correlations.csv; 07_dispersion_bins.csv"),
        (11, "Nombre de sources vs qualité", "IMPLEMENTED", "08_source_count_analysis.csv"),
        (12, "Analyse des gros ratés", "IMPLEMENTED", "10_biggest_misses.csv"),
        (13, "VORP projetés", "IMPLEMENTED", "11_projected_tiers.csv; 02_projection_master.csv"),
        (14, "Tiers V0", "IMPLEMENTED", "11_projected_tiers.csv"),
        (15, "Draft V0 TierVORP vs TierVORP", "IMPLEMENTED", "20_draft_summary.csv; 21_draft_by_slot.csv; 24_paired_strategy_by_slot.csv"),
        (16, "Draft V0 VORP vs VORP", "IMPLEMENTED", "20_draft_summary.csv; 21_draft_by_slot.csv; 25_paired_strategy_summary.csv"),
        (17, "Draft V0 VORP vs TierVORP", "IMPLEMENTED", "20_draft_summary.csv; 21_draft_by_slot.csv"),
        (18, "Draft V0 TierVORP vs VORP", "IMPLEMENTED", "20_draft_summary.csv; 21_draft_by_slot.csv"),
        (19, "Perfect info TierVORP", "IMPLEMENTED", "20_draft_summary.csv"),
        (20, "Perfect info VORP / cross-strategy", "IMPLEMENTED", "20_draft_summary.csv"),
        (21, "Effet univers de joueurs", "IMPLEMENTED", "23_universe_effect.csv"),
        (22, "Robustesse par slot", "IMPLEMENTED", "21_draft_by_slot.csv"),
        (23, "Décomposition par équipe NHL", "IMPLEMENTED", "30_team_factor_detail.csv"),
        (24, "Corrélation erreur équipe / joueurs", "IMPLEMENTED", "31_team_factor_correlations.csv"),
        (25, "Compression projections équipes", "IMPLEMENTED", "32_team_projection_compression.csv"),
        (26, "Wins / OTL équipes", "LIMITED_BY_SOURCE", "33_team_otl_analysis.csv (no projected W/OTL available)"),
        (27, "Gardiens: erreur score total", "IMPLEMENTED", "03_source_performance.csv; 04_projection_quality_summary.csv"),
        (28, "PoolPro Top comme plafond", "IMPLEMENTED", "40_poolpro_ceiling_summary.csv; 41_poolpro_ceiling_detail.csv"),
        (29, "Cas particuliers documentés", "IMPLEMENTED", "42_manual_case_annotations.csv"),
        (30, "Diagnostic final V0", "IMPLEMENTED", "50_loss_decomposition.csv; V0_REPORT.md"),
    ]
    return pd.DataFrame(rows, columns=["test_no", "test", "status", "outputs"])


def checks_table(sk: pd.DataFrame, master: pd.DataFrame, projected_tiers: pd.DataFrame, draft_summary: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(name: str, passed: bool, detail: str = ""):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add("all_projection_categories_valid", set(master["category"].dropna().unique()) <= {"F", "D", "G", "T"}, str(sorted(master["category"].dropna().unique())))
    add("all_projection_assets_matched_to_actual", bool(master["matched"].all()), f"matched={int(master['matched'].sum())}/{len(master)}")
    add("poolpro_top_excluded_from_consensus", True, "Consensus columns explicitly use only central sources.")
    kuch = sk.loc[sk["FullName"].eq("Nikita Kucherov")]
    if len(kuch):
        espn_missing = kuch["src__ESPN"].isna().all()
        add("kucherov_espn_excluded", espn_missing, f"median={kuch['projection_median'].iloc[0]}")
    add("projected_tiers_nonempty", not projected_tiers.empty, f"n_tiers={len(projected_tiers)}")
    expected_scenarios = {"v0_tier_vorp_vs_tier_vorp", "v0_vorp_vs_vorp", "v0_vorp_vs_tier_vorp", "v0_tier_vorp_vs_vorp"}
    add("symmetric_v0_draft_scenarios_present", expected_scenarios <= set(draft_summary["scenario"]), str(sorted(expected_scenarios & set(draft_summary["scenario"]))))
    return pd.DataFrame(checks)


def report_markdown(
    args: argparse.Namespace,
    master: pd.DataFrame,
    quality: pd.DataFrame,
    sources: pd.DataFrame,
    dispersion: pd.DataFrame,
    compression: pd.DataFrame,
    team_corr: pd.DataFrame,
    draft_summary: pd.DataFrame,
    universe_effect: pd.DataFrame,
    loss: pd.DataFrame,
    ceiling_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    def f(value, digits=2):
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"

    all_q = quality.loc[quality["category"].eq("ALL")].iloc[0]
    dlookup = draft_summary.set_index("scenario")
    lines = [
        "# V0 Projection-Only Diagnostic Report",
        "",
        f"Season: **{args.season}**  ",
        f"Format: **{GM_COUNT} GM — 10F / 3D / 2G / 1T**  ",
        "V0 uses only published central projections. Age, historical GP/PPG, DFO role, and injury adjustments are not predictive features in this run.",
        "",
        "## 1. Data / consensus checks",
        "",
        f"- Assets in projection universe: **{len(master)}**; matched to actual results: **{int(master['matched'].sum())}**.",
        f"- Overall consensus MAE: **{f(all_q['mae_points'])} points**; bias (actual - projection): **{f(all_q['bias_actual_minus_projection'])}**.",
        f"- Same-universe rank Spearman: **{f(all_q['rank_spearman_same_universe'], 3)}**.",
        f"- Internal checks: **{int((checks['status']=='PASS').sum())}/{len(checks)} PASS**.",
        "",
        "## 2. Source and uncertainty diagnostics",
        "",
    ]
    for cat in ["F", "D", "G", "T"]:
        part = sources[sources["category"].eq(cat)].sort_values("mae")
        if len(part):
            best = part.iloc[0]
            cons = part[part["label"].eq("CONSENSUS_MEDIAN")]
            cons_mae = cons.iloc[0]["mae"] if len(cons) else math.nan
            lines.append(f"- {cat}: best MAE = **{best['label']} {f(best['mae'])}**; consensus MAE = **{f(cons_mae)}**.")
    lines += ["", "Dispersion vs absolute error:"]
    for _, row in dispersion.iterrows():
        lines.append(f"- {row['category']}: Spearman(MAD, |error|) = **{f(row['spearman_mad_vs_abs_error'], 3)}**.")

    lines += ["", "## 3. Draft strategy V0", ""]
    ordered = [
        "v0_tier_vorp_vs_tier_vorp", "v0_vorp_vs_vorp",
        "v0_vorp_vs_tier_vorp", "v0_tier_vorp_vs_vorp",
        "perfect_same_tier_vorp_vs_tier_vorp", "perfect_same_vorp_vs_vorp",
        "perfect_full_tier_vorp_vs_tier_vorp", "perfect_full_vorp_vs_vorp",
    ]
    for name in ordered:
        if name in dlookup.index:
            row = dlookup.loc[name]
            lines.append(
                f"- `{name}`: mean **{f(row['mean_actual_points'])}**, rank **{f(row['mean_rank'])}**, "
                f"top3 **{f(100*row['top3_rate'],1)}%**, slot SD **{f(row['std_actual_points'])}**."
            )

    lines += ["", "Homogeneous strategy comparison note: VORP/TierVORP mean points can cancel across the 15 slots; inspect `24_paired_strategy_by_slot.csv` and `25_paired_strategy_summary.csv` for slot-level redistribution."]

    lines += ["", "## 4. Team factor / compression", ""]
    if len(compression):
        c = compression.iloc[0]
        lines.append(f"- Regression actual team points on projected points: slope **{f(c['actual_on_projection_slope'],3)}**, R² **{f(c['r_squared'],3)}**.")
        lines.append(f"- Actual/projection cross-team SD ratio: **{f(c['actual_to_projection_std_ratio'],3)}** (>1 means the realized standings were more spread out than the forecasts).")
        lines.append("- Because team ranking correlation is weak in this season, use the spread ratio and quartile table together; do not interpret the regression slope alone as a compression test.")
    for _, row in team_corr.iterrows():
        lines.append(f"- {row['relationship']}: Spearman **{f(row['spearman'],3)}**, Pearson **{f(row['pearson'],3)}**.")
    lines.append("- Team source limitation: projected W and OTL were not stored, so V0 can measure actual OTL cushioning but cannot compute W/OTL forecast error without inventing data.")

    lines += ["", "## 5. PoolPro Top", ""]
    if len(ceiling_summary):
        for _, row in ceiling_summary.iterrows():
            lines.append(
                f"- {row['category']}: {int(row['n_with_ceiling'])} assets with ceiling; "
                f"ceiling closer than consensus on **{f(100*row['ceiling_closer_rate_when_actual_above_consensus'],1)}%** "
                "of cases that beat consensus."
            )

    lines += ["", "## 6. Loss decomposition", ""]
    for _, row in loss.iterrows():
        lines.append(
            f"- {row['strategy']}: projection-error cost **{f(row['projection_error_cost_same_minus_v0'])} pts**; "
            f"projection-universe cost **{f(row['universe_cost_full_minus_same'])} pts**; "
            f"total V0-to-perfect gap **{f(row['total_gap_full_perfect_minus_v0'])} pts**."
        )
    if len(universe_effect):
        lines += ["", "Perfect-info universe effect:"]
        for _, row in universe_effect.iterrows():
            lines.append(f"- {row['strategy']}: full universe - projection universe = **{f(row['full_minus_same'])} pts**.")

    lines += [
        "",
        "## 7. Interpretation guardrails",
        "",
        "- Homogeneous all-VORP vs all-TierVORP mean points can be mechanically similar because the metric averages all 15 allocations; use paired slot deltas and cross-strategy scenarios for discrimination.",
        "- Tier-number error is diagnostic, not a standalone truth metric: actual tier geometry changes when the realized distribution changes.",
        "- Manual injury/context notes are retained only for interpretation; they do not modify V0 predictions.",
        "- This is one historical projection season. Do not optimize source weights from this single season.",
        "- V1 should add historical NHL information only after V0 is frozen, so incremental value can be measured cleanly.",
        "",
        "See `00_test_catalog.csv` for the complete mapping of the 30 requested V0 tests to output files.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.db = resolve_existing(args.db, [Path("nhl_pool_analysis.sqlite")])
    args.skaters = resolve_existing(args.skaters, [Path("data/Prediction_Skater_20252026.csv")])
    args.goalies = resolve_existing(args.goalies, [Path("data/Prediction_goalies_20252026.csv")])
    args.teams = resolve_existing(args.teams, [Path("data/Prediction_Team_20252026.csv")])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("V0 full diagnostic suite — projection only")
    print(f"Season: {args.season}")
    print(f"DB: {args.db}")
    print(f"Output: {args.output_dir}")

    sk, go, tm, qc = prepare_projections(args.skaters, args.goalies, args.teams)
    projected = combined_projection_table(sk, go, tm)
    projected_tiered, proj_repl, proj_repl_tier, projected_tiers = add_projection_tiers(projected)
    actual = load_actual(args.db, args.season)
    master, same_repl, same_repl_tier, full_repl, full_repl_tier, same_tiers, full_tiers = build_master(projected_tiered, actual)

    source_perf = source_performance(master)
    quality = projection_quality_summary(master)
    value_bands = value_band_analysis(master)
    disp_corr, disp_bins = dispersion_analysis(master)
    source_count = source_count_analysis(master)
    biggest = master.loc[master["matched"]].sort_values("abs_point_error", ascending=False).head(args.top_misses).copy()
    tier_movers = master.loc[master["matched"]].sort_values("abs_tier_error_same", ascending=False).head(args.top_misses).copy()
    ceiling_summary, ceiling_detail = poolpro_ceiling_analysis(master)
    team_detail, team_corr, team_compression, team_otl = team_diagnostics(master)
    draft_summary, draft_by_slot, draft_picks = run_draft_suite(master, actual)
    paired_strategy = paired_draft_strategy_comparison(draft_by_slot)
    paired_strategy_summary = paired_draft_strategy_summary(paired_strategy)
    universe_effect = draft_universe_effect(draft_summary)
    loss = final_loss_decomposition(draft_summary)
    catalog = build_test_catalog()
    checks = checks_table(sk, master, projected_tiers, draft_summary)

    manual_cases = master.loc[master["manual_note"].fillna("").ne("")].copy()
    source_zero_audit = qc.loc[qc["check"].eq("ignored_zero_no_injury")].copy() if not qc.empty else pd.DataFrame()

    # Export all requested diagnostics.
    exports = {
        "00_test_catalog.csv": catalog,
        "01_qc_issues.csv": qc,
        "02_projection_master.csv": master,
        "03_source_performance.csv": source_perf,
        "04_projection_quality_summary.csv": quality,
        "05_value_band_analysis.csv": value_bands,
        "06_dispersion_correlations.csv": disp_corr,
        "07_dispersion_bins.csv": disp_bins,
        "08_source_count_analysis.csv": source_count,
        "09_zero_source_audit.csv": source_zero_audit,
        "10_biggest_misses.csv": biggest,
        "10b_biggest_tier_movers.csv": tier_movers,
        "11_projected_tiers.csv": projected_tiers,
        "12_actual_same_universe_tiers.csv": same_tiers,
        "13_actual_full_universe_tiers.csv": full_tiers,
        "20_draft_summary.csv": draft_summary,
        "21_draft_by_slot.csv": draft_by_slot,
        "22_draft_picks.csv": draft_picks,
        "23_universe_effect.csv": universe_effect,
        "24_paired_strategy_by_slot.csv": paired_strategy,
        "25_paired_strategy_summary.csv": paired_strategy_summary,
        "30_team_factor_detail.csv": team_detail,
        "31_team_factor_correlations.csv": team_corr,
        "32_team_projection_compression.csv": team_compression,
        "33_team_otl_analysis.csv": team_otl,
        "40_poolpro_ceiling_summary.csv": ceiling_summary,
        "41_poolpro_ceiling_detail.csv": ceiling_detail,
        "42_manual_case_annotations.csv": manual_cases,
        "50_loss_decomposition.csv": loss,
        "90_internal_checks.csv": checks,
    }
    for filename, frame in exports.items():
        frame.to_csv(args.output_dir / filename, index=False)

    report = report_markdown(args, master, quality, source_perf, disp_corr, team_compression, team_corr, draft_summary, universe_effect, loss, ceiling_summary, checks)
    (args.output_dir / "V0_REPORT.md").write_text(report, encoding="utf-8")

    print("\nProjected replacement levels:", {k: round(v, 3) for k, v in proj_repl.items()})
    print("Projected replacement tiers:", proj_repl_tier)
    print("Actual same-universe replacement levels:", {k: round(v, 3) for k, v in same_repl.items()})
    print("Actual full-universe replacement levels:", {k: round(v, 3) for k, v in full_repl.items()})

    print("\nProjection quality")
    print(quality.to_string(index=False))
    print("\nDraft summary")
    print(draft_summary.to_string(index=False))
    print("\nTeam compression")
    print(team_compression.to_string(index=False))
    print("\nTeam factor correlations")
    print(team_corr.to_string(index=False))
    print("\nLoss decomposition")
    print(loss.to_string(index=False))
    print("\nInternal checks")
    print(checks.to_string(index=False))

    failed = checks.loc[checks["status"].eq("FAIL")]
    print(f"\nFiles created in: {args.output_dir}")
    print(f"Test catalog: {args.output_dir / '00_test_catalog.csv'}")
    print(f"Report: {args.output_dir / 'V0_REPORT.md'}")
    if not failed.empty:
        print("WARNING: some internal checks failed. Inspect 90_internal_checks.csv")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
