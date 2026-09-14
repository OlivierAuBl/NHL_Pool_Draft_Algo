from __future__ import annotations

import math
import unicodedata

import pandas as pd


TEAM_ABBREVIATION_ALIASES = {
    "ARI": "UTA",
}


def _identifier(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value).strip()


def _team(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip())
    return "".join(char for char in text if not unicodedata.combining(char)).upper().rstrip("*")


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _v0_key(row: pd.Series) -> str:
    if str(row["category"]).upper() == "T":
        return f"T:{_team(row.get('team_key'))}"
    return f"P:{_identifier(row.get('NHLID'))}"


def _team_name_map(v0_master: pd.DataFrame) -> dict[str, str]:
    aliases = {"UTAH HOCKEY CLUB": "UTA", "UTAH MAMMOTH": "UTA"}
    teams = v0_master.loc[v0_master["category"] == "T"]
    for _, row in teams.iterrows():
        abbreviation = _team(row.get("team_key"))
        if not abbreviation:
            continue
        for column in ("Team", "FullName", "actual_name"):
            name = _team(row.get(column))
            if name:
                aliases[name] = abbreviation
    return aliases


def _v1_key(row: pd.Series, team_names: dict[str, str]) -> str:
    if str(row["category"]).upper() == "T":
        abbreviation = _team(row.get("nhl_team"))
        abbreviation = TEAM_ABBREVIATION_ALIASES.get(abbreviation, abbreviation)
        name = _team(row.get("name"))
        resolved = abbreviation or team_names.get(name)
        return f"T:{resolved}" if resolved else f"TNAME:{name}"
    return f"P:{_identifier(row.get('entity_id'))}"


def build_v1_comparison(v0_master: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    """Restrict V1 to the frozen V0 universe and retain an explicit fallback."""

    _require(
        v0_master,
        {"category", "projection_median", "actual_points", "NHLID", "team_key"},
        "V0 master",
    )
    _require(
        v1,
        {"entity_id", "name", "category", "nhl_team", "projected_points"},
        "V1 projections",
    )

    left = v0_master.copy()
    left["category"] = left["category"].astype(str).str.upper()
    left["join_key"] = left.apply(_v0_key, axis=1)
    left["v0_projected_points"] = pd.to_numeric(left["projection_median"], errors="coerce")
    left["actual_points"] = pd.to_numeric(left["actual_points"], errors="coerce")

    right = v1.copy()
    right["category"] = right["category"].astype(str).str.upper()
    team_names = _team_name_map(left)
    right["join_key"] = right.apply(lambda row: _v1_key(row, team_names), axis=1)
    if right["join_key"].duplicated().any():
        duplicates = sorted(right.loc[right["join_key"].duplicated(), "join_key"].unique())
        raise ValueError(f"V1 projections contain duplicate keys: {duplicates[:10]}")

    rename = {
        "name": "v1_name",
        "projected_points": "v1_projected_points",
        "projected_ppg": "v1_projected_ppg",
        "projected_gp": "v1_projected_gp",
        "projected_starts": "v1_projected_starts",
        "projected_start_share": "v1_projected_start_share",
        "projected_win_rate": "v1_projected_win_rate",
        "projected_shutout_rate": "v1_projected_shutout_rate",
        "projected_otl_rate": "v1_projected_otl_rate",
        "starts_source": "v1_starts_source",
        "stddev_points": "v1_stddev_points",
    }
    available = [column for column in rename if column in right]
    right = right[["join_key", *available]].rename(columns=rename)
    comparison = left.merge(right, on="join_key", how="left", validate="one_to_one")
    comparison["v1_projected_points"] = pd.to_numeric(
        comparison.get("v1_projected_points"), errors="coerce"
    )
    comparison["v1_available"] = comparison["v1_projected_points"].notna()
    comparison["v1_with_v0_fallback_points"] = comparison["v1_projected_points"].fillna(
        comparison["v0_projected_points"]
    )
    comparison["v0_error"] = comparison["actual_points"] - comparison["v0_projected_points"]
    comparison["v1_error"] = comparison["actual_points"] - comparison["v1_projected_points"]
    comparison["v1_fallback_error"] = (
        comparison["actual_points"] - comparison["v1_with_v0_fallback_points"]
    )
    return comparison


def _metrics(
    frame: pd.DataFrame,
    label: str,
    prediction_column: str,
    category: str,
) -> dict[str, object]:
    part = frame if category == "ALL" else frame.loc[frame["category"] == category]
    clean = part[[prediction_column, "actual_points"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if clean.empty:
        return {
            "category": category,
            "model": label,
            "n": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "bias_actual_minus_projection": math.nan,
            "pearson": math.nan,
            "spearman": math.nan,
        }
    errors = clean["actual_points"] - clean[prediction_column]
    can_correlate = (
        len(clean) >= 2
        and clean[prediction_column].nunique() >= 2
        and clean["actual_points"].nunique() >= 2
    )
    pearson = clean[prediction_column].corr(clean["actual_points"]) if can_correlate else math.nan
    spearman = (
        clean[prediction_column].rank().corr(clean["actual_points"].rank())
        if can_correlate
        else math.nan
    )
    return {
        "category": category,
        "model": label,
        "n": len(clean),
        "mae": float(errors.abs().mean()),
        "rmse": float((errors.pow(2).mean()) ** 0.5),
        "bias_actual_minus_projection": float(errors.mean()),
        "pearson": float(pearson) if pd.notna(pearson) else math.nan,
        "spearman": float(spearman) if pd.notna(spearman) else math.nan,
    }


def projection_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare V0 and V1 on fair coverage plus the complete V0 universe."""

    categories = ["ALL", "F", "D", "G", "T"]
    covered = comparison.loc[comparison["v1_available"]].copy()
    rows: list[dict[str, object]] = []
    for category in categories:
        rows.append(_metrics(covered, "V0_same_V1_coverage", "v0_projected_points", category))
        rows.append(_metrics(covered, "V1_history_components", "v1_projected_points", category))
        rows.append(
            _metrics(
                comparison,
                "V1_with_V0_fallback_full_universe",
                "v1_with_v0_fallback_points",
                category,
            )
        )
        rows.append(_metrics(comparison, "V0_full_universe", "v0_projected_points", category))
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[
        metrics["model"].eq("V0_same_V1_coverage"), ["category", "mae"]
    ].rename(columns={"mae": "v0_same_coverage_mae"})
    metrics = metrics.merge(baseline, on="category", how="left")
    metrics["mae_delta_vs_v0_same_coverage"] = (
        metrics["mae"] - metrics["v0_same_coverage_mae"]
    )
    return metrics


def skater_component_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    skaters = comparison.loc[
        comparison["category"].isin({"F", "D"}) & comparison["v1_available"]
    ].copy()
    gp = pd.to_numeric(skaters.get("games_played"), errors="coerce")
    actual_points = (
        pd.to_numeric(skaters.get("goals"), errors="coerce").fillna(0)
        + pd.to_numeric(skaters.get("assists"), errors="coerce").fillna(0)
    )
    skaters["actual_gp"] = gp
    skaters["actual_ppg"] = actual_points.div(gp.where(gp > 0))
    skaters["gp_error_actual_minus_projection"] = (
        skaters["actual_gp"] - pd.to_numeric(skaters.get("v1_projected_gp"), errors="coerce")
    )
    skaters["ppg_error_actual_minus_projection"] = (
        skaters["actual_ppg"] - pd.to_numeric(skaters.get("v1_projected_ppg"), errors="coerce")
    )
    return skaters


def coverage_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in ["ALL", "F", "D", "G", "T"]:
        part = comparison if category == "ALL" else comparison.loc[comparison["category"] == category]
        covered = int(part["v1_available"].sum())
        rows.append({
            "category": category,
            "v0_universe": len(part),
            "v1_history_available": covered,
            "v0_fallback_required": len(part) - covered,
            "v1_coverage_rate": covered / len(part) if len(part) else math.nan,
        })
    return pd.DataFrame(rows)


def active_universe_projections(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the V0 target universe in the existing projection-loader contract."""

    names = comparison.get("FullName", pd.Series(index=comparison.index, dtype=object)).copy()
    if "Team" in comparison:
        names = names.fillna(comparison["Team"])
    if "v1_name" in comparison:
        names = names.fillna(comparison["v1_name"])

    v1_stddev = pd.to_numeric(
        comparison.get("v1_stddev_points", pd.Series(index=comparison.index, dtype=float)),
        errors="coerce",
    )
    v0_stddev = pd.to_numeric(
        comparison.get("projection_std", pd.Series(index=comparison.index, dtype=float)),
        errors="coerce",
    )
    output = pd.DataFrame({
        "entity_id": comparison["join_key"],
        "name": names,
        "category": comparison["category"],
        "nhl_team": comparison["team_key"].map(_team),
        "projected_points": comparison["v1_with_v0_fallback_points"],
        "stddev_points": v1_stddev.fillna(v0_stddev).fillna(0.0),
        "projection_source": comparison["v1_available"].map(
            {True: "V1_HISTORY", False: "V0_FALLBACK"}
        ),
    })
    return output.sort_values(
        ["category", "projected_points"], ascending=[True, False]
    ).reset_index(drop=True)
