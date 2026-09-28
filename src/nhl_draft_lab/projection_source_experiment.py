from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping

import pandas as pd

from nhl_draft_lab.draft.engine import run_draft
from nhl_draft_lab.models import DraftAsset, RosterConfig
from nhl_draft_lab.projection_backtest import _ProjectionViewStrategy
from nhl_draft_lab.strategies.factory import build_strategy


CONSENSUS_METHODS = (
    "mean",
    "median",
    "trimmed_mean_minmax",
    "trimmed_median_minmax",
)
DEFAULT_TRIM_MIN_SOURCES = 5


@dataclass(frozen=True)
class ProjectionMethod:
    name: str
    projection_type: str
    value_column: str
    fallback_column: str
    trim_fallback_column: str


@dataclass(frozen=True)
class PreparedExperiment:
    universe: pd.DataFrame
    methods: tuple[ProjectionMethod, ...]
    coverage: pd.DataFrame
    exclusions: pd.DataFrame
    opponent_value_column: str
    source_columns: Mapping[str, Mapping[str, str]]


def _canonical_source(raw_name: str) -> tuple[str, str]:
    cleaned = re.sub(r"^calc_", "", raw_name, flags=re.IGNORECASE)
    cleaned = re.sub(r"_score$", "", cleaned, flags=re.IGNORECASE)
    key = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
    if key == "hastag":
        key = "hashtag"
        cleaned = "Hashtag"
    return key, cleaned


def discover_source_columns(master: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Discover the best compatible source column for every roster category."""

    candidates: dict[str, list[tuple[str, str]]] = {}
    displays: dict[str, str] = {}
    for column in master.columns:
        if not column.startswith("src__") or column.startswith("src_status__"):
            continue
        key, display = _canonical_source(column.removeprefix("src__"))
        candidates.setdefault(key, []).append((column, display))
        # Prefer the non-calculated label as the display name.
        if key not in displays or not column.startswith("src__calc_"):
            displays[key] = display

    categories = sorted(master["category"].dropna().astype(str).str.upper().unique())
    discovered: dict[str, dict[str, str]] = {}
    for key, columns in candidates.items():
        display = displays[key]
        per_category: dict[str, str] = {}
        for category in categories:
            mask = master["category"].astype(str).str.upper().eq(category)
            ranked = sorted(
                columns,
                key=lambda item: (
                    pd.to_numeric(master.loc[mask, item[0]], errors="coerce").notna().sum(),
                    not item[0].startswith("src__calc_"),
                ),
                reverse=True,
            )
            if ranked:
                per_category[category] = ranked[0][0]
        discovered[display] = per_category
    return dict(sorted(discovered.items()))


def _entity_id(row: pd.Series) -> str:
    if row["category"] == "T":
        team = str(row.get("team_key", row.get("Team", ""))).strip().upper()
        return f"T:{team}"
    numeric = pd.to_numeric(pd.Series([row.get("NHLID")]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Missing NHLID for {row.get('FullName', 'player')}")
    return f"P:{int(numeric)}"


def _column_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _source_values(
    frame: pd.DataFrame,
    source_columns: Mapping[str, str],
) -> pd.Series:
    values = pd.Series(float("nan"), index=frame.index, dtype=float)
    for category, column in source_columns.items():
        mask = frame["category"].eq(category)
        values.loc[mask] = pd.to_numeric(frame.loc[mask, column], errors="coerce")
    return values


def _consensus_for_row(values: list[float], method: str, trim_min_sources: int) -> tuple[float | None, bool]:
    available = pd.Series(values, dtype=float).dropna().tolist()
    if not available:
        return None, False
    if method == "mean":
        return float(pd.Series(available).mean()), False
    if method == "median":
        return float(pd.Series(available).median()), False

    use_mean = method == "trimmed_mean_minmax"
    if len(available) < trim_min_sources:
        value = pd.Series(available).mean() if use_mean else pd.Series(available).median()
        return float(value), True
    ordered = sorted(available)[1:-1]
    value = pd.Series(ordered).mean() if use_mean else pd.Series(ordered).median()
    return float(value), False


def prepare_projection_experiment(
    master: pd.DataFrame,
    *,
    gm_count: int,
    roster_config: RosterConfig,
    opponent_source: str = "PoolPro",
    trim_min_sources: int = DEFAULT_TRIM_MIN_SOURCES,
) -> PreparedExperiment:
    if gm_count < 2:
        raise ValueError("gm_count must be at least 2")
    if trim_min_sources < 3:
        raise ValueError("trim_min_sources must be at least 3")
    required = {"category", "actual_points", "projection_median"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Projection master is missing columns: {sorted(missing)}")

    universe = master.copy()
    universe["category"] = universe["category"].astype(str).str.upper()
    universe = universe[universe["category"].isin(roster_config.counts)].copy()
    universe["actual_points"] = pd.to_numeric(universe["actual_points"], errors="coerce")
    universe["projection_median"] = pd.to_numeric(
        universe["projection_median"], errors="coerce"
    )
    if universe["actual_points"].isna().any():
        count = int(universe["actual_points"].isna().sum())
        raise ValueError(
            f"Actual scoring is unavailable for {count} assets; projections cannot be "
            "evaluated against themselves"
        )
    if universe["projection_median"].isna().any():
        count = int(universe["projection_median"].isna().sum())
        raise ValueError(f"Consolidated fallback is unavailable for {count} assets")

    names = universe.get("FullName", pd.Series(index=universe.index, dtype=object))
    if "Team" in universe:
        names = names.fillna(universe["Team"])
    universe["name"] = names.fillna("unknown").astype(str)
    universe["entity_id"] = universe.apply(_entity_id, axis=1)
    if universe["entity_id"].duplicated().any():
        duplicates = universe.loc[universe["entity_id"].duplicated(), "entity_id"].tolist()
        raise ValueError(f"Duplicate asset identifiers: {duplicates[:10]}")

    source_columns = discover_source_columns(universe)
    if opponent_source not in source_columns:
        matches = [name for name in source_columns if name.lower() == opponent_source.lower()]
        if not matches:
            raise ValueError(
                f"Opponent source {opponent_source!r} not found; discovered {sorted(source_columns)}"
            )
        opponent_source = matches[0]

    required_counts = {
        category: gm_count * int(slots)
        for category, slots in roster_config.counts.items()
    }
    category_counts = universe["category"].value_counts().to_dict()
    for category, minimum in required_counts.items():
        if category_counts.get(category, 0) < minimum:
            raise ValueError(
                f"Projection universe has only {category_counts.get(category, 0)} {category}; "
                f"need {minimum}"
            )

    raw_values = {
        source: _source_values(universe, columns)
        for source, columns in source_columns.items()
    }
    pool_pro_raw = raw_values[opponent_source]
    base_fallback = pool_pro_raw.fillna(universe["projection_median"])
    if base_fallback.isna().any():
        raise ValueError("Pool Pro plus consolidated median fallback is incomplete")
    universe["opponent_projection"] = base_fallback.astype(float)

    coverage_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    category_eligible: dict[str, dict[str, bool]] = {}
    for source, values in raw_values.items():
        category_eligible[source] = {}
        for category, minimum in required_counts.items():
            mask = universe["category"].eq(category)
            available = int(values.loc[mask].notna().sum())
            total = int(mask.sum())
            eligible = available >= minimum
            category_eligible[source][category] = eligible
            fallback_count = total - available if eligible else total
            coverage_rows.append(
                {
                    "projection_method": source,
                    "projection_type": "single_source",
                    "category": category,
                    "available_count": available,
                    "universe_count": total,
                    "coverage_pct": available / total if total else 0.0,
                    "minimum_required": minimum,
                    "category_eligible": eligible,
                    "fallback_count": fallback_count,
                    "fallback_pct": fallback_count / total if total else 0.0,
                }
            )
            if not eligible:
                exclusion_rows.append(
                    {
                        "projection_method": source,
                        "category": category,
                        "available_count": available,
                        "minimum_required": minimum,
                        "reason": "coverage below complete-draft requirement; category uses fallback",
                    }
                )

    methods: list[ProjectionMethod] = []
    for source, values in raw_values.items():
        token = _column_token(source)
        value_column = f"method__{token}"
        fallback_column = f"fallback__{token}"
        trim_column = f"trim_fallback__{token}"
        usable = pd.Series(float("nan"), index=universe.index, dtype=float)
        for category, eligible in category_eligible[source].items():
            if eligible:
                mask = universe["category"].eq(category)
                usable.loc[mask] = values.loc[mask]
        fallback = usable.isna()
        universe[value_column] = usable.fillna(base_fallback).astype(float)
        universe[fallback_column] = fallback
        universe[trim_column] = False
        methods.append(
            ProjectionMethod(source, "single_source", value_column, fallback_column, trim_column)
        )

    eligible_source_values: dict[str, list[pd.Series]] = {}
    for category in required_counts:
        mask = universe["category"].eq(category)
        eligible_source_values[category] = [
            values.loc[mask]
            for source, values in raw_values.items()
            if category_eligible[source][category]
        ]

    for method_name in CONSENSUS_METHODS:
        value_column = f"method__{method_name}"
        fallback_column = f"fallback__{method_name}"
        trim_column = f"trim_fallback__{method_name}"
        calculated = pd.Series(float("nan"), index=universe.index, dtype=float)
        trim_fallback = pd.Series(False, index=universe.index, dtype=bool)
        source_count = pd.Series(0, index=universe.index, dtype=int)
        for category in required_counts:
            mask = universe["category"].eq(category)
            series_list = eligible_source_values[category]
            if not series_list:
                continue
            comparable = pd.concat(series_list, axis=1)
            source_count.loc[mask] = comparable.notna().sum(axis=1).astype(int)
            results = comparable.apply(
                lambda row: _consensus_for_row(row.tolist(), method_name, trim_min_sources),
                axis=1,
            )
            calculated.loc[mask] = results.map(lambda result: result[0])
            trim_fallback.loc[mask] = results.map(lambda result: result[1])
        fallback = calculated.isna()
        universe[value_column] = calculated.fillna(base_fallback).astype(float)
        universe[fallback_column] = fallback
        universe[trim_column] = trim_fallback
        universe[f"source_count__{method_name}"] = source_count
        methods.append(
            ProjectionMethod(method_name, "consensus", value_column, fallback_column, trim_column)
        )
        for category in required_counts:
            mask = universe["category"].eq(category)
            available = int((source_count.loc[mask] > 0).sum())
            total = int(mask.sum())
            coverage_rows.append(
                {
                    "projection_method": method_name,
                    "projection_type": "consensus",
                    "category": category,
                    "available_count": available,
                    "universe_count": total,
                    "coverage_pct": available / total if total else 0.0,
                    "minimum_required": required_counts[category],
                    "category_eligible": available >= required_counts[category],
                    "fallback_count": int(fallback.loc[mask].sum()),
                    "fallback_pct": float(fallback.loc[mask].mean()) if total else 0.0,
                    "trim_rule_fallback_count": int(trim_fallback.loc[mask].sum()),
                }
            )

    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["projection_type", "projection_method", "category"]
    ).reset_index(drop=True)
    exclusions = pd.DataFrame(
        exclusion_rows,
        columns=[
            "projection_method",
            "category",
            "available_count",
            "minimum_required",
            "reason",
        ],
    )
    return PreparedExperiment(
        universe=universe.reset_index(drop=True),
        methods=tuple(methods),
        coverage=coverage,
        exclusions=exclusions,
        opponent_value_column="opponent_projection",
        source_columns=source_columns,
    )


def _assets_for_experiment(prepared: PreparedExperiment) -> list[DraftAsset]:
    method_columns = [method.value_column for method in prepared.methods]
    flag_columns = [
        column
        for method in prepared.methods
        for column in (method.fallback_column, method.trim_fallback_column)
    ]
    assets: list[DraftAsset] = []
    for row in prepared.universe.to_dict(orient="records"):
        metadata = {
            "actual_points": float(row["actual_points"]),
            prepared.opponent_value_column: float(row[prepared.opponent_value_column]),
        }
        metadata.update({column: float(row[column]) for column in method_columns})
        metadata.update({column: bool(row[column]) for column in flag_columns})
        assets.append(
            DraftAsset(
                entity_id=str(row["entity_id"]),
                name=str(row["name"]),
                category=str(row["category"]),
                value=float(row[prepared.opponent_value_column]),
                nhl_team=str(row.get("team_key", "")),
                metadata=metadata,
            )
        )
    return assets


def _competition_rank(totals: Mapping[int, float], gm_id: int) -> int:
    focal = totals[gm_id]
    return 1 + sum(value > focal for other, value in totals.items() if other != gm_id)


def run_projection_source_drafts(
    prepared: PreparedExperiment,
    *,
    gm_count: int,
    roster_config: RosterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = _assets_for_experiment(prepared)
    detail_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []

    for method in prepared.methods:
        for focal_slot in range(1, gm_count + 1):
            strategies = {
                gm_id: build_strategy("vorp") for gm_id in range(1, gm_count + 1)
            }
            strategies[focal_slot] = _ProjectionViewStrategy(
                strategies[focal_slot],
                value_column=method.value_column,
                gm_count=gm_count,
            )
            result = run_draft(assets, gm_count, roster_config, strategies)
            actual_totals = {
                gm_id: float(sum(a.metadata["actual_points"] for a in roster))
                for gm_id, roster in result.rosters.items()
            }
            focal_roster = result.rosters[focal_slot]
            focal_points = actual_totals[focal_slot]
            field = [
                points for gm_id, points in actual_totals.items() if gm_id != focal_slot
            ]
            category_points = {
                category: float(
                    sum(
                        a.metadata["actual_points"]
                        for a in focal_roster
                        if a.category == category
                    )
                )
                for category in roster_config.counts
            }
            detail_rows.append(
                {
                    "draft_slot": focal_slot,
                    "projection_method": method.name,
                    "projection_type": method.projection_type,
                    "strategy": "vorp",
                    "opponent_projection_method": "PoolPro",
                    "final_points": focal_points,
                    "final_rank": _competition_rank(actual_totals, focal_slot),
                    "winner_points": max(actual_totals.values()),
                    "gap_to_winner": max(actual_totals.values()) - focal_points,
                    "field_mean_points": fmean(field),
                    "margin_vs_field": focal_points - fmean(field),
                    "projected_roster_points": float(
                        sum(a.metadata[method.value_column] for a in focal_roster)
                    ),
                    "fallback_picks": sum(
                        bool(a.metadata[method.fallback_column]) for a in focal_roster
                    ),
                    "trim_rule_fallback_picks": sum(
                        bool(a.metadata[method.trim_fallback_column]) for a in focal_roster
                    ),
                    **{f"actual_points_{category}": points for category, points in category_points.items()},
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
                        "overall_pick": pick.overall_pick,
                        "round_no": pick.round_no,
                        "category": pick.asset.category,
                        "entity_id": pick.asset.entity_id,
                        "name": pick.asset.name,
                        "decision_value": pick.asset.metadata[method.value_column],
                        "actual_points": pick.asset.metadata["actual_points"],
                        "used_fallback": pick.asset.metadata[method.fallback_column],
                        "used_trim_rule_fallback": pick.asset.metadata[
                            method.trim_fallback_column
                        ],
                    }
                )

    details = pd.DataFrame(detail_rows).sort_values(
        ["draft_slot", "projection_type", "projection_method"]
    ).reset_index(drop=True)
    picks = pd.DataFrame(pick_rows).sort_values(
        ["projection_method", "draft_slot", "overall_pick"]
    ).reset_index(drop=True)
    return details, picks


def summarize_methods(details: pd.DataFrame) -> pd.DataFrame:
    work = details.assign(
        win=details["final_rank"].eq(1).astype(float),
        top_3=details["final_rank"].le(3).astype(float),
        last_place=details["final_rank"].eq(details["final_rank"].max()).astype(float),
    )
    return (
        work.groupby(["projection_method", "projection_type"], as_index=False)
        .agg(
            simulations=("draft_slot", "size"),
            mean_final_points=("final_points", "mean"),
            median_final_points=("final_points", "median"),
            std_final_points=("final_points", "std"),
            min_final_points=("final_points", "min"),
            mean_rank=("final_rank", "mean"),
            median_rank=("final_rank", "median"),
            worst_rank=("final_rank", "max"),
            win_rate=("win", "mean"),
            top_3_rate=("top_3", "mean"),
            last_place_rate=("last_place", "mean"),
            mean_gap_to_winner=("gap_to_winner", "mean"),
            mean_margin_vs_field=("margin_vs_field", "mean"),
        )
        .sort_values(["mean_rank", "mean_final_points"], ascending=[True, False])
        .reset_index(drop=True)
    )


def results_by_slot(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.pivot(index="draft_slot", columns="projection_method", values="final_rank")
        .reset_index()
        .rename_axis(columns=None)
        .sort_values("draft_slot")
    )


def consensus_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    singles = summary[summary["projection_type"].eq("single_source")].copy()
    consensus = summary[summary["projection_type"].eq("consensus")].copy()
    metric_columns = [
        "mean_final_points",
        "median_final_points",
        "std_final_points",
        "min_final_points",
        "mean_rank",
        "median_rank",
        "worst_rank",
        "win_rate",
        "top_3_rate",
        "last_place_rate",
        "mean_gap_to_winner",
        "mean_margin_vs_field",
    ]
    best = singles.sort_values(["mean_rank", "mean_final_points"], ascending=[True, False]).iloc[0]
    rows = [
        {
            "comparison_group": "best_single_source",
            "projection_method": best["projection_method"],
            **{column: best[column] for column in metric_columns},
        },
        {
            "comparison_group": "average_single_sources",
            "projection_method": "average_single_sources",
            **{column: singles[column].mean() for column in metric_columns},
        },
    ]
    for row in consensus.itertuples(index=False):
        rows.append(
            {
                "comparison_group": "consensus",
                "projection_method": row.projection_method,
                **{column: getattr(row, column) for column in metric_columns},
            }
        )
    output = pd.DataFrame(rows)
    average = output.loc[
        output["comparison_group"].eq("average_single_sources")
    ].iloc[0]
    output["mean_points_delta_vs_average_single"] = (
        output["mean_final_points"] - average["mean_final_points"]
    )
    output["mean_rank_delta_vs_average_single"] = (
        output["mean_rank"] - average["mean_rank"]
    )
    output["top_3_rate_delta_vs_average_single"] = (
        output["top_3_rate"] - average["top_3_rate"]
    )
    return output


def consensus_by_slot(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slot, group in details.groupby("draft_slot"):
        singles = group[group["projection_type"].eq("single_source")]
        consensus = group[group["projection_type"].eq("consensus")]
        best_consensus = consensus.sort_values(
            ["final_rank", "final_points"], ascending=[True, False]
        ).iloc[0]
        rows.append(
            {
                "draft_slot": int(slot),
                "mean_single_points": singles["final_points"].mean(),
                "mean_consensus_points": consensus["final_points"].mean(),
                "consensus_points_delta": (
                    consensus["final_points"].mean() - singles["final_points"].mean()
                ),
                "mean_single_rank": singles["final_rank"].mean(),
                "mean_consensus_rank": consensus["final_rank"].mean(),
                "consensus_rank_delta": (
                    consensus["final_rank"].mean() - singles["final_rank"].mean()
                ),
                "best_consensus_method": best_consensus["projection_method"],
                "best_consensus_rank": best_consensus["final_rank"],
            }
        )
    return pd.DataFrame(rows).sort_values("draft_slot").reset_index(drop=True)


def build_report(
    *,
    master_path: Path,
    output_dir: Path,
    gm_count: int,
    roster_config: RosterConfig,
    trim_min_sources: int,
    random_seed: int,
    season_label: str,
    prepared: PreparedExperiment,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    by_slot_comparison: pd.DataFrame,
) -> str:
    best = summary.iloc[0]
    best_single = summary[summary["projection_type"].eq("single_source")].iloc[0]
    best_consensus = summary[summary["projection_type"].eq("consensus")].iloc[0]
    favorable_slots = int((by_slot_comparison["consensus_rank_delta"] < 0).sum())
    average_single = comparison.loc[
        comparison["comparison_group"].eq("average_single_sources")
    ].iloc[0]
    consensus_beats_best_single = (
        best_consensus.mean_rank < best_single.mean_rank
        or (
            best_consensus.mean_rank == best_single.mean_rank
            and best_consensus.mean_final_points > best_single.mean_final_points
        )
    )
    best_single_comparison = (
        "et surpasse la meilleure source individuelle."
        if consensus_beats_best_single
        else "mais ne dépasse pas la meilleure source individuelle."
    )
    exclusions = (
        "Aucune."
        if prepared.exclusions.empty
        else "; ".join(
            f"{row.projection_method}/{row.category} ({row.available_count} < {row.minimum_required})"
            for row in prepared.exclusions.itertuples(index=False)
        )
    )
    table_columns = [
        "projection_method",
        "projection_type",
        "mean_final_points",
        "mean_rank",
        "win_rate",
        "top_3_rate",
        "mean_gap_to_winner",
        "min_final_points",
        "worst_rank",
    ]
    report_table = summary[table_columns].copy()
    for column in table_columns[2:]:
        report_table[column] = report_table[column].map(lambda value: f"{value:.3f}")
    comparison_table = comparison[
        [
            "comparison_group",
            "projection_method",
            "mean_final_points",
            "mean_rank",
            "top_3_rate",
            "min_final_points",
        ]
    ].copy()
    for column in comparison_table.columns[2:]:
        comparison_table[column] = comparison_table[column].map(lambda value: f"{value:.3f}")

    return "\n".join(
        [
            "# Test — source unique vs consensus de projections",
            "",
            "## Cadre",
            "",
            f"- Données: `{master_path.as_posix()}` (projections et résultats réels {season_label}).",
            f"- Format: {gm_count} GMs; roster {dict(roster_config.counts)}; draft serpentin.",
            "- Tous les GMs utilisent le VOR brut. Les adversaires utilisent Pool Pro; seule la projection du GM testé change.",
            "- Les rosters sont évalués avec `actual_points`, jamais avec la projection de draft.",
            "- L'expérience est déterministe; le seed documenté est " + str(random_seed) + ".",
            f"- Consensus tronqués: retrait min/max à partir de {trim_min_sources} sources; sinon retour explicite à la version non tronquée.",
            "- Valeur manquante: méthode testée, puis Pool Pro, puis médiane consolidée. Aucun zéro implicite.",
            "- Admissibilité par catégorie: la source doit couvrir au moins tout le besoin du draft (GMs × places de roster).",
            "",
            "## Couverture et exclusions",
            "",
            f"Sources découvertes automatiquement: {', '.join(prepared.source_columns)}.",
            f"Catégories-source exclues au profit du fallback: {exclusions}",
            "Voir `04_projection_coverage.csv` et `05_projection_exclusions.csv` pour les détails.",
            "",
            "## Résultats",
            "",
            report_table.to_markdown(index=False),
            "",
            "## Consensus vs sources individuelles",
            "",
            comparison_table.to_markdown(index=False),
            "",
            f"Meilleure méthode globale au rang moyen: **{best.projection_method}** "
            f"(rang {best.mean_rank:.3f}, {best.mean_final_points:.3f} points moyens).",
            f"Meilleure source individuelle: **{best_single.projection_method}**; "
            f"meilleur consensus: **{best_consensus.projection_method}**.",
            f"La meilleure source individuelle termine dans le top 3 dans {best_single.top_3_rate:.1%} des slots.",
            f"Le meilleur consensus améliore le rang moyen de {average_single.mean_rank - best_consensus.mean_rank:.3f} "
            f"place(s) par rapport à la moyenne des sources individuelles, {best_single_comparison}",
            f"Le consensus obtient un meilleur rang moyen que la moyenne des sources individuelles dans {favorable_slots}/16 slots.",
            "`median` et `trimmed_median_minmax` donnent ici exactement le même résultat: retirer symétriquement le minimum et le maximum ne change pas la médiane des valeurs restantes.",
            "Le taux de victoire porte sur seulement 16 simulations par méthode; il ne doit pas être interprété seul.",
            "",
            "## Limites",
            "",
            "Il s'agit d'un backtest sur une seule saison avec 16 positions de départ déterministes, pas de 16 saisons indépendantes. Les résultats ne concernent que les sources présentes et comparables dans le master historique; les sources prospectives plus récentes ne peuvent être ajoutées rétroactivement.",
            "",
            f"Fichiers détaillés: `{output_dir.as_posix()}`.",
        ]
    )


def run_projection_source_experiment(
    *,
    master_path: Path,
    output_dir: Path,
    gm_count: int = 16,
    roster_config: RosterConfig = RosterConfig({"F": 10, "D": 3, "G": 2, "T": 1}),
    opponent_source: str = "PoolPro",
    trim_min_sources: int = DEFAULT_TRIM_MIN_SOURCES,
    random_seed: int = 20260927,
    season_label: str = "2025-2026",
) -> dict[str, pd.DataFrame]:
    master = pd.read_csv(master_path, low_memory=False)
    prepared = prepare_projection_experiment(
        master,
        gm_count=gm_count,
        roster_config=roster_config,
        opponent_source=opponent_source,
        trim_min_sources=trim_min_sources,
    )
    details, picks = run_projection_source_drafts(
        prepared,
        gm_count=gm_count,
        roster_config=roster_config,
    )
    summary = summarize_methods(details)
    by_slot = results_by_slot(details)
    comparison = consensus_comparison(summary)
    by_slot_comparison = consensus_by_slot(details)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "details": details,
        "summary": summary,
        "by_slot": by_slot,
        "coverage": prepared.coverage,
        "exclusions": prepared.exclusions,
        "consensus_comparison": comparison,
        "consensus_by_slot": by_slot_comparison,
        "picks": picks,
    }
    filenames = {
        "details": "01_projection_source_details.csv",
        "summary": "02_projection_source_summary.csv",
        "by_slot": "03_projection_source_by_slot.csv",
        "coverage": "04_projection_coverage.csv",
        "exclusions": "05_projection_exclusions.csv",
        "consensus_comparison": "06_consensus_vs_single.csv",
        "consensus_by_slot": "07_consensus_vs_single_by_slot.csv",
        "picks": "08_projection_source_picks.csv",
    }
    for key, frame in outputs.items():
        frame.to_csv(output_dir / filenames[key], index=False)

    report = build_report(
        master_path=master_path,
        output_dir=output_dir,
        gm_count=gm_count,
        roster_config=roster_config,
        trim_min_sources=trim_min_sources,
        random_seed=random_seed,
        season_label=season_label,
        prepared=prepared,
        summary=summary,
        comparison=comparison,
        by_slot_comparison=by_slot_comparison,
    )
    (output_dir / "09_projection_source_report.md").write_text(report, encoding="utf-8")
    return outputs
