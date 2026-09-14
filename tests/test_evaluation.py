from __future__ import annotations

import pandas as pd
import pytest

from nhl_draft_lab.evaluation import (
    active_universe_projections,
    build_v1_comparison,
    coverage_summary,
    projection_metrics,
    skater_component_errors,
)


def v0_master() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "category": "F", "NHLID": 1, "team_key": "AAA",
            "FullName": "Veteran", "projection_median": 90, "actual_points": 100,
            "games_played": 82, "goals": 40, "assists": 60,
        },
        {
            "category": "F", "NHLID": 2, "team_key": "BBB",
            "FullName": "Rookie", "projection_median": 40, "actual_points": 30,
            "games_played": 60, "goals": 10, "assists": 20,
        },
        {
            "category": "G", "NHLID": 3, "team_key": "CCC",
            "FullName": "Goalie", "projection_median": 60, "actual_points": 50,
            "games_played": 40, "goals": 0, "assists": 0,
        },
        {
            "category": "T", "NHLID": pd.NA, "team_key": "DDD",
            "Team": "Team", "projection_median": 90, "actual_points": 90,
            "games_played": 82, "goals": 0, "assists": 0,
        },
    ])


def v1_projections() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "entity_id": "1", "category": "F", "nhl_team": "AAA",
            "name": "Veteran", "projected_points": 95,
            "projected_ppg": 1.1, "projected_gp": 80,
        },
        {
            "entity_id": "3", "category": "G", "nhl_team": "CCC",
            "name": "Goalie", "projected_points": 55,
            "projected_starts": 42, "projected_start_share": 42 / 82,
        },
        {
            "entity_id": "30", "category": "T", "nhl_team": "DDD",
            "name": "Team", "projected_points": 88,
        },
        {
            "entity_id": "999", "category": "F", "nhl_team": "ZZZ",
            "name": "Inactive historical player", "projected_points": 70,
        },
    ])


def test_comparison_uses_v0_as_target_universe_and_falls_back_for_rookies():
    comparison = build_v1_comparison(v0_master(), v1_projections())

    assert len(comparison) == 4
    assert "Inactive historical player" not in set(comparison.get("v1_name", []))
    rookie = comparison.loc[comparison["NHLID"] == 2].iloc[0]
    assert not rookie["v1_available"]
    assert rookie["v1_with_v0_fallback_points"] == 40


def test_metrics_compare_v0_and_v1_on_identical_coverage():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    metrics = projection_metrics(comparison)
    overall = metrics.loc[metrics["category"] == "ALL"].set_index("model")

    assert overall.loc["V0_same_V1_coverage", "n"] == 3
    assert overall.loc["V0_same_V1_coverage", "mae"] == pytest.approx(20 / 3)
    assert overall.loc["V1_history_components", "mae"] == pytest.approx(4)
    assert overall.loc["V1_with_V0_fallback_full_universe", "mae"] == pytest.approx(5.5)
    assert overall.loc["V0_full_universe", "mae"] == pytest.approx(7.5)


def test_component_output_separates_skater_gp_and_ppg_errors():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    components = skater_component_errors(comparison)

    assert list(components["FullName"]) == ["Veteran"]
    assert components.iloc[0]["actual_ppg"] == pytest.approx(100 / 82)
    assert components.iloc[0]["gp_error_actual_minus_projection"] == pytest.approx(2)


def test_coverage_reports_missing_history_separately():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    coverage = coverage_summary(comparison).set_index("category")

    assert coverage.loc["ALL", "v0_universe"] == 4
    assert coverage.loc["ALL", "v1_history_available"] == 3
    assert coverage.loc["F", "v0_fallback_required"] == 1


def test_active_universe_output_uses_existing_projection_contract():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    active = active_universe_projections(comparison)

    assert len(active) == 4
    assert {"entity_id", "name", "category", "projected_points", "stddev_points"} <= set(active)
    rookie = active.loc[active["entity_id"] == "P:2"].iloc[0]
    assert rookie["projection_source"] == "V0_FALLBACK"
    assert rookie["projected_points"] == 40
