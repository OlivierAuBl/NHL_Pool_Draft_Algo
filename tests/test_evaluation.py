from __future__ import annotations

import pandas as pd
import pytest

from nhl_draft_lab.evaluation import (
    active_universe_projections,
    add_v0_ppg_v1_gp_hybrid,
    add_weighted_gp_candidate,
    blend_grid_metrics,
    build_v1_comparison,
    coverage_summary,
    draft_zone_disagreements,
    draft_zone_metrics,
    gp_correction_grid_metrics,
    hybrid_active_universe_projections,
    projection_metrics,
    skater_component_errors,
    weighted_gp_candidate_projections,
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
            # Older ranking databases can have no team abbreviation.  The
            # evaluator must recover it from V0's team name mapping.
            "entity_id": "30", "category": "T", "nhl_team": "",
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


def test_hybrid_uses_v0_rate_v1_gp_and_rookie_gp_default():
    master = pd.concat([
        v0_master(),
        pd.DataFrame([{
            "category": "D", "NHLID": 4, "team_key": "DDD",
            "FullName": "Rookie Defenseman", "projection_median": 42,
            "actual_points": 30, "games_played": 60, "goals": 5, "assists": 25,
        }]),
    ], ignore_index=True)
    comparison = add_v0_ppg_v1_gp_hybrid(
        build_v1_comparison(master, v1_projections()),
        v0_reference_games=84,
        rookie_gp=50,
        rookie_defense_gp=60,
    )

    veteran = comparison.loc[comparison["NHLID"] == 1].iloc[0]
    rookie = comparison.loc[comparison["NHLID"] == 2].iloc[0]
    goalie = comparison.loc[comparison["NHLID"] == 3].iloc[0]
    rookie_defenseman = comparison.loc[comparison["NHLID"] == 4].iloc[0]
    assert veteran["hybrid_projected_points"] == pytest.approx(90 / 84 * 80)
    assert veteran["hybrid_gp_source"] == "V1_HISTORY"
    assert rookie["hybrid_projected_points"] == pytest.approx(40 / 84 * 50)
    assert rookie["hybrid_gp_source"] == "NO_HISTORY_F_DEFAULT"
    assert rookie_defenseman["hybrid_projected_points"] == pytest.approx(42 / 84 * 60)
    assert rookie_defenseman["hybrid_gp_source"] == "NO_HISTORY_D_DEFAULT"
    assert goalie["hybrid_projected_points"] == 60


def test_hybrid_is_exported_in_projection_loader_contract():
    comparison = add_v0_ppg_v1_gp_hybrid(
        build_v1_comparison(v0_master(), v1_projections())
    )
    hybrid = hybrid_active_universe_projections(comparison)

    assert len(hybrid) == 4
    assert {"entity_id", "name", "category", "projected_points", "stddev_points"} <= set(hybrid)
    assert set(hybrid["projection_source"]) == {"V0_PPG_X_V1_GP"}


def test_hybrid_assumptions_must_be_valid():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    with pytest.raises(ValueError, match="v0_reference_games must be positive"):
        add_v0_ppg_v1_gp_hybrid(comparison, v0_reference_games=0)
    with pytest.raises(ValueError, match="rookie_gp cannot be negative"):
        add_v0_ppg_v1_gp_hybrid(comparison, rookie_gp=-1)
    with pytest.raises(ValueError, match="rookie_gp cannot exceed"):
        add_v0_ppg_v1_gp_hybrid(comparison, v0_reference_games=84, rookie_gp=85)
    with pytest.raises(ValueError, match="rookie_defense_gp cannot exceed"):
        add_v0_ppg_v1_gp_hybrid(
            comparison, v0_reference_games=84, rookie_defense_gp=85
        )


def test_weighted_gp_candidate_uses_category_weight_and_fixed_no_history_gp():
    comparison = add_v0_ppg_v1_gp_hybrid(
        build_v1_comparison(v0_master(), v1_projections()),
        v0_reference_games=84,
        rookie_gp=50,
        rookie_defense_gp=60,
    )
    candidate = add_weighted_gp_candidate(
        comparison, forward_gp_weight=0.30, defense_gp_weight=0.60
    )

    veteran = candidate.loc[candidate["NHLID"] == 1].iloc[0]
    rookie = candidate.loc[candidate["NHLID"] == 2].iloc[0]
    goalie = candidate.loc[candidate["NHLID"] == 3].iloc[0]
    full_hybrid = 90 / 84 * 80
    assert veteran["candidate_projected_points"] == pytest.approx(
        90 + 0.30 * (full_hybrid - 90)
    )
    assert veteran["candidate_gp_weight"] == pytest.approx(0.30)
    assert rookie["candidate_projected_points"] == pytest.approx(40 / 84 * 50)
    assert rookie["candidate_projection_source"] == "NO_HISTORY_F_DEFAULT"
    assert goalie["candidate_projected_points"] == 60


def test_weighted_gp_candidate_is_exported_for_the_draft_engine():
    comparison = add_weighted_gp_candidate(
        add_v0_ppg_v1_gp_hybrid(
            build_v1_comparison(v0_master(), v1_projections())
        )
    )
    candidate = weighted_gp_candidate_projections(comparison)

    assert len(candidate) == 4
    assert {"entity_id", "name", "category", "projected_points", "stddev_points"} <= set(candidate)
    assert candidate.loc[candidate["entity_id"] == "P:1", "gp_weight"].iloc[0] == pytest.approx(0.30)

    metrics = projection_metrics(comparison)
    assert "V1_weighted_gp_candidate" in set(metrics["model"])


def test_weighted_gp_candidate_rejects_non_convex_weights():
    comparison = add_v0_ppg_v1_gp_hybrid(
        build_v1_comparison(v0_master(), v1_projections())
    )
    with pytest.raises(ValueError, match="forward_gp_weight must be between 0 and 1"):
        add_weighted_gp_candidate(comparison, forward_gp_weight=1.1)


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


def test_legacy_arizona_abbreviation_matches_utah():
    master = v0_master().copy()
    master.loc[master["category"] == "T", "team_key"] = "UTA"
    master.loc[master["category"] == "T", "Team"] = "Utah Mammoth"
    projections = v1_projections().copy()
    projections.loc[projections["category"] == "T", "nhl_team"] = "ARI"
    projections.loc[projections["category"] == "T", "name"] = "Utah Hockey Club"

    comparison = build_v1_comparison(master, projections)
    utah = comparison.loc[comparison["category"] == "T"].iloc[0]

    assert utah["v1_available"]
    assert utah["v1_projected_points"] == 88


def test_draft_zone_excludes_players_below_category_cutoff():
    comparison = build_v1_comparison(v0_master(), v1_projections())

    metrics = draft_zone_metrics(
        comparison,
        draft_counts={"F": 1, "G": 1, "T": 1},
        defense_focus_ranks=(6, 9),
    )
    overall_v0 = metrics.loc[
        metrics["segment"].eq("DRAFTABLE")
        & metrics["category"].eq("ALL")
        & metrics["model"].eq("V0_full_universe")
    ].iloc[0]

    assert overall_v0["segment_assets"] == 3
    assert overall_v0["n"] == 3


def test_defense_focus_windows_are_reported_separately():
    master = pd.concat([
        v0_master(),
        pd.DataFrame([
            {
                "category": "D", "NHLID": number, "team_key": "AAA",
                "FullName": f"D{number}", "projection_median": 100 - number,
                "actual_points": 90 - number, "games_played": 82,
                "goals": 10, "assists": 30,
            }
            for number in range(10, 20)
        ]),
    ], ignore_index=True)
    projections = pd.concat([
        v1_projections(),
        pd.DataFrame([
            {
                "entity_id": str(number), "category": "D", "nhl_team": "AAA",
                "name": f"D{number}", "projected_points": 92 - number,
            }
            for number in range(10, 20)
        ]),
    ], ignore_index=True)
    comparison = build_v1_comparison(master, projections)

    metrics = draft_zone_metrics(
        comparison,
        draft_counts={"D": 10},
        defense_focus_ranks=(6, 9),
    )

    top6 = metrics.loc[metrics["segment"].eq("D_TOP_6")].iloc[0]
    top9 = metrics.loc[metrics["segment"].eq("D_TOP_9")].iloc[0]
    assert top6["segment_assets"] == 6
    assert top9["segment_assets"] == 9


def test_blend_grid_keeps_v0_endpoint_and_marks_best_in_sample_weight():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    grid = blend_grid_metrics(
        comparison,
        draft_counts={"F": 2, "G": 1, "T": 1},
        weights=(0.0, 0.5, 1.0),
    )
    overall = grid.loc[
        grid["segment"].eq("DRAFTABLE") & grid["category"].eq("ALL")
    ].set_index("v1_weight")

    assert overall.loc[0.0, "mae"] == pytest.approx(20 / 3)
    assert overall.loc[1.0, "mae"] == pytest.approx(4)
    assert overall.loc[1.0, "is_best_mae_in_sample"]


def test_blend_weights_must_be_convex():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_grid_metrics(comparison, {"F": 2}, weights=(-0.1, 0.5))


def test_gp_correction_grid_keeps_no_history_default_fixed():
    comparison = add_v0_ppg_v1_gp_hybrid(
        build_v1_comparison(v0_master(), v1_projections()),
        v0_reference_games=84,
        rookie_gp=50,
        rookie_defense_gp=60,
    )
    grid = gp_correction_grid_metrics(
        comparison,
        draft_counts={"F": 2},
        weights=(0.0, 0.5, 1.0),
    )
    forwards = grid.loc[
        grid["segment"].eq("DRAFTABLE") & grid["category"].eq("F")
    ].set_index("history_gp_weight")

    rookie_error = abs(30 - 40 / 84 * 50)
    assert forwards.loc[0.0, "mae"] == pytest.approx((10 + rookie_error) / 2)
    assert forwards.loc[1.0, "mae"] == pytest.approx(
        (abs(100 - 90 / 84 * 80) + rookie_error) / 2
    )
    assert set(forwards["fixed_no_history_assets"]) == {1}
    assert forwards.loc[0.0, "is_best_mae_in_sample"]


def test_gp_correction_weights_must_be_convex():
    comparison = add_v0_ppg_v1_gp_hybrid(
        build_v1_comparison(v0_master(), v1_projections())
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        gp_correction_grid_metrics(comparison, {"F": 2}, weights=(0.0, 1.1))


def test_disagreement_detail_stays_inside_draft_zone():
    comparison = build_v1_comparison(v0_master(), v1_projections())
    detail = draft_zone_disagreements(
        comparison,
        draft_counts={"F": 1, "G": 1, "T": 1},
    )

    assert len(detail) == 3
    assert set(detail["display_name"]) == {"Veteran", "Goalie", "Team"}
    assert detail.iloc[0]["absolute_v1_v0_disagreement"] >= detail.iloc[-1][
        "absolute_v1_v0_disagreement"
    ]
