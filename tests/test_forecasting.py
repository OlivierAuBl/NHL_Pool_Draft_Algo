from __future__ import annotations

import pandas as pd
import pytest

from nhl_draft_lab.forecasting import ForecastConfig, project_v1
from nhl_draft_lab.data.projections import load_projection_csv


def history_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season_id": 20232024, "category": "F", "entity_id": "f1",
            "name": "Forward One", "nhl_team": "AAA", "games_played": 60,
            "goals": 10, "assists": 20, "power_play_points": 6, "shots": 100,
            "pool_points_raw": 30,
        },
        {
            "season_id": 20242025, "category": "F", "entity_id": "f1",
            "name": "Forward One", "nhl_team": "AAA", "games_played": 80,
            "goals": 30, "assists": 50, "power_play_points": 24, "shots": 200,
            "pool_points_raw": 80,
        },
        {
            "season_id": 20232024, "category": "G", "entity_id": "g1",
            "name": "Goalie One", "nhl_team": "BBB", "games_played": 45,
            "games_started": 40, "wins": 20, "shutouts": 4, "ot_losses": 4,
            "goals": 0, "assists": 0, "pool_points_raw": 56,
        },
        {
            "season_id": 20242025, "category": "G", "entity_id": "g1",
            "name": "Goalie One", "nhl_team": "BBB", "games_played": 55,
            "games_started": 50, "wins": 30, "shutouts": 5, "ot_losses": 5,
            "goals": 0, "assists": 0, "pool_points_raw": 80,
        },
        {
            "season_id": 20242025, "category": "T", "entity_id": "t1",
            "name": "Team One", "nhl_team": "CCC", "games_played": 82,
            "wins": 40, "ot_losses": 10, "pool_points_raw": 90,
        },
    ])


def test_skater_projects_ppg_and_gp_as_separate_components():
    result = project_v1(
        history_frame(),
        config=ForecastConfig(recency_weights=(2, 1)),
    )
    skater = result.loc[result["entity_id"] == "f1"].iloc[0]

    assert skater["projected_ppg"] == pytest.approx((2 * 1.0 + 1 * 0.5) / 3)
    assert skater["projected_gp"] == pytest.approx((2 * 80 + 1 * 60) / 3)
    assert skater["projected_points"] == pytest.approx(
        skater["projected_ppg"] * skater["projected_gp"]
    )
    assert skater["historical_pp_points_per_game"] == pytest.approx(
        (2 * 24 / 80 + 1 * 6 / 60) / 3
    )


def test_goalie_projects_start_share_and_scoring_rates():
    result = project_v1(
        history_frame(),
        config=ForecastConfig(recency_weights=(2, 1)),
    )
    goalie = result.loc[result["entity_id"] == "g1"].iloc[0]

    assert goalie["projected_starts"] == pytest.approx((2 * 50 + 1 * 40) / 3)
    assert goalie["projected_win_rate"] == pytest.approx((2 * 30 / 50 + 1 * 20 / 40) / 3)
    assert goalie["projected_shutout_rate"] == pytest.approx(0.1)
    assert goalie["projected_otl_rate"] == pytest.approx(0.1)
    assert goalie["starts_source"] == "games_started"


def test_manual_context_can_override_injury_and_future_role():
    manual = pd.DataFrame([
        {
            "entity_id": "f1",
            "projected_ppg_override": 1.1,
            "availability": 70 / 82,
            "projected_line": 1,
            "projected_pp_unit": 1,
            "future_linemate_quality": 0.9,
        },
        {
            "entity_id": "g1",
            "availability": 0.75,
            "projected_start_share_override": 0.60,
            "projected_win_rate_override": 0.55,
        },
    ])

    result = project_v1(history_frame(), manual_context=manual)
    skater = result.loc[result["entity_id"] == "f1"].iloc[0]
    goalie = result.loc[result["entity_id"] == "g1"].iloc[0]

    assert skater["projected_points"] == pytest.approx(77.0)
    assert skater["projected_gp"] == pytest.approx(70.0)
    assert skater["projected_pp_unit"] == 1
    assert goalie["projected_starts"] == pytest.approx(82 * 0.60 * 0.75)
    assert goalie["availability"] == pytest.approx(0.75)
    assert goalie["projected_win_rate"] == pytest.approx(0.55)


def test_manual_context_can_match_a_player_by_name_for_auditable_notes():
    manual = pd.DataFrame([{
        "name": "Forward One",
        "pp_role_note": "Moved from QB2 to QB1",
        "injury_note": "Known injury context",
    }])

    result = project_v1(history_frame(), manual_context=manual)
    skater = result.loc[result["entity_id"] == "f1"].iloc[0]

    assert skater["pp_role_note"] == "Moved from QB2 to QB1"
    assert skater["injury_note"] == "Known injury context"


def test_goalie_falls_back_to_games_played_when_starts_are_unavailable():
    history = history_frame().drop(columns="games_started")
    result = project_v1(history)
    goalie = result.loc[result["entity_id"] == "g1"].iloc[0]
    assert goalie["starts_source"] == "games_played_fallback"


def test_manual_rates_must_be_probabilities():
    manual = pd.DataFrame([{"entity_id": "g1", "availability": 1.2}])
    with pytest.raises(ValueError, match="availability must be between 0 and 1"):
        project_v1(history_frame(), manual_context=manual)


def test_team_projection_keeps_team_category_in_draft_universe():
    result = project_v1(history_frame())
    team = result.loc[result["entity_id"] == "t1"].iloc[0]
    assert team["projected_points"] == pytest.approx(90.0)


def test_shortened_season_gp_and_start_share_are_normalized_to_target_schedule():
    history = pd.DataFrame([
        {
            "season_id": 20202021, "category": "T", "entity_id": "t1",
            "name": "Team", "games_played": 56, "wins": 30, "ot_losses": 5,
        },
        {
            "season_id": 20202021, "category": "F", "entity_id": "f1",
            "name": "Forward", "games_played": 56, "goals": 20, "assists": 20,
        },
        {
            "season_id": 20202021, "category": "G", "entity_id": "g1",
            "name": "Goalie", "games_played": 30, "games_started": 28,
            "wins": 15, "shutouts": 2, "ot_losses": 3,
        },
    ])
    result = project_v1(history)
    skater = result.loc[result["entity_id"] == "f1"].iloc[0]
    goalie = result.loc[result["entity_id"] == "g1"].iloc[0]

    assert skater["projected_gp"] == pytest.approx(82)
    assert goalie["projected_start_share"] == pytest.approx(0.5)
    assert goalie["projected_starts"] == pytest.approx(41)


def test_v1_csv_is_compatible_with_existing_projection_loader(tmp_path):
    path = tmp_path / "v1.csv"
    project_v1(history_frame()).to_csv(path, index=False)

    projections = load_projection_csv(path)

    assert len(projections) == 3
    assert {projection.asset.category for projection in projections} == {"F", "G", "T"}
