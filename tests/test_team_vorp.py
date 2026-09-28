from __future__ import annotations

from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.projection_source_experiment import prepare_projection_experiment
from nhl_draft_lab.strategies.team_vorp import TeamAwareVorpStrategy
from nhl_draft_lab.team_field_experiment import (
    FIELD_REGIMES,
    paired_field_deltas,
    run_team_field_drafts,
)
from nhl_draft_lab.team_tiebreak_experiment import (
    SELECTION_SCENARIOS,
    paired_scenario_deltas,
    run_team_tiebreak_drafts,
)
import pandas as pd


def _context(available: list[DraftAsset], roster: list[DraftAsset]) -> DraftContext:
    return DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=3,
        round_no=2,
        picks_until_next=1,
        roster=tuple(roster),
        roster_config=RosterConfig({"F": 3}),
        available_assets=tuple(available),
        replacement_levels={"F": 50},
    )


def test_stack_prefers_an_existing_team_inside_three_vor_points():
    roster = [DraftAsset("r", "Roster", "F", 90, nhl_team="EDM")]
    best = DraftAsset("a", "Best", "F", 100, nhl_team="COL")
    stack = DraftAsset("b", "Stack", "F", 98, nhl_team="EDM")

    chosen = TeamAwareVorpStrategy("stack", 3).choose(_context([best, stack], roster))

    assert chosen == stack


def test_diversify_avoids_an_existing_team_inside_three_vor_points():
    roster = [DraftAsset("r", "Roster", "F", 90, nhl_team="EDM")]
    concentrated = DraftAsset("a", "Concentrated", "F", 100, nhl_team="EDM")
    diverse = DraftAsset("b", "Diverse", "F", 98, nhl_team="COL")

    chosen = TeamAwareVorpStrategy("diversify", 3).choose(
        _context([concentrated, diverse], roster)
    )

    assert chosen == diverse


def test_team_preference_never_reaches_outside_the_vor_window():
    roster = [DraftAsset("r", "Roster", "F", 90, nhl_team="EDM")]
    best = DraftAsset("a", "Best", "F", 100, nhl_team="COL")
    outside = DraftAsset("b", "Outside", "F", 96.9, nhl_team="EDM")

    chosen = TeamAwareVorpStrategy("stack", 3).choose(_context([best, outside], roster))

    assert chosen == best


def test_team_tiebreak_grid_keeps_raw_baseline_and_adds_two_scenarios():
    rows = []
    for category, nhl_id, team, projected, actual in [
        ("F", 1, "AAA", 100, 80),
        ("F", 2, "BBB", 98, 90),
        ("F", 3, "AAA", 90, 70),
        ("D", 4, "AAA", 70, 50),
        ("D", 5, "BBB", 68, 60),
        ("D", 6, "CCC", 60, 40),
    ]:
        rows.append(
            {
                "category": category,
                "NHLID": nhl_id,
                "FullName": f"Player {nhl_id}",
                "team_key": team,
                "actual_points": actual,
                "projection_median": projected,
                "src__PoolPro": projected,
                "src__ESPN": projected - 1,
                "src__CBS": projected + 1,
                "src__Hashtag": projected + 2,
                "src__ScottCullen": projected - 2,
            }
        )
    prepared = prepare_projection_experiment(
        pd.DataFrame(rows),
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )

    details, picks = run_team_tiebreak_drafts(
        prepared,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )
    paired = paired_scenario_deltas(details)
    method_count = 6  # five single sources plus trimmed_mean_minmax

    assert len(details) == method_count * len(SELECTION_SCENARIOS) * 2
    assert len(picks) == len(details) * 2
    assert set(details["selection_scenario"]) == set(SELECTION_SCENARIOS)
    assert len(paired) == method_count * 2 * 2


def test_team_field_grid_inverts_focal_and_opponent_stacking():
    rows = []
    for category, nhl_id, team, projected, actual in [
        ("F", 1, "AAA", 100, 80),
        ("F", 2, "BBB", 98, 90),
        ("F", 3, "AAA", 90, 70),
        ("D", 4, "AAA", 70, 50),
        ("D", 5, "BBB", 68, 60),
        ("D", 6, "CCC", 60, 40),
    ]:
        rows.append(
            {
                "category": category,
                "NHLID": nhl_id,
                "FullName": f"Player {nhl_id}",
                "team_key": team,
                "actual_points": actual,
                "projection_median": projected,
                "src__PoolPro": projected,
                "src__ESPN": projected - 1,
                "src__CBS": projected + 1,
                "src__Hashtag": projected + 2,
                "src__ScottCullen": projected - 2,
            }
        )
    prepared = prepare_projection_experiment(
        pd.DataFrame(rows),
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )

    details, picks = run_team_field_drafts(
        prepared,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )
    paired = paired_field_deltas(details)
    method_count = 6

    assert len(details) == method_count * len(FIELD_REGIMES) * 2
    assert len(picks) == len(details) * 2
    assert set(details["field_regime"]) == set(FIELD_REGIMES)
    assert len(paired) == method_count * 2 * 2
