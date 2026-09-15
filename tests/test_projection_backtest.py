from __future__ import annotations

import pandas as pd
import pytest

from nhl_draft_lab.models import RosterConfig
from nhl_draft_lab.projection_backtest import (
    build_projection_draft_universe,
    draft_category_totals,
    paired_category_deltas,
    paired_candidate_deltas,
    run_projection_draft_comparison,
    summarize_category_deltas,
    summarize_paired_deltas,
    summarize_projection_drafts,
)


def projection_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = [
        ("F", 1, "Risky F", 100, 70, 10),
        ("F", 2, "Safe F", 90, 100, 100),
        ("F", 3, "Breakout F", 80, 90, 90),
        ("D", 4, "D1", 60, 60, 60),
        ("D", 5, "D2", 50, 50, 50),
    ]
    master = pd.DataFrame([
        {
            "category": category,
            "NHLID": nhl_id,
            "team_key": "AAA",
            "FullName": name,
            "projection_median": v0,
            "actual_points": actual,
        }
        for category, nhl_id, name, v0, _, actual in specifications
    ])
    candidate = pd.DataFrame([
        {
            "entity_id": f"P:{nhl_id}",
            "category": category,
            "name": name,
            "projected_points": projected,
        }
        for category, nhl_id, name, _, projected, _ in specifications
    ])
    return master, candidate


def test_projection_drafts_are_scored_with_actual_points_and_paired_by_slot():
    master, candidate = projection_frames()
    universe, audit = build_projection_draft_universe(master, candidate)

    details, picks = run_projection_draft_comparison(
        universe,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
        strategy_names=("basic",),
    )
    summary = summarize_projection_drafts(details)
    paired = paired_candidate_deltas(details)
    paired_summary = summarize_paired_deltas(paired).iloc[0]
    categories = draft_category_totals(picks)
    category_paired = paired_category_deltas(categories)
    category_summary = summarize_category_deltas(category_paired)

    assert len(universe) == 5
    assert set(audit["match_status"]) == {"MATCHED"}
    assert len(details) == 4
    assert len(picks) == 8
    assert set(summary["projection_model"]) == {"V0", "V1_WEIGHTED_GP_CANDIDATE"}
    assert paired["candidate_actual_points"].sum() > paired["v0_actual_points"].sum()
    assert paired_summary["mean_actual_points_delta"] == pytest.approx(40)
    assert paired_summary["slots_improved"] == 1
    assert paired_summary["slots_worsened"] == 1
    assert len(categories) == 8
    forward_delta = category_summary.loc[
        category_summary["category"].eq("F"), "total_actual_points_delta"
    ].item()
    assert forward_delta == pytest.approx(80)


def test_projection_draft_universe_audits_missing_candidate_assets():
    master, candidate = projection_frames()
    universe, audit = build_projection_draft_universe(master, candidate.iloc[:-1])

    assert len(universe) == 4
    missing = audit.loc[audit["match_status"].eq("MISSING_CANDIDATE")]
    assert missing["name"].tolist() == ["D2"]


def test_projection_draft_rejects_an_insufficient_category_pool():
    master, candidate = projection_frames()
    universe, _ = build_projection_draft_universe(master, candidate)

    with pytest.raises(ValueError, match="need 4"):
        run_projection_draft_comparison(
            universe,
            gm_count=2,
            roster_config=RosterConfig({"D": 2}),
            strategy_names=("basic",),
        )
