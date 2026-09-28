from __future__ import annotations

import pandas as pd
import pytest

from nhl_draft_lab.models import RosterConfig
from nhl_draft_lab.projection_source_experiment import (
    discover_source_columns,
    prepare_projection_experiment,
    run_projection_source_drafts,
    summarize_methods,
)


def _master() -> pd.DataFrame:
    rows = []
    specifications = [
        ("F", 1, "F1", 100, 10),
        ("F", 2, "F2", 90, 40),
        ("F", 3, "F3", 80, 30),
        ("D", 4, "D1", 70, 20),
        ("D", 5, "D2", 60, 50),
        ("D", 6, "D3", 50, 15),
    ]
    for category, nhl_id, name, pool_pro, actual in specifications:
        rows.append(
            {
                "category": category,
                "NHLID": nhl_id,
                "FullName": name,
                "team_key": "AAA",
                "actual_points": actual,
                "projection_median": pool_pro - 1,
                "src__PoolPro": pool_pro,
                "src__ESPN": pool_pro - 10,
                "src__ScottCullen": pool_pro - 5,
                "src__CBS": pool_pro + 5,
                "src__Hastag": pool_pro + 10,
            }
        )
    return pd.DataFrame(rows)


def test_source_discovery_pairs_hastag_skater_and_hashtag_goalie_columns():
    master = _master()
    master["src__calc_Hashtag_Score"] = pd.NA
    goalie = master.iloc[[0]].copy()
    goalie["category"] = "G"
    goalie["src__Hastag"] = pd.NA
    goalie["src__calc_Hashtag_Score"] = 77
    combined = pd.concat([master, goalie], ignore_index=True)

    discovered = discover_source_columns(combined)

    assert discovered["Hashtag"]["F"] == "src__Hastag"
    assert discovered["Hashtag"]["G"] == "src__calc_Hashtag_Score"


def test_trimmed_consensus_removes_min_and_max_with_five_sources():
    master = _master()
    master.loc[0, [
        "src__PoolPro",
        "src__ESPN",
        "src__ScottCullen",
        "src__CBS",
        "src__Hastag",
    ]] = [1, 2, 3, 4, 100]
    prepared = prepare_projection_experiment(
        master,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
        trim_min_sources=5,
    )
    row = prepared.universe.iloc[0]

    assert row["method__trimmed_mean_minmax"] == pytest.approx(3)
    assert row["method__trimmed_median_minmax"] == pytest.approx(3)
    assert not bool(row["trim_fallback__trimmed_mean_minmax"])


def test_trimmed_consensus_uses_untrimmed_rule_when_sources_are_too_few():
    master = _master().drop(columns="src__Hastag")
    master.loc[0, ["src__PoolPro", "src__ESPN", "src__ScottCullen", "src__CBS"]] = [
        1,
        2,
        3,
        100,
    ]
    prepared = prepare_projection_experiment(
        master,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
        trim_min_sources=5,
    )
    row = prepared.universe.iloc[0]

    assert row["method__trimmed_mean_minmax"] == pytest.approx(26.5)
    assert row["method__trimmed_median_minmax"] == pytest.approx(2.5)
    assert bool(row["trim_fallback__trimmed_mean_minmax"])


def test_incomplete_source_category_is_replaced_by_pool_pro_and_reported():
    master = _master()
    master.loc[master["category"].eq("D"), "src__ESPN"] = pd.NA
    master.loc[master["category"].eq("D").idxmax(), "src__ESPN"] = 999
    prepared = prepare_projection_experiment(
        master,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )

    defense = prepared.universe[prepared.universe["category"].eq("D")]
    assert defense["method__espn"].tolist() == defense["opponent_projection"].tolist()
    excluded = prepared.exclusions.query("projection_method == 'ESPN' and category == 'D'")
    assert len(excluded) == 1
    coverage = prepared.coverage.query(
        "projection_method == 'ESPN' and category == 'D'"
    ).iloc[0]
    assert coverage["fallback_count"] == 3
    assert not bool(coverage["category_eligible"])


def test_focal_drafts_keep_vorp_fixed_and_score_with_actual_points():
    prepared = prepare_projection_experiment(
        _master(),
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )
    details, picks = run_projection_source_drafts(
        prepared,
        gm_count=2,
        roster_config=RosterConfig({"F": 1, "D": 1}),
    )
    summary = summarize_methods(details)

    assert len(details) == len(prepared.methods) * 2
    assert len(picks) == len(prepared.methods) * 2 * 2
    assert set(details["strategy"]) == {"vorp"}
    assert set(details["opponent_projection_method"]) == {"PoolPro"}
    assert details["final_points"].between(0, 100).all()
    assert set(summary["simulations"]) == {2}
