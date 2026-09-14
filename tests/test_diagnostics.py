import pandas as pd

from nhl_draft_lab.backtest import summarize_backtests_by_slot
from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.strategies.basic import BasicStrategy
from nhl_draft_lab.strategies.diagnostic import DiagnosticStrategy
from nhl_draft_lab.strategies.vorp import VorpStrategy


def test_diagnostic_strategy_records_divergence_from_vorp():
    f1 = DraftAsset("f1", "F1", "F", 100)
    d1 = DraftAsset("d1", "D1", "D", 80)
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=4,
        roster=(),
        roster_config=RosterConfig({"F": 1, "D": 1}),
        available_assets=(f1, d1),
        replacement_levels={"F": 95.0, "D": 60.0},
    )

    strategy = DiagnosticStrategy(BasicStrategy(), VorpStrategy())
    chosen = strategy.choose(context)

    assert chosen == f1
    assert len(strategy.records) == 1
    record = strategy.records[0]
    assert record.diverged is True
    assert record.baseline_name == "D1"
    assert record.direct_points_delta == 20.0
    assert record.direct_vorp_delta == -15.0


def test_by_slot_summary_computes_weighted_divergence_rate():
    frame = pd.DataFrame([
        {
            "season_id": 1,
            "gm_count": 2,
            "draft_slot": 1,
            "focal_strategy": "lookahead2",
            "opponent_strategy": "vorp",
            "total_points": 100.0,
            "rank": 1,
            "winner_points": 100.0,
            "gap_to_winner": 0.0,
            "margin_vs_field_mean": 5.0,
            "focal_picks": 4,
            "divergence_count": 1,
            "direct_points_delta_sum": -2.0,
            "direct_vorp_delta_sum": -3.0,
        },
        {
            "season_id": 2,
            "gm_count": 2,
            "draft_slot": 1,
            "focal_strategy": "lookahead2",
            "opponent_strategy": "vorp",
            "total_points": 90.0,
            "rank": 2,
            "winner_points": 95.0,
            "gap_to_winner": 5.0,
            "margin_vs_field_mean": -5.0,
            "focal_picks": 4,
            "divergence_count": 3,
            "direct_points_delta_sum": -6.0,
            "direct_vorp_delta_sum": -9.0,
        },
    ])

    summary = summarize_backtests_by_slot(frame)
    row = summary.iloc[0]
    assert row["divergence_rate"] == 0.5
    assert row["mean_direct_points_delta_when_diverging"] == -2.0
    assert row["mean_direct_vorp_delta_when_diverging"] == -3.0
