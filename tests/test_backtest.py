from nhl_draft_lab.backtest import run_one_focal_backtest, summarize_backtests
from nhl_draft_lab.models import DraftAsset, RosterConfig


def test_focal_backtest_returns_rank_and_margin():
    assets = [
        *[DraftAsset(f"f{i}", f"F{i}", "F", 120 - i) for i in range(1, 13)],
        *[DraftAsset(f"d{i}", f"D{i}", "D", 70 - i) for i in range(1, 7)],
    ]
    result = run_one_focal_backtest(
        assets,
        season_id=20242025,
        gm_count=3,
        roster_config=RosterConfig({"F": 2, "D": 1}),
        draft_slot=2,
        focal_strategy="vorp",
        opponent_strategy="basic",
    )
    assert 1 <= result.rank <= 3
    assert result.gap_to_winner >= 0


def test_summary_groups_strategy_matchups():
    import pandas as pd

    frame = pd.DataFrame([
        {
            "season_id": 1,
            "gm_count": 2,
            "draft_slot": 1,
            "focal_strategy": "vorp",
            "opponent_strategy": "basic",
            "total_points": 100.0,
            "rank": 1,
            "winner_points": 100.0,
            "gap_to_winner": 0.0,
            "margin_vs_field_mean": 10.0,
        },
        {
            "season_id": 2,
            "gm_count": 2,
            "draft_slot": 1,
            "focal_strategy": "vorp",
            "opponent_strategy": "basic",
            "total_points": 90.0,
            "rank": 2,
            "winner_points": 95.0,
            "gap_to_winner": 5.0,
            "margin_vs_field_mean": -5.0,
        },
    ])
    summary = summarize_backtests(frame)
    row = summary.iloc[0]
    assert row["win_rate"] == 0.5
    assert row["mean_rank"] == 1.5
