from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.strategies.basic import BasicStrategy
from nhl_draft_lab.strategies.plateau import PlateauStrategy
from nhl_draft_lab.strategies.vorp import VorpStrategy


def ctx(assets, replacements):
    return DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=4,
        roster=(),
        roster_config=RosterConfig({"F": 1, "D": 1}),
        available_assets=tuple(assets),
        replacement_levels=replacements,
    )


def test_basic_takes_highest_points():
    assets = [DraftAsset("f1", "F1", "F", 100), DraftAsset("d1", "D1", "D", 80)]
    assert BasicStrategy().choose(ctx(assets, {"F": 70, "D": 50})).name == "F1"


def test_vorp_uses_replacement_level():
    assets = [DraftAsset("f1", "F1", "F", 100), DraftAsset("d1", "D1", "D", 80)]
    assert VorpStrategy().choose(ctx(assets, {"F": 90, "D": 50})).name == "D1"


def test_plateau_rewards_local_cliff():
    assets = [
        DraftAsset("f1", "F1", "F", 100),
        DraftAsset("f2", "F2", "F", 99),
        DraftAsset("f3", "F3", "F", 98),
        DraftAsset("d1", "D1", "D", 80),
        DraftAsset("d2", "D2", "D", 60),
        DraftAsset("d3", "D3", "D", 59),
    ]
    # Equal-ish VORP, but D has a much steeper local curve.
    choice = PlateauStrategy(window=2, weight=1.0).choose(ctx(assets, {"F": 70, "D": 50}))
    assert choice.name == "D1"
