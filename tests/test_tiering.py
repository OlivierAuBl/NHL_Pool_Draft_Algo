from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.strategies.tier_lookahead import TierLookaheadStrategy
from nhl_draft_lab.strategies.tier_vorp import TierVorpStrategy
from nhl_draft_lab.tiering import TierConfig, build_tier_book


def _asset(name: str, category: str, value: float) -> DraftAsset:
    return DraftAsset(name.lower(), name, category, value)


def test_tiers_use_anchored_relative_width_not_chaining():
    assets = [
        _asset("F1", "F", 100),
        _asset("F2", "F", 97),
        _asset("F3", "F", 94),
    ]
    book = build_tier_book(
        assets,
        {"F": 0.0},
        TierConfig(relative_width=0.04, max_size=10, superstar_tiers=0),
    )

    tiers = book.tiers_by_category["F"]
    assert tiers[0].asset_names == ("F1", "F2")
    assert tiers[1].asset_names == ("F3",)
    assert tiers[0].tier_value == 98.5


def test_superstar_tiers_can_have_smaller_cap():
    assets = [_asset(f"F{i}", "F", 101 - i) for i in range(1, 8)]
    book = build_tier_book(
        assets,
        {"F": 0.0},
        TierConfig(
            relative_width=1.0,
            max_size=5,
            superstar_max_size=2,
            superstar_tiers=1,
        ),
    )

    tiers = book.tiers_by_category["F"]
    assert tiers[0].size == 2
    assert tiers[1].size == 5


def test_tier_vorp_masks_individual_vorp_between_categories():
    # F's active tier median is 95, while D's active tier median is 96.
    # The best F itself (100) is better than the best D (97), but TierVORP
    # deliberately chooses the D tier because individual precision is masked.
    assets = (
        _asset("F1", "F", 100),
        _asset("F2", "F", 90),
        _asset("D1", "D", 97),
        _asset("D2", "D", 95),
    )
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=2,
        roster=(),
        roster_config=RosterConfig({"F": 1, "D": 1}),
        available_assets=assets,
        replacement_levels={"F": 0.0, "D": 0.0},
        initial_assets=assets,
    )
    strategy = TierVorpStrategy(
        TierConfig(relative_width=0.11, max_size=5, superstar_tiers=0)
    )

    assert strategy.choose(context).name == "D1"


def test_tier_lookahead_values_tier_survival():
    # GM1 picks, GM2 has two consecutive picks, then GM1 picks again.
    # F top tier has 3 members and survives; D top tier is a singleton and will
    # disappear if GM1 waits. TierVORP greedily prefers F (99 > 95), while the
    # two-pick tier lookahead secures D now and gets F later.
    assets = (
        _asset("F1", "F", 100),
        _asset("F2", "F", 99),
        _asset("F3", "F", 98),
        _asset("D1", "D", 95),
        _asset("D2", "D", 70),
        _asset("D3", "D", 69),
    )
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=2,
        roster=(),
        roster_config=RosterConfig({"F": 1, "D": 1}),
        available_assets=assets,
        replacement_levels={"F": 0.0, "D": 0.0},
        rosters_by_gm={1: (), 2: ()},
        future_order=(2, 2, 1),
        initial_assets=assets,
    )
    config = TierConfig(
        relative_width=0.05,
        max_size=5,
        superstar_max_size=3,
        superstar_tiers=2,
    )

    assert TierVorpStrategy(config).choose(context).name == "F1"
    assert TierLookaheadStrategy(config, focal_picks=2).choose(context).name == "D1"
