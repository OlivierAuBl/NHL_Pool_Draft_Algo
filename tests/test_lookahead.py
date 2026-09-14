from nhl_draft_lab.draft.engine import _picks_until_next, snake_order
from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.strategies.lookahead import LookaheadVorpStrategy
from nhl_draft_lab.strategies.vorp import VorpStrategy


def test_snake_horizon_for_slot_4_with_15_gms_alternates_22_and_6():
    order = snake_order(15, 4)
    slot4_indexes = [i for i, gm in enumerate(order) if gm == 4]

    first_gap = _picks_until_next(order, slot4_indexes[0], 4)
    second_gap = _picks_until_next(order, slot4_indexes[1], 4)

    assert first_gap == 22
    assert second_gap == 6


def test_lookahead_can_prefer_scarce_position_that_opponents_will_remove():
    roster_config = RosterConfig({"F": 1, "D": 1})
    f1 = DraftAsset("f1", "F1", "F", 120)
    f2 = DraftAsset("f2", "F2", "F", 110)
    f3 = DraftAsset("f3", "F3", "F", 100)
    d1 = DraftAsset("d1", "D1", "D", 75)
    d2 = DraftAsset("d2", "D2", "D", 70)
    d3 = DraftAsset("d3", "D3", "D", 65)

    # Opponents already filled F, so before GM1's next pick they can only take D.
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=2,
        roster=(),
        roster_config=roster_config,
        available_assets=(f1, f2, f3, d1, d2, d3),
        replacement_levels={"F": 100.0, "D": 65.0},
        rosters_by_gm={
            1: (),
            2: (DraftAsset("of2", "Opponent F2", "F", 1),),
            3: (DraftAsset("of3", "Opponent F3", "F", 1),),
        },
        future_order=(2, 3, 1),
    )

    # Greedy VORP prefers F1: 20 VORP vs D1: 10 VORP.
    assert VorpStrategy().choose(context) == f1

    # Lookahead sees that D1/D2 disappear before GM1 picks again, while F1 survives.
    strategy = LookaheadVorpStrategy(candidates_per_category=3)
    assert strategy.choose(context) == d1


def test_lookahead_double_pick_has_zero_intervening_picks():
    roster_config = RosterConfig({"F": 1, "D": 1})
    f1 = DraftAsset("f1", "F1", "F", 120)
    d1 = DraftAsset("d1", "D1", "D", 75)
    f2 = DraftAsset("f2", "F2", "F", 100)
    d2 = DraftAsset("d2", "D2", "D", 65)

    context = DraftContext(
        gm_id=15,
        draft_slot=15,
        overall_pick=15,
        round_no=1,
        picks_until_next=0,
        roster=(),
        roster_config=roster_config,
        available_assets=(f1, d1, f2, d2),
        replacement_levels={"F": 100.0, "D": 65.0},
        rosters_by_gm={15: ()},
        future_order=(15,),
    )

    evaluations = LookaheadVorpStrategy(candidates_per_category=2).evaluate(context)
    assert evaluations
    assert all(e.intervening_picks == 0 for e in evaluations)
    assert max(e.total_two_pick_value for e in evaluations) == 30.0
