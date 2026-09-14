from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.strategies.lookahead2 import Lookahead2VorpStrategy


def _assets():
    return (
        DraftAsset("f1", "F1", "F", 120),
        DraftAsset("f2", "F2", "F", 110),
        DraftAsset("f3", "F3", "F", 100),
        DraftAsset("f4", "F4", "F", 90),
        DraftAsset("d1", "D1", "D", 80),
        DraftAsset("d2", "D2", "D", 70),
        DraftAsset("d3", "D3", "D", 60),
        DraftAsset("d4", "D4", "D", 50),
        DraftAsset("g1", "G1", "G", 75),
        DraftAsset("g2", "G2", "G", 65),
        DraftAsset("g3", "G3", "G", 55),
        DraftAsset("g4", "G4", "G", 45),
    )


def test_lookahead2_sees_double_pick_then_longer_gap():
    context = DraftContext(
        gm_id=3,
        draft_slot=3,
        overall_pick=3,
        round_no=1,
        picks_until_next=0,
        roster=(),
        roster_config=RosterConfig({"F": 1, "D": 1, "G": 1}),
        available_assets=_assets(),
        replacement_levels={"F": 100.0, "D": 60.0, "G": 55.0},
        rosters_by_gm={1: (), 2: (), 3: ()},
        # focal pick immediately, then two opponents, then focal pick again
        future_order=(3, 1, 2, 3),
    )

    evaluations = Lookahead2VorpStrategy(focal_picks=3).evaluate(context)
    assert evaluations
    assert all(evaluation.intervening_gaps[:2] == (0, 2) for evaluation in evaluations)
    assert all(len(evaluation.planned_sequence) == 3 for evaluation in evaluations)


def test_lookahead2_only_branches_on_best_asset_per_category():
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=None,
        roster=(),
        roster_config=RosterConfig({"F": 1, "D": 1, "G": 1}),
        available_assets=_assets(),
        replacement_levels={"F": 100.0, "D": 60.0, "G": 55.0},
        rosters_by_gm={1: ()},
        future_order=(),
    )

    evaluations = Lookahead2VorpStrategy(focal_picks=3).evaluate(context)
    assert {evaluation.candidate.name for evaluation in evaluations} == {"F1", "D1", "G1"}
