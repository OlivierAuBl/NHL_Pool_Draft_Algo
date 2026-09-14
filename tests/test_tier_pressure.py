from math import isclose

from nhl_draft_lab.models import DraftAsset, DraftContext, RosterConfig
from nhl_draft_lab.pressure import (
    PressureConfig,
    build_tier_pressure_book,
    nearest_horizon_label,
)
from nhl_draft_lab.strategies.tier_pressure import TierPressureStrategy
from nhl_draft_lab.tiering import TierConfig, build_tier_book


def _asset(name: str, category: str, value: float) -> DraftAsset:
    return DraftAsset(name.lower(), name, category, value)


def _standard_roster() -> RosterConfig:
    return RosterConfig({"F": 10, "D": 3, "G": 2, "T": 1})


def test_equal_frontier_values_fall_back_to_roster_slot_shares():
    assets = []
    for category, count in {"F": 10, "D": 3, "G": 2, "T": 1}.items():
        for i in range(count):
            assets.append(_asset(f"{category}{i}", category, 50.0))

    book = build_tier_book(
        assets,
        {"F": 0.0, "D": 0.0, "G": 0.0, "T": 0.0},
        TierConfig(relative_width=1.0, max_size=20, superstar_tiers=0),
    )
    pressure = build_tier_pressure_book(book, _standard_roster(), PressureConfig())

    f_score = pressure.for_tier(book.tiers_by_category["F"][0])
    d_score = pressure.for_tier(book.tiers_by_category["D"][0])
    assert isclose(f_score.estimated_pick_share, 10 / 16, rel_tol=1e-9)
    assert isclose(d_score.estimated_pick_share, 3 / 16, rel_tol=1e-9)


def test_pressure_bonus_is_capped_relative_to_tier_value():
    assets = (
        _asset("F1", "F", 70), _asset("F2", "F", 69), _asset("F3", "F", 68),
        _asset("F4", "F", 67), _asset("F5", "F", 66), _asset("F6", "F", 65),
        _asset("D1", "D", 80), _asset("D2", "D", 79),
        _asset("D3", "D", 60), _asset("D4", "D", 59),
        _asset("G1", "G", 64), _asset("G2", "G", 63),
        _asset("T1", "T", 62), _asset("T2", "T", 61),
    )
    book = build_tier_book(
        assets,
        {"F": 0.0, "D": 0.0, "G": 0.0, "T": 0.0},
        TierConfig(relative_width=0.04, max_size=5, superstar_tiers=0),
    )
    cap = 0.03
    pressure = build_tier_pressure_book(
        book,
        _standard_roster(),
        PressureConfig(short_horizon=4, mid_horizon=14, long_horizon=24, cap=cap),
    )

    d1 = pressure.for_tier(book.tiers_by_category["D"][0])
    max_bonus = cap * d1.tier_value
    assert d1.size == 2
    assert d1.gap_to_next_tier > 15
    assert d1.estimated_pick_share > 3 / 16
    assert d1.short_exhaustion_risk < d1.mid_exhaustion_risk < d1.long_exhaustion_risk
    assert 0 < d1.short_score - d1.tier_value <= max_bonus + 1e-9
    assert 0 < d1.mid_score - d1.tier_value <= max_bonus + 1e-9
    assert 0 < d1.long_score - d1.tier_value <= max_bonus + 1e-9
    # Once the cap is reached, longer horizons do not keep inflating the score.
    assert isclose(d1.short_score, d1.mid_score, rel_tol=1e-9)
    assert isclose(d1.mid_score, d1.long_score, rel_tol=1e-9)


def test_horizon_mapping_uses_only_three_printable_columns():
    config = PressureConfig(short_horizon=4, mid_horizon=14, long_horizon=24)
    assert nearest_horizon_label(0, config) == "short"
    assert nearest_horizon_label(6, config) == "short"
    assert nearest_horizon_label(14, config) == "mid"
    assert nearest_horizon_label(20, config) == "long"
    assert nearest_horizon_label(28, config) == "long"
    assert nearest_horizon_label(None, config) is None


def test_cap_prevents_pressure_from_overriding_a_clearly_better_tier():
    assets = (
        _asset("F1", "F", 70), _asset("F2", "F", 69), _asset("F3", "F", 68),
        _asset("F4", "F", 67), _asset("F5", "F", 66), _asset("F6", "F", 65),
        _asset("D1", "D", 64), _asset("D2", "D", 63),
        _asset("D3", "D", 40), _asset("D4", "D", 39),
    )
    roster = RosterConfig({"F": 1, "D": 1, "G": 0, "T": 0})
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=28,
        roster=(),
        roster_config=roster,
        available_assets=assets,
        replacement_levels={"F": 0.0, "D": 0.0, "G": 0.0, "T": 0.0},
        initial_assets=assets,
    )
    strategy = TierPressureStrategy(
        tier_config=TierConfig(relative_width=0.04, max_size=5, superstar_tiers=0),
        pressure_config=PressureConfig(cap=0.03),
    )

    # D has the bigger cliff, but its tier is clearly worse than F; a 3% cap
    # prevents pressure from manufacturing a huge leapfrog.
    assert strategy.choose(context).category == "F"


def test_pressure_can_break_a_close_tier_decision():
    assets = (
        _asset("F1", "F", 70.0), _asset("F2", "F", 69.5), _asset("F3", "F", 69.0),
        _asset("F4", "F", 68.5), _asset("F5", "F", 68.0), _asset("F6", "F", 67.5),
        _asset("D1", "D", 69.2), _asset("D2", "D", 68.8),
        _asset("D3", "D", 48.0), _asset("D4", "D", 47.0),
    )
    roster = RosterConfig({"F": 1, "D": 1, "G": 0, "T": 0})
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=24,
        roster=(),
        roster_config=roster,
        available_assets=assets,
        replacement_levels={"F": 0.0, "D": 0.0, "G": 0.0, "T": 0.0},
        initial_assets=assets,
    )
    strategy = TierPressureStrategy(
        tier_config=TierConfig(relative_width=0.04, max_size=5, superstar_tiers=0),
        pressure_config=PressureConfig(cap=0.03),
    )

    # Here the two tiers are close enough that a capped urgency bonus can
    # legitimately break the tie in favour of the scarce D tier.
    assert strategy.choose(context).category == "D"


def test_zero_cap_reduces_to_tier_vorp_choice():
    from nhl_draft_lab.strategies.tier_vorp import TierVorpStrategy

    assets = (
        _asset("F1", "F", 70), _asset("F2", "F", 69), _asset("F3", "F", 68),
        _asset("D1", "D", 69), _asset("D2", "D", 68),
        _asset("D3", "D", 48), _asset("D4", "D", 47),
    )
    roster = RosterConfig({"F": 1, "D": 1, "G": 0, "T": 0})
    context = DraftContext(
        gm_id=1,
        draft_slot=1,
        overall_pick=1,
        round_no=1,
        picks_until_next=24,
        roster=(),
        roster_config=roster,
        available_assets=assets,
        replacement_levels={"F": 0.0, "D": 0.0, "G": 0.0, "T": 0.0},
        initial_assets=assets,
    )
    tier_config = TierConfig(relative_width=0.04, max_size=5, superstar_tiers=0)
    baseline = TierVorpStrategy(tier_config=tier_config)
    pressure = TierPressureStrategy(
        tier_config=tier_config,
        pressure_config=PressureConfig(cap=0.0),
    )
    assert pressure.choose(context).entity_id == baseline.choose(context).entity_id
