from __future__ import annotations

from nhl_draft_lab.strategies.base import Strategy
from nhl_draft_lab.strategies.basic import BasicStrategy
from nhl_draft_lab.strategies.lookahead import LookaheadVorpStrategy
from nhl_draft_lab.strategies.lookahead2 import Lookahead2VorpStrategy
from nhl_draft_lab.strategies.plateau import PlateauStrategy
from nhl_draft_lab.strategies.tier_lookahead import TierLookaheadStrategy
from nhl_draft_lab.strategies.tier_pressure import TierPressureStrategy
from nhl_draft_lab.strategies.tier_vorp import TierVorpStrategy
from nhl_draft_lab.strategies.vorp import VorpStrategy
from nhl_draft_lab.pressure import PressureConfig
from nhl_draft_lab.tiering import TierConfig


STRATEGY_NAMES = (
    "basic",
    "vorp",
    "plateau",
    "lookahead",
    "lookahead2",
    "tier_vorp",
    "tier_lookahead",
    "tier_pressure",
)


def build_strategy(
    name: str,
    *,
    plateau_window: int = 5,
    plateau_weight: float = 1.0,
    lookahead_candidates_per_category: int = 5,
    lookahead2_focal_picks: int = 3,
    tier_relative_width: float = 0.05,
    tier_max_size: int = 5,
    tier_superstar_max_size: int = 3,
    tier_superstar_tiers: int = 2,
    tier_lookahead_focal_picks: int = 3,
    tier_pressure_short_horizon: int = 4,
    tier_pressure_mid_horizon: int = 14,
    tier_pressure_long_horizon: int = 24,
    tier_pressure_temperature: float = 10.0,
    tier_pressure_cap: float = 0.03,
) -> Strategy:
    """Create one Stage-1 strategy.

    Stage 1 is omniscient: every DraftAsset carries the player's/team's *real
    final season value*. Tier strategies deliberately discard some of that
    precision after fixed pre-draft tiers are built.
    """
    tier_config = TierConfig(
        relative_width=tier_relative_width,
        max_size=tier_max_size,
        superstar_max_size=tier_superstar_max_size,
        superstar_tiers=tier_superstar_tiers,
    )


    pressure_config = PressureConfig(
        short_horizon=tier_pressure_short_horizon,
        mid_horizon=tier_pressure_mid_horizon,
        long_horizon=tier_pressure_long_horizon,
        temperature=tier_pressure_temperature,
        cap=tier_pressure_cap,
    )

    if name == "basic":
        return BasicStrategy()
    if name == "vorp":
        return VorpStrategy()
    if name == "plateau":
        return PlateauStrategy(window=plateau_window, weight=plateau_weight)
    if name == "lookahead":
        return LookaheadVorpStrategy(
            candidates_per_category=lookahead_candidates_per_category
        )
    if name == "lookahead2":
        return Lookahead2VorpStrategy(focal_picks=lookahead2_focal_picks)
    if name == "tier_vorp":
        return TierVorpStrategy(tier_config=tier_config)
    if name == "tier_lookahead":
        return TierLookaheadStrategy(
            tier_config=tier_config,
            focal_picks=tier_lookahead_focal_picks,
        )
    if name == "tier_pressure":
        return TierPressureStrategy(
            tier_config=tier_config,
            pressure_config=pressure_config,
        )
    raise ValueError(f"Unknown strategy {name!r}; expected one of {STRATEGY_NAMES}")
