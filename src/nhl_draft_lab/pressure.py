from __future__ import annotations

from dataclasses import dataclass
from math import comb, exp
from typing import Mapping

from nhl_draft_lab.models import RosterConfig
from nhl_draft_lab.tiering import Tier, TierBook


@dataclass(frozen=True)
class PressureConfig:
    """Configuration for printable SHORT / MID / LONG tier scores.

    Horizons are expressed as the number of *other picks* before the focal GM
    drafts again.  For a 15-GM snake, 4 / 14 / 24 are useful representative
    horizons for a short turn, the middle of the snake, and a long turn.

    ``temperature`` controls how strongly a higher median VORP tier attracts
    picks relative to competing positions.  If competing tier values are equal,
    demand falls back to the roster slot proportions.  A 10-point advantage at
    temperature=10 multiplies the position's demand weight by e (~2.72).
    """

    short_horizon: int = 4
    mid_horizon: int = 14
    long_horizon: int = 24
    temperature: float = 10.0
    cap: float = 0.03

    def __post_init__(self) -> None:
        if self.short_horizon < 0:
            raise ValueError("short_horizon must be >= 0")
        if not self.short_horizon <= self.mid_horizon <= self.long_horizon:
            raise ValueError("Expected short_horizon <= mid_horizon <= long_horizon")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 <= self.cap <= 1.0:
            raise ValueError("cap must be between 0 and 1")


@dataclass(frozen=True)
class TierPressureScore:
    category: str
    tier_no: int
    tier_value: float
    size: int
    gap_to_next_tier: float
    market_pick: float
    estimated_pick_share: float
    short_exhaustion_risk: float
    mid_exhaustion_risk: float
    long_exhaustion_risk: float
    short_score: float
    mid_score: float
    long_score: float

    def score(self, horizon: str) -> float:
        label = horizon.lower()
        if label == "short":
            return self.short_score
        if label == "mid":
            return self.mid_score
        if label == "long":
            return self.long_score
        raise ValueError(f"Unknown horizon {horizon!r}; expected short, mid, or long")


class TierPressureBook:
    """Fixed printable scores attached to every pre-draft tier."""

    def __init__(self, scores: Mapping[tuple[str, int], TierPressureScore]) -> None:
        self._scores = dict(scores)

    def for_tier(self, tier: Tier) -> TierPressureScore:
        return self._scores[(tier.category, tier.tier_no)]

    def all_scores(self) -> list[TierPressureScore]:
        return list(self._scores.values())


def _binomial_tail(n: int, p: float, at_least: int) -> float:
    """P(X >= at_least) for X ~ Binomial(n, p), without scipy."""
    if at_least <= 0:
        return 1.0
    if n < at_least or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return sum(
        comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
        for k in range(at_least, n + 1)
    )


def _tier_at_ordinal(tiers: tuple[Tier, ...], ordinal: float) -> Tier:
    """Return the tier containing a (possibly fractional) category draft rank."""
    target = max(1.0, ordinal)
    cumulative = 0
    for tier in tiers:
        cumulative += tier.size
        if target <= cumulative:
            return tier
    return tiers[-1]


def _market_frontier(
    target: Tier,
    tier_book: TierBook,
    roster_config: RosterConfig,
) -> tuple[float, dict[str, Tier]]:
    """Estimate which tier from each category competes with ``target``.

    We locate the target tier around the draft depth at which its midpoint would
    normally be reached if category selections followed roster slot shares.
    Then we inspect the corresponding tier in every other category at the same
    draft depth.  This creates a fixed, pre-draft comparison set and therefore
    keeps the eventual SHORT/MID/LONG scores printable.
    """
    total_slots = roster_config.total_rounds()
    target_slots = roster_config.required(target.category)
    if total_slots <= 0 or target_slots <= 0:
        raise ValueError("Roster configuration must contain positive slots")

    target_tiers = tier_book.tiers_by_category[target.category]
    before = 0
    for tier in target_tiers:
        if tier.tier_no == target.tier_no:
            break
        before += tier.size

    midpoint_category_rank = before + (target.size + 1) / 2.0
    target_share = target_slots / total_slots
    market_pick = midpoint_category_rank / target_share

    frontier: dict[str, Tier] = {}
    for category, tiers in tier_book.tiers_by_category.items():
        slots = roster_config.required(category)
        if slots <= 0 or not tiers:
            continue
        expected_category_rank = market_pick * (slots / total_slots)
        frontier[category] = _tier_at_ordinal(tiers, expected_category_rank)

    return market_pick, frontier


def _category_pick_shares(
    frontier: Mapping[str, Tier],
    roster_config: RosterConfig,
    temperature: float,
) -> dict[str, float]:
    """Softmax-like demand shares around a fixed market frontier.

    Roster slot proportions are the neutral prior.  Tier median VORP tilts that
    prior: a position whose current tier is much better than competing tiers is
    expected to absorb a larger fraction of the next picks.
    """
    total_slots = roster_config.total_rounds()
    if not frontier or total_slots <= 0:
        return {}

    max_value = max(tier.tier_value for tier in frontier.values())
    weights: dict[str, float] = {}
    for category, tier in frontier.items():
        slot_share = roster_config.required(category) / total_slots
        weights[category] = slot_share * exp((tier.tier_value - max_value) / temperature)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return {category: 0.0 for category in weights}
    return {category: weight / total_weight for category, weight in weights.items()}


def build_tier_pressure_book(
    tier_book: TierBook,
    roster_config: RosterConfig,
    config: PressureConfig | None = None,
) -> TierPressureBook:
    """Pre-compute three fixed scores for every tier.

    raw_pressure = gap_to_next_tier * P(tier exhausted before return)
    pressure_bonus = min(raw_pressure, cap * median tier VORP)
    score = median tier VORP + pressure_bonus

    The exhaustion probability uses only fixed pre-draft information:
    - tier size;
    - SHORT / MID / LONG pick horizon;
    - roster slot proportions;
    - the tier's median VORP relative to competing tiers at a similar draft depth.

    No opponent roster state is used, so the output can be printed before the
    draft and used as a simple reference sheet.
    """
    config = config or PressureConfig()
    scores: dict[tuple[str, int], TierPressureScore] = {}

    for category, tiers in tier_book.tiers_by_category.items():
        for index, tier in enumerate(tiers):
            next_tier = tiers[index + 1] if index + 1 < len(tiers) else None
            gap = max(0.0, tier.tier_value - next_tier.tier_value) if next_tier else 0.0

            if roster_config.required(category) <= 0:
                market_pick = 0.0
                p = 0.0
            else:
                market_pick, frontier = _market_frontier(tier, tier_book, roster_config)
                shares = _category_pick_shares(frontier, roster_config, config.temperature)
                p = shares.get(category, 0.0)

            short_risk = _binomial_tail(config.short_horizon, p, tier.size)
            mid_risk = _binomial_tail(config.mid_horizon, p, tier.size)
            long_risk = _binomial_tail(config.long_horizon, p, tier.size)

            scores[(category, tier.tier_no)] = TierPressureScore(
                category=category,
                tier_no=tier.tier_no,
                tier_value=tier.tier_value,
                size=tier.size,
                gap_to_next_tier=gap,
                market_pick=market_pick,
                estimated_pick_share=p,
                short_exhaustion_risk=short_risk,
                mid_exhaustion_risk=mid_risk,
                long_exhaustion_risk=long_risk,
                short_score=tier.tier_value + min(gap * short_risk, config.cap * max(tier.tier_value, 0.0)),
                mid_score=tier.tier_value + min(gap * mid_risk, config.cap * max(tier.tier_value, 0.0)),
                long_score=tier.tier_value + min(gap * long_risk, config.cap * max(tier.tier_value, 0.0)),
            )

    return TierPressureBook(scores)


def nearest_horizon_label(picks_until_next: int | None, config: PressureConfig) -> str | None:
    """Map a real snake gap to the closest printable horizon column."""
    if picks_until_next is None:
        return None
    candidates = {
        "short": config.short_horizon,
        "mid": config.mid_horizon,
        "long": config.long_horizon,
    }
    return min(candidates, key=lambda label: (abs(candidates[label] - picks_until_next), candidates[label]))
