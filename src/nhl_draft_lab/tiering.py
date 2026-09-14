from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from nhl_draft_lab.models import DraftAsset


@dataclass(frozen=True)
class TierConfig:
    """Configuration for fixed, pre-draft VORP tiers.

    ``relative_width`` is anchored on the first (best) player of each tier and
    expressed relative to that player's raw pool value. This avoids unstable
    ratios when VORP approaches zero.

    The first ``superstar_tiers`` tiers use ``superstar_max_size``; later tiers
    use ``max_size``. Set ``superstar_tiers=0`` to disable the special cap.
    """

    relative_width: float = 0.05
    max_size: int = 5
    superstar_max_size: int = 3
    superstar_tiers: int = 2

    def __post_init__(self) -> None:
        if self.relative_width < 0:
            raise ValueError("relative_width must be >= 0")
        if self.max_size < 1:
            raise ValueError("max_size must be >= 1")
        if self.superstar_max_size < 1:
            raise ValueError("superstar_max_size must be >= 1")
        if self.superstar_tiers < 0:
            raise ValueError("superstar_tiers must be >= 0")


@dataclass(frozen=True)
class Tier:
    category: str
    tier_no: int
    asset_ids: tuple[str, ...]
    asset_names: tuple[str, ...]
    tier_value: float
    leader_vorp: float
    min_vorp: float
    median_vorp: float

    @property
    def size(self) -> int:
        return len(self.asset_ids)


@dataclass(frozen=True)
class TierState:
    tier: Tier
    remaining: tuple[DraftAsset, ...]
    next_tier: Tier | None

    @property
    def remaining_count(self) -> int:
        return len(self.remaining)

    @property
    def gap_to_next_tier(self) -> float | None:
        if self.next_tier is None:
            return None
        return self.tier.tier_value - self.next_tier.tier_value


class TierBook:
    """Fixed tier definitions built once from the initial draft universe."""

    def __init__(self, tiers_by_category: Mapping[str, Sequence[Tier]]) -> None:
        self.tiers_by_category = {
            category: tuple(tiers) for category, tiers in tiers_by_category.items()
        }
        self._tier_by_asset_id: dict[str, Tier] = {}
        for tiers in self.tiers_by_category.values():
            for tier in tiers:
                for asset_id in tier.asset_ids:
                    self._tier_by_asset_id[asset_id] = tier

    def tier_for_asset(self, asset: DraftAsset) -> Tier:
        return self._tier_by_asset_id[asset.entity_id]

    def _active_state_from_map(
        self,
        category: str,
        available_by_id: Mapping[str, DraftAsset],
    ) -> TierState | None:
        tiers = self.tiers_by_category.get(category, ())
        for index, tier in enumerate(tiers):
            remaining = tuple(
                available_by_id[asset_id]
                for asset_id in tier.asset_ids
                if asset_id in available_by_id
            )
            if not remaining:
                continue

            next_tier = None
            for candidate in tiers[index + 1 :]:
                if any(asset_id in available_by_id for asset_id in candidate.asset_ids):
                    next_tier = candidate
                    break
            return TierState(tier=tier, remaining=remaining, next_tier=next_tier)
        return None

    def active_state(
        self,
        category: str,
        available_assets: Sequence[DraftAsset],
    ) -> TierState | None:
        available_by_id = {asset.entity_id: asset for asset in available_assets}
        return self._active_state_from_map(category, available_by_id)

    def active_states(
        self,
        categories: Sequence[str],
        available_assets: Sequence[DraftAsset],
    ) -> dict[str, TierState]:
        """Resolve several categories with one availability index build."""
        available_by_id = {asset.entity_id: asset for asset in available_assets}
        states: dict[str, TierState] = {}
        for category in categories:
            state = self._active_state_from_map(category, available_by_id)
            if state is not None:
                states[category] = state
        return states


def _vorp(asset: DraftAsset, replacement_levels: Mapping[str, float]) -> float:
    return asset.value - replacement_levels.get(asset.category, 0.0)


def build_tier_book(
    assets: Sequence[DraftAsset],
    replacement_levels: Mapping[str, float],
    config: TierConfig,
) -> TierBook:
    """Build anchored, fixed tiers independently for each category.

    Rules:
    - sort by true Stage-1 VORP descending;
    - start a tier at the best unassigned asset;
    - subsequent assets may join only while their VORP gap from the tier leader,
      divided by the leader's raw pool value, is <= ``relative_width``;
    - the tier is also capped by ``max_size`` (or the superstar cap);
    - tier value is the *median VORP* of its original members and never changes
      during the draft.
    """

    categories = sorted({asset.category for asset in assets})
    tiers_by_category: dict[str, list[Tier]] = {}

    for category in categories:
        ranked = sorted(
            (asset for asset in assets if asset.category == category),
            key=lambda asset: (
                _vorp(asset, replacement_levels),
                asset.value,
                asset.name,
            ),
            reverse=True,
        )

        tiers: list[Tier] = []
        cursor = 0
        while cursor < len(ranked):
            tier_no = len(tiers) + 1
            leader = ranked[cursor]
            leader_vorp = _vorp(leader, replacement_levels)
            denominator = max(abs(leader.value), 1e-9)
            cap = (
                config.superstar_max_size
                if tier_no <= config.superstar_tiers
                else config.max_size
            )

            members = [leader]
            cursor += 1
            while cursor < len(ranked) and len(members) < cap:
                candidate = ranked[cursor]
                candidate_vorp = _vorp(candidate, replacement_levels)
                relative_gap = (leader_vorp - candidate_vorp) / denominator
                if relative_gap > config.relative_width:
                    break
                members.append(candidate)
                cursor += 1

            vorps = [_vorp(asset, replacement_levels) for asset in members]
            med = float(median(vorps))
            tiers.append(
                Tier(
                    category=category,
                    tier_no=tier_no,
                    asset_ids=tuple(asset.entity_id for asset in members),
                    asset_names=tuple(asset.name for asset in members),
                    tier_value=med,
                    leader_vorp=max(vorps),
                    min_vorp=min(vorps),
                    median_vorp=med,
                )
            )

        tiers_by_category[category] = tiers

    return TierBook(tiers_by_category)
