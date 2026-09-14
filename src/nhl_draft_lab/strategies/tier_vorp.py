from __future__ import annotations

from dataclasses import dataclass

from nhl_draft_lab.models import DraftAsset, DraftContext
from nhl_draft_lab.tiering import TierBook, TierConfig, TierState, build_tier_book


@dataclass(frozen=True)
class TierChoice:
    category: str
    tier_no: int
    tier_value: float
    remaining_in_tier: int
    gap_to_next_tier: float | None
    asset: DraftAsset


class TierVorpStrategy:
    """VORP with fixed pre-draft tiers.

    Individual VORPs are *masked* for the decision between categories. Each
    category is represented only by the fixed median VORP of its best active
    tier. After a tier/category is selected, the best real VORP remaining inside
    that tier is drafted.
    """

    name = "tier_vorp"

    def __init__(self, tier_config: TierConfig | None = None) -> None:
        self.tier_config = tier_config or TierConfig()
        self._tier_book: TierBook | None = None

    def _book(self, context: DraftContext) -> TierBook:
        if self._tier_book is None:
            universe = context.initial_assets or context.available_assets
            self._tier_book = build_tier_book(
                universe,
                context.replacement_levels,
                self.tier_config,
            )
        return self._tier_book

    @staticmethod
    def _vorp(asset: DraftAsset, context: DraftContext) -> float:
        return asset.value - context.replacement_levels.get(asset.category, 0.0)

    def _best_asset_in_state(self, state: TierState, context: DraftContext) -> DraftAsset:
        return max(
            state.remaining,
            key=lambda asset: (self._vorp(asset, context), asset.value, asset.name),
        )

    def choices(self, context: DraftContext) -> list[TierChoice]:
        book = self._book(context)
        choices: list[TierChoice] = []
        needed_categories = [
            category
            for category in context.roster_config.counts
            if context.category_needed(category)
        ]
        states = book.active_states(needed_categories, context.available_assets)
        for category in needed_categories:
            state = states.get(category)
            if state is None:
                continue
            choices.append(
                TierChoice(
                    category=category,
                    tier_no=state.tier.tier_no,
                    tier_value=state.tier.tier_value,
                    remaining_in_tier=state.remaining_count,
                    gap_to_next_tier=state.gap_to_next_tier,
                    asset=self._best_asset_in_state(state, context),
                )
            )
        return choices

    def tier_value_for_asset(self, asset: DraftAsset, context: DraftContext) -> float:
        return self._book(context).tier_for_asset(asset).tier_value

    def choose(self, context: DraftContext) -> DraftAsset:
        choices = self.choices(context)
        if not choices:
            raise RuntimeError("No eligible asset remains for this roster")

        best = max(
            choices,
            key=lambda choice: (
                choice.tier_value,
                self._vorp(choice.asset, context),
                choice.asset.value,
                choice.asset.name,
            ),
        )
        return best.asset
