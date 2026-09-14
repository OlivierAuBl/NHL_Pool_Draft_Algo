from __future__ import annotations

from dataclasses import dataclass

from nhl_draft_lab.models import DraftAsset, DraftContext
from nhl_draft_lab.pressure import (
    PressureConfig,
    TierPressureBook,
    build_tier_pressure_book,
    nearest_horizon_label,
)
from nhl_draft_lab.tiering import TierBook, TierConfig, TierState, build_tier_book


@dataclass(frozen=True)
class TierPressureChoice:
    category: str
    tier_no: int
    horizon: str | None
    tier_value: float
    pressure_score: float
    remaining_in_tier: int
    gap_to_next_tier: float | None
    asset: DraftAsset


class TierPressureStrategy:
    """Use one of three pre-computed printable scores for the active tier.

    The scores themselves are fixed before the draft.  During the draft the
    strategy only chooses SHORT / MID / LONG based on the actual number of picks
    until the focal GM's next turn.  Once a category/tier wins, the best real
    VORP player remaining inside that tier is selected.
    """

    name = "tier_pressure"

    def __init__(
        self,
        tier_config: TierConfig | None = None,
        pressure_config: PressureConfig | None = None,
    ) -> None:
        self.tier_config = tier_config or TierConfig()
        self.pressure_config = pressure_config or PressureConfig()
        self._tier_book: TierBook | None = None
        self._pressure_book: TierPressureBook | None = None

    def _books(self, context: DraftContext) -> tuple[TierBook, TierPressureBook]:
        if self._tier_book is None:
            universe = context.initial_assets or context.available_assets
            self._tier_book = build_tier_book(
                universe,
                context.replacement_levels,
                self.tier_config,
            )
            self._pressure_book = build_tier_pressure_book(
                self._tier_book,
                context.roster_config,
                self.pressure_config,
            )
        assert self._pressure_book is not None
        return self._tier_book, self._pressure_book

    @staticmethod
    def _vorp(asset: DraftAsset, context: DraftContext) -> float:
        return asset.value - context.replacement_levels.get(asset.category, 0.0)

    def _best_asset_in_state(self, state: TierState, context: DraftContext) -> DraftAsset:
        return max(
            state.remaining,
            key=lambda asset: (self._vorp(asset, context), asset.value, asset.name),
        )

    def choices(self, context: DraftContext) -> list[TierPressureChoice]:
        tier_book, pressure_book = self._books(context)
        needed_categories = [
            category
            for category in context.roster_config.counts
            if context.category_needed(category)
        ]
        states = tier_book.active_states(needed_categories, context.available_assets)
        horizon = nearest_horizon_label(context.picks_until_next, self.pressure_config)

        choices: list[TierPressureChoice] = []
        for category in needed_categories:
            state = states.get(category)
            if state is None:
                continue
            pressure = pressure_book.for_tier(state.tier)
            score = state.tier.tier_value if horizon is None else pressure.score(horizon)
            choices.append(
                TierPressureChoice(
                    category=category,
                    tier_no=state.tier.tier_no,
                    horizon=horizon,
                    tier_value=state.tier.tier_value,
                    pressure_score=score,
                    remaining_in_tier=state.remaining_count,
                    gap_to_next_tier=state.gap_to_next_tier,
                    asset=self._best_asset_in_state(state, context),
                )
            )
        return choices

    def choose(self, context: DraftContext) -> DraftAsset:
        choices = self.choices(context)
        if not choices:
            raise RuntimeError("No eligible asset remains for this roster")

        best = max(
            choices,
            key=lambda choice: (
                choice.pressure_score,
                choice.tier_value,
                self._vorp(choice.asset, context),
                choice.asset.value,
                choice.asset.name,
            ),
        )
        return best.asset
