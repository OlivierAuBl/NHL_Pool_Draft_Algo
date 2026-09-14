from __future__ import annotations

from dataclasses import dataclass

from nhl_draft_lab.models import DraftAsset, DraftContext
from nhl_draft_lab.strategies.tier_vorp import TierVorpStrategy
from nhl_draft_lab.tiering import TierConfig


@dataclass(frozen=True)
class TierLookaheadEvaluation:
    candidate: DraftAsset
    current_tier_value: float
    future_tier_value: float
    total_tier_value: float
    planned_sequence: tuple[str, ...]
    intervening_gaps: tuple[int, ...]


@dataclass(frozen=True)
class _Plan:
    total_tier_value: float
    assets: tuple[DraftAsset, ...]
    gaps: tuple[int, ...]


class TierLookaheadStrategy:
    """Look ahead using fixed tiers rather than individual VORP precision.

    The focal GM branches on the best currently available asset from each active
    position tier. Opponents are simulated as deterministic ``tier_vorp`` GMs.

    Because a tier keeps a fixed median value until exhausted, this simulation
    naturally values:
    - how many players remain in the current tier;
    - whether the tier survives until the focal GM's next snake pick;
    - the fixed drop to the next tier once it is exhausted.

    No explicit scarcity coefficient is used in V1: the tier depletion itself
    creates the pressure signal.
    """

    name = "tier_lookahead"

    def __init__(
        self,
        tier_config: TierConfig | None = None,
        focal_picks: int = 3,
    ) -> None:
        if focal_picks < 1:
            raise ValueError("focal_picks must be >= 1")
        self.tier_config = tier_config or TierConfig()
        self.focal_picks = focal_picks
        self._tier_vorp = TierVorpStrategy(self.tier_config)
        self._opponent = TierVorpStrategy(self.tier_config)

    @staticmethod
    def _picks_until_next_from_future_order(
        future_order: tuple[int, ...], gm_id: int
    ) -> int | None:
        for index, future_gm in enumerate(future_order):
            if future_gm == gm_id:
                return index
        return None

    def _tier_value(self, asset: DraftAsset, context: DraftContext) -> float:
        return self._tier_vorp.tier_value_for_asset(asset, context)

    def _candidate_assets(self, context: DraftContext) -> list[DraftAsset]:
        return [choice.asset for choice in self._tier_vorp.choices(context)]

    def _advance_to_next_focal_pick(
        self,
        context: DraftContext,
        candidate: DraftAsset,
    ) -> tuple[DraftContext | None, int]:
        if not context.future_order:
            return None, 0

        available = list(context.available_assets)
        available.remove(candidate)

        rosters = {gm: list(roster) for gm, roster in context.rosters_by_gm.items()}
        rosters.setdefault(context.gm_id, list(context.roster))
        rosters[context.gm_id].append(candidate)

        for index, future_gm in enumerate(context.future_order):
            if future_gm == context.gm_id:
                remaining_order = tuple(context.future_order[index + 1 :])
                next_context = DraftContext(
                    gm_id=context.gm_id,
                    draft_slot=context.draft_slot,
                    overall_pick=context.overall_pick + index + 1,
                    round_no=context.round_no,
                    picks_until_next=self._picks_until_next_from_future_order(
                        remaining_order, context.gm_id
                    ),
                    roster=tuple(rosters[context.gm_id]),
                    roster_config=context.roster_config,
                    available_assets=tuple(available),
                    replacement_levels=context.replacement_levels,
                    rosters_by_gm={gm: tuple(roster) for gm, roster in rosters.items()},
                    future_order=remaining_order,
                    initial_assets=context.initial_assets,
                )
                return next_context, index

            opponent_context = DraftContext(
                gm_id=future_gm,
                draft_slot=future_gm,
                overall_pick=context.overall_pick + index + 1,
                round_no=context.round_no,
                picks_until_next=None,
                roster=tuple(rosters.get(future_gm, ())),
                roster_config=context.roster_config,
                available_assets=tuple(available),
                replacement_levels=context.replacement_levels,
                rosters_by_gm={gm: tuple(roster) for gm, roster in rosters.items()},
                future_order=(),
                initial_assets=context.initial_assets,
            )
            chosen = self._opponent.choose(opponent_context)
            available.remove(chosen)
            rosters.setdefault(future_gm, []).append(chosen)

        return None, len(context.future_order)

    def _best_plan(self, context: DraftContext, picks_remaining: int) -> _Plan:
        if picks_remaining <= 0:
            return _Plan(0.0, (), ())

        candidates = self._candidate_assets(context)
        if not candidates:
            return _Plan(0.0, (), ())

        best: _Plan | None = None
        for candidate in candidates:
            current_tier_value = self._tier_value(candidate, context)
            next_context, gap = self._advance_to_next_focal_pick(context, candidate)

            future = _Plan(0.0, (), ())
            if picks_remaining > 1 and next_context is not None:
                future = self._best_plan(next_context, picks_remaining - 1)

            plan = _Plan(
                total_tier_value=current_tier_value + future.total_tier_value,
                assets=(candidate, *future.assets),
                gaps=((gap,) + future.gaps) if next_context is not None else future.gaps,
            )

            if best is None:
                best = plan
                continue

            plan_key = (
                plan.total_tier_value,
                candidate.value,
                candidate.name,
            )
            best_candidate = best.assets[0]
            best_key = (
                best.total_tier_value,
                best_candidate.value,
                best_candidate.name,
            )
            if plan_key > best_key:
                best = plan

        assert best is not None
        return best

    def evaluate(self, context: DraftContext) -> list[TierLookaheadEvaluation]:
        candidates = self._candidate_assets(context)
        if not candidates:
            raise RuntimeError("No eligible asset remains for this roster")

        evaluations: list[TierLookaheadEvaluation] = []
        for candidate in candidates:
            current_tier_value = self._tier_value(candidate, context)
            next_context, gap = self._advance_to_next_focal_pick(context, candidate)

            future = _Plan(0.0, (), ())
            if self.focal_picks > 1 and next_context is not None:
                future = self._best_plan(next_context, self.focal_picks - 1)

            gaps = ((gap,) + future.gaps) if next_context is not None else future.gaps
            evaluations.append(
                TierLookaheadEvaluation(
                    candidate=candidate,
                    current_tier_value=current_tier_value,
                    future_tier_value=future.total_tier_value,
                    total_tier_value=current_tier_value + future.total_tier_value,
                    planned_sequence=(candidate.name, *(asset.name for asset in future.assets)),
                    intervening_gaps=gaps,
                )
            )
        return evaluations

    def choose(self, context: DraftContext) -> DraftAsset:
        evaluations = self.evaluate(context)
        best = max(
            evaluations,
            key=lambda evaluation: (
                evaluation.total_tier_value,
                evaluation.candidate.value,
                evaluation.candidate.name,
            ),
        )
        return best.candidate
