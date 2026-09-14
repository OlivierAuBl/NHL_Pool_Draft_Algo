from __future__ import annotations

from dataclasses import dataclass

from nhl_draft_lab.models import DraftAsset, DraftContext
from nhl_draft_lab.strategies.vorp import VorpStrategy


@dataclass(frozen=True)
class LookaheadEvaluation:
    candidate: DraftAsset
    current_vorp: float
    next_best_vorp: float
    total_two_pick_value: float
    intervening_picks: int


class LookaheadVorpStrategy:
    """One-pick omniscient lookahead on top of fixed VORP.

    Stage 1 only: every DraftAsset.value is already the realised end-of-season
    pool score.  For each candidate we:

    1. take the candidate now;
    2. simulate every opponent pick until our next real pick, assuming opponents
       use the fixed VORP strategy;
    3. add the VORP of our best asset at that next pick;
    4. choose the candidate maximising the two-pick value.

    This makes the snake horizon matter automatically.  A GM can have 22 picks
    before one selection and only 6 before the next.  At a snake turn, the next
    focal pick can be immediate (0 intervening picks), so the strategy naturally
    evaluates the two consecutive selections as a pair.
    """

    name = "lookahead"

    def __init__(self, candidates_per_category: int = 5) -> None:
        if candidates_per_category < 1:
            raise ValueError("candidates_per_category must be >= 1")
        self.candidates_per_category = candidates_per_category
        self._vorp = VorpStrategy()

    @staticmethod
    def _vorp_value(asset: DraftAsset, context: DraftContext) -> float:
        return asset.value - context.replacement_levels.get(asset.category, 0.0)

    def _candidate_pool(self, context: DraftContext) -> list[DraftAsset]:
        """Keep the top few VORP candidates in every still-needed category.

        Looking at a handful per category keeps the backtest fast while avoiding
        a global top-K that could accidentally exclude a scarce position.
        """
        eligible = context.eligible_assets()
        if not eligible:
            return []

        by_category: dict[str, list[DraftAsset]] = {}
        for asset in eligible:
            by_category.setdefault(asset.category, []).append(asset)

        candidates: list[DraftAsset] = []
        for assets in by_category.values():
            ranked = sorted(
                assets,
                key=lambda a: (
                    self._vorp_value(a, context),
                    a.value,
                    a.name,
                ),
                reverse=True,
            )
            candidates.extend(ranked[: self.candidates_per_category])
        return candidates

    def _simulate_until_next_focal_pick(
        self,
        context: DraftContext,
        candidate: DraftAsset,
    ) -> tuple[DraftContext | None, int]:
        if not context.rosters_by_gm or not context.future_order:
            return None, 0

        available = list(context.available_assets)
        available.remove(candidate)

        rosters = {
            gm: list(roster)
            for gm, roster in context.rosters_by_gm.items()
        }
        rosters.setdefault(context.gm_id, list(context.roster))
        rosters[context.gm_id].append(candidate)

        intervening = 0
        for future_gm in context.future_order:
            if future_gm == context.gm_id:
                next_context = DraftContext(
                    gm_id=context.gm_id,
                    draft_slot=context.draft_slot,
                    overall_pick=context.overall_pick + intervening + 1,
                    round_no=context.round_no,
                    picks_until_next=None,
                    roster=tuple(rosters[context.gm_id]),
                    roster_config=context.roster_config,
                    available_assets=tuple(available),
                    replacement_levels=context.replacement_levels,
                    rosters_by_gm={gm: tuple(r) for gm, r in rosters.items()},
                    future_order=(),
                )
                return next_context, intervening

            opponent_context = DraftContext(
                gm_id=future_gm,
                draft_slot=future_gm,
                overall_pick=context.overall_pick + intervening + 1,
                round_no=context.round_no,
                picks_until_next=None,
                roster=tuple(rosters.get(future_gm, ())),
                roster_config=context.roster_config,
                available_assets=tuple(available),
                replacement_levels=context.replacement_levels,
                rosters_by_gm={gm: tuple(r) for gm, r in rosters.items()},
                future_order=(),
            )
            chosen = self._vorp.choose(opponent_context)
            available.remove(chosen)
            rosters.setdefault(future_gm, []).append(chosen)
            intervening += 1

        return None, intervening

    def evaluate(self, context: DraftContext) -> list[LookaheadEvaluation]:
        candidates = self._candidate_pool(context)
        if not candidates:
            raise RuntimeError("No eligible asset remains for this roster")

        evaluations: list[LookaheadEvaluation] = []
        for candidate in candidates:
            current_vorp = self._vorp_value(candidate, context)
            next_context, intervening = self._simulate_until_next_focal_pick(
                context, candidate
            )

            next_best_vorp = 0.0
            if next_context is not None:
                next_eligible = next_context.eligible_assets()
                if next_eligible:
                    next_best = self._vorp.choose(next_context)
                    next_best_vorp = self._vorp_value(next_best, next_context)

            evaluations.append(
                LookaheadEvaluation(
                    candidate=candidate,
                    current_vorp=current_vorp,
                    next_best_vorp=next_best_vorp,
                    total_two_pick_value=current_vorp + next_best_vorp,
                    intervening_picks=intervening,
                )
            )

        return evaluations

    def choose(self, context: DraftContext) -> DraftAsset:
        evaluations = self.evaluate(context)
        best = max(
            evaluations,
            key=lambda e: (
                e.total_two_pick_value,
                e.current_vorp,
                e.candidate.value,
                e.candidate.name,
            ),
        )
        return best.candidate
