from __future__ import annotations

from dataclasses import dataclass

from nhl_draft_lab.models import DraftAsset, DraftContext
from nhl_draft_lab.strategies.vorp import VorpStrategy


@dataclass(frozen=True)
class Lookahead2Evaluation:
    candidate: DraftAsset
    current_vorp: float
    future_vorp: float
    total_three_pick_value: float
    planned_sequence: tuple[str, ...]
    intervening_gaps: tuple[int, ...]


@dataclass(frozen=True)
class _Plan:
    total_vorp: float
    assets: tuple[DraftAsset, ...]
    gaps: tuple[int, ...]


class Lookahead2VorpStrategy:
    """Perfect-information lookahead over the focal GM's next three picks.

    Compared with ``lookahead`` (current pick + one future pick), this version
    optimises:

        current pick + next focal pick + following focal pick

    Opponents are deterministic and use fixed VORP during the simulated picks.
    That makes the geometry of the snake emerge naturally.  For example, with
    15 GMs, slot 4 can see a 22-pick gap followed by a 6-pick gap, while slot 15
    can see 0 then 28.

    To keep the tree small, the focal GM branches only on the *best available
    asset in each still-needed category*.  Under Stage-1 perfect information,
    taking the second-best asset in the same category while the best is still
    available is dominated: both consume the same roster slot.
    """

    name = "lookahead2"

    def __init__(self, focal_picks: int = 3) -> None:
        if focal_picks < 1:
            raise ValueError("focal_picks must be >= 1")
        self.focal_picks = focal_picks
        self._vorp = VorpStrategy()

    @staticmethod
    def _vorp_value(asset: DraftAsset, context: DraftContext) -> float:
        return asset.value - context.replacement_levels.get(asset.category, 0.0)

    def _category_candidates(self, context: DraftContext) -> list[DraftAsset]:
        eligible = context.eligible_assets()
        if not eligible:
            return []

        best_by_category: dict[str, DraftAsset] = {}
        for asset in eligible:
            current = best_by_category.get(asset.category)
            if current is None:
                best_by_category[asset.category] = asset
                continue

            asset_key = (self._vorp_value(asset, context), asset.value, asset.name)
            current_key = (self._vorp_value(current, context), current.value, current.name)
            if asset_key > current_key:
                best_by_category[asset.category] = asset

        return list(best_by_category.values())

    @staticmethod
    def _picks_until_next_from_future_order(
        future_order: tuple[int, ...], gm_id: int
    ) -> int | None:
        for index, future_gm in enumerate(future_order):
            if future_gm == gm_id:
                return index
        return None

    def _advance_to_next_focal_pick(
        self,
        context: DraftContext,
        candidate: DraftAsset,
    ) -> tuple[DraftContext | None, int]:
        """Take ``candidate`` and simulate VORP opponents to the next focal pick."""
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
            )
            chosen = self._vorp.choose(opponent_context)
            available.remove(chosen)
            rosters.setdefault(future_gm, []).append(chosen)

        return None, len(context.future_order)

    def _best_plan(self, context: DraftContext, picks_remaining: int) -> _Plan:
        if picks_remaining <= 0:
            return _Plan(0.0, (), ())

        candidates = self._category_candidates(context)
        if not candidates:
            return _Plan(0.0, (), ())

        best: _Plan | None = None
        for candidate in candidates:
            current_vorp = self._vorp_value(candidate, context)
            next_context, gap = self._advance_to_next_focal_pick(context, candidate)

            future = _Plan(0.0, (), ())
            if picks_remaining > 1 and next_context is not None:
                future = self._best_plan(next_context, picks_remaining - 1)

            plan = _Plan(
                total_vorp=current_vorp + future.total_vorp,
                assets=(candidate, *future.assets),
                gaps=((gap,) + future.gaps) if next_context is not None else future.gaps,
            )

            if best is None:
                best = plan
                continue

            plan_key = (
                plan.total_vorp,
                current_vorp,
                candidate.value,
                candidate.name,
            )
            best_current = best.assets[0]
            best_key = (
                best.total_vorp,
                self._vorp_value(best_current, context),
                best_current.value,
                best_current.name,
            )
            if plan_key > best_key:
                best = plan

        assert best is not None
        return best

    def evaluate(self, context: DraftContext) -> list[Lookahead2Evaluation]:
        candidates = self._category_candidates(context)
        if not candidates:
            raise RuntimeError("No eligible asset remains for this roster")

        evaluations: list[Lookahead2Evaluation] = []
        for candidate in candidates:
            current_vorp = self._vorp_value(candidate, context)
            next_context, gap = self._advance_to_next_focal_pick(context, candidate)

            future = _Plan(0.0, (), ())
            if self.focal_picks > 1 and next_context is not None:
                future = self._best_plan(next_context, self.focal_picks - 1)

            gaps = ((gap,) + future.gaps) if next_context is not None else future.gaps
            sequence = (candidate.name, *(asset.name for asset in future.assets))
            evaluations.append(
                Lookahead2Evaluation(
                    candidate=candidate,
                    current_vorp=current_vorp,
                    future_vorp=future.total_vorp,
                    total_three_pick_value=current_vorp + future.total_vorp,
                    planned_sequence=sequence,
                    intervening_gaps=gaps,
                )
            )

        return evaluations

    def choose(self, context: DraftContext) -> DraftAsset:
        evaluations = self.evaluate(context)
        best = max(
            evaluations,
            key=lambda evaluation: (
                evaluation.total_three_pick_value,
                evaluation.current_vorp,
                evaluation.candidate.value,
                evaluation.candidate.name,
            ),
        )
        return best.candidate
