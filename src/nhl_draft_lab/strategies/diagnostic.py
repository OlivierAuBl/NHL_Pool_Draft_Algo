from __future__ import annotations

from dataclasses import dataclass

from nhl_draft_lab.models import DraftAsset, DraftContext
from nhl_draft_lab.strategies.base import Strategy


@dataclass(frozen=True)
class DecisionComparison:
    overall_pick: int
    round_no: int
    picks_until_next: int | None
    chosen_name: str
    chosen_category: str
    chosen_value: float
    baseline_name: str
    baseline_category: str
    baseline_value: float
    diverged: bool
    direct_points_delta: float
    direct_vorp_delta: float


class DiagnosticStrategy:
    """Wrap a focal strategy and compare each decision with a baseline strategy.

    For classic Stage-1 strategies the baseline is normally greedy VORP. For
    ``tier_lookahead`` the useful baseline is ``tier_vorp`` so diagnostics tell
    us whether lookahead improves the tiered decision rule rather than merely
    whether it differs from raw VORP.
    """

    def __init__(self, wrapped: Strategy, baseline: Strategy) -> None:
        self.wrapped = wrapped
        self.name = wrapped.name
        self.records: list[DecisionComparison] = []
        self._baseline = baseline

    @staticmethod
    def _vorp_value(asset: DraftAsset, context: DraftContext) -> float:
        return asset.value - context.replacement_levels.get(asset.category, 0.0)

    def choose(self, context: DraftContext) -> DraftAsset:
        baseline = self._baseline.choose(context)
        chosen = self.wrapped.choose(context)

        chosen_vorp = self._vorp_value(chosen, context)
        baseline_vorp = self._vorp_value(baseline, context)
        diverged = chosen.entity_id != baseline.entity_id

        self.records.append(
            DecisionComparison(
                overall_pick=context.overall_pick,
                round_no=context.round_no,
                picks_until_next=context.picks_until_next,
                chosen_name=chosen.name,
                chosen_category=chosen.category,
                chosen_value=chosen.value,
                baseline_name=baseline.name,
                baseline_category=baseline.category,
                baseline_value=baseline.value,
                diverged=diverged,
                direct_points_delta=(chosen.value - baseline.value) if diverged else 0.0,
                direct_vorp_delta=(chosen_vorp - baseline_vorp) if diverged else 0.0,
            )
        )
        return chosen
