from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from nhl_draft_lab.models import DraftAsset, DraftContext, DraftPick, RosterConfig
from nhl_draft_lab.strategies.base import Strategy


@dataclass(frozen=True)
class DraftResult:
    picks: list[DraftPick]
    rosters: dict[int, list[DraftAsset]]
    replacement_levels: dict[str, float]

    def totals(self) -> dict[int, float]:
        return {gm: sum(a.value for a in roster) for gm, roster in self.rosters.items()}


def snake_order(gm_count: int, rounds: int) -> list[int]:
    order: list[int] = []
    for round_no in range(1, rounds + 1):
        ids = range(1, gm_count + 1) if round_no % 2 else range(gm_count, 0, -1)
        order.extend(ids)
    return order


def replacement_levels(
    assets: list[DraftAsset], gm_count: int, roster_config: RosterConfig
) -> dict[str, float]:
    levels: dict[str, float] = {}
    for category, slots in roster_config.counts.items():
        if slots <= 0:
            continue
        ranked = sorted((a.value for a in assets if a.category == category), reverse=True)
        replacement_rank = gm_count * slots
        if len(ranked) < replacement_rank:
            raise ValueError(
                f"Not enough {category} assets: need rank {replacement_rank}, have {len(ranked)}"
            )
        levels[category] = ranked[replacement_rank - 1]
    return levels


def _picks_until_next(order: list[int], current_index: int, gm_id: int) -> int | None:
    for future_index in range(current_index + 1, len(order)):
        if order[future_index] == gm_id:
            return future_index - current_index - 1
    return None


def run_draft(
    assets: list[DraftAsset],
    gm_count: int,
    roster_config: RosterConfig,
    strategies: Mapping[int, Strategy],
) -> DraftResult:
    rounds = roster_config.total_rounds()
    order = snake_order(gm_count, rounds)
    available = list(assets)
    rosters: dict[int, list[DraftAsset]] = defaultdict(list)
    picks: list[DraftPick] = []
    levels = replacement_levels(assets, gm_count, roster_config)

    for index, gm_id in enumerate(order):
        strategy = strategies[gm_id]
        round_no = index // gm_count + 1
        context = DraftContext(
            gm_id=gm_id,
            draft_slot=gm_id,
            overall_pick=index + 1,
            round_no=round_no,
            picks_until_next=_picks_until_next(order, index, gm_id),
            roster=tuple(rosters[gm_id]),
            roster_config=roster_config,
            available_assets=tuple(available),
            replacement_levels=levels,
            rosters_by_gm={gm: tuple(roster) for gm, roster in rosters.items()},
            future_order=tuple(order[index + 1 :]),
            initial_assets=tuple(assets),
        )
        chosen = strategy.choose(context)
        if chosen not in available:
            raise RuntimeError(f"Strategy {strategy.name} selected unavailable asset {chosen.name}")
        if not context.category_needed(chosen.category):
            raise RuntimeError(f"Strategy {strategy.name} selected filled category {chosen.category}")

        available.remove(chosen)
        rosters[gm_id].append(chosen)
        picks.append(DraftPick(index + 1, round_no, gm_id, chosen, strategy.name))

    return DraftResult(picks=picks, rosters=dict(rosters), replacement_levels=levels)
