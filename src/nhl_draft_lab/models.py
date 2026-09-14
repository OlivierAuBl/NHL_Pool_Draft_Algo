from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DraftAsset:
    entity_id: str
    name: str
    category: str
    value: float
    nhl_team: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RosterConfig:
    counts: Mapping[str, int]

    def total_rounds(self) -> int:
        return sum(self.counts.values())

    def required(self, category: str) -> int:
        return int(self.counts.get(category, 0))


@dataclass(frozen=True)
class DraftPick:
    overall_pick: int
    round_no: int
    gm_id: int
    asset: DraftAsset
    strategy_name: str


@dataclass(frozen=True)
class DraftContext:
    gm_id: int
    draft_slot: int
    overall_pick: int
    round_no: int
    picks_until_next: int | None
    roster: Sequence[DraftAsset]
    roster_config: RosterConfig
    available_assets: Sequence[DraftAsset]
    replacement_levels: Mapping[str, float]
    # Stage-1 lookahead state.  Empty defaults keep existing strategies/tests
    # compatible while allowing an omniscient strategy to simulate the snake.
    rosters_by_gm: Mapping[int, Sequence[DraftAsset]] = field(default_factory=dict)
    future_order: Sequence[int] = field(default_factory=tuple)
    # Complete immutable draft universe, used by fixed pre-draft tier strategies.
    initial_assets: Sequence[DraftAsset] = field(default_factory=tuple)

    def roster_count(self, category: str) -> int:
        return sum(1 for asset in self.roster if asset.category == category)

    def category_needed(self, category: str) -> bool:
        return self.roster_count(category) < self.roster_config.required(category)

    def eligible_assets(self) -> list[DraftAsset]:
        return [a for a in self.available_assets if self.category_needed(a.category)]
