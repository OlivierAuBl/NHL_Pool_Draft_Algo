from __future__ import annotations

from collections import Counter
from typing import Literal

from nhl_draft_lab.models import DraftAsset, DraftContext


TeamPreference = Literal["stack", "diversify"]


class TeamAwareVorpStrategy:
    """Apply a team-composition tiebreak inside a near-best raw-VOR window."""

    def __init__(self, preference: TeamPreference, tolerance: float = 3.0) -> None:
        if preference not in {"stack", "diversify"}:
            raise ValueError("preference must be 'stack' or 'diversify'")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.preference = preference
        self.tolerance = float(tolerance)
        self.name = f"vorp_team_{preference}_within_{tolerance:g}"

    @staticmethod
    def _team_counts(context: DraftContext) -> Counter[str]:
        return Counter(
            asset.nhl_team
            for asset in context.roster
            if asset.category != "T" and asset.nhl_team
        )

    def choose(self, context: DraftContext) -> DraftAsset:
        eligible = context.eligible_assets()
        if not eligible:
            raise RuntimeError("No eligible asset remains for this roster")

        def vor(asset: DraftAsset) -> float:
            return asset.value - context.replacement_levels.get(asset.category, 0.0)

        best_vor = max(vor(asset) for asset in eligible)
        near_best = [
            asset for asset in eligible if best_vor - vor(asset) <= self.tolerance
        ]
        team_counts = self._team_counts(context)

        def preference_score(asset: DraftAsset) -> tuple[float, float, float, str]:
            count = team_counts.get(asset.nhl_team, 0) if asset.nhl_team else 0
            team_score = float(count if self.preference == "stack" else -count)
            return (team_score, vor(asset), asset.value, asset.name)

        return max(near_best, key=preference_score)
