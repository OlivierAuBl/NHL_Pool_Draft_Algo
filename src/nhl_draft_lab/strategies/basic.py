from __future__ import annotations

from nhl_draft_lab.models import DraftAsset, DraftContext


class BasicStrategy:
    """Take the highest-value player regardless of position, subject to roster needs."""

    name = "basic"

    def choose(self, context: DraftContext) -> DraftAsset:
        eligible = context.eligible_assets()
        if not eligible:
            raise RuntimeError("No eligible asset remains for this roster")
        return max(eligible, key=lambda a: (a.value, a.name))
