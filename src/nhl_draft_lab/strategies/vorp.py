from __future__ import annotations

from nhl_draft_lab.models import DraftAsset, DraftContext


class VorpStrategy:
    """Fixed replacement-level strategy.

    Replacement level is determined from the initial player universe:
    GM count x required roster slots in the category.
    Example: 15 GM x 10 forwards => F150.
    """

    name = "vorp"

    def choose(self, context: DraftContext) -> DraftAsset:
        eligible = context.eligible_assets()
        if not eligible:
            raise RuntimeError("No eligible asset remains for this roster")

        def score(asset: DraftAsset) -> tuple[float, float, str]:
            replacement = context.replacement_levels.get(asset.category, 0.0)
            return (asset.value - replacement, asset.value, asset.name)

        return max(eligible, key=score)
