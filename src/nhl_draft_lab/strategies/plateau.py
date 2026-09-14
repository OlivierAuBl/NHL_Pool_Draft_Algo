from __future__ import annotations

from statistics import fmean

from nhl_draft_lab.models import DraftAsset, DraftContext


class PlateauStrategy:
    """VORP plus a local curve-shape bonus.

    V0 intentionally ignores snake-draft timing. For each still-needed category:

        VORP = best_available - fixed_replacement_level
        cliff = best_available - mean(next `window` available players)
        score = VORP + weight * cliff

    A flat category has a small cliff, so waiting is less costly. A category whose
    curve drops sharply receives a larger bonus. `weight` and `window` are left
    explicit so they can be calibrated later rather than hard-coded as truth.
    """

    name = "plateau"

    def __init__(self, window: int = 5, weight: float = 1.0) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.weight = weight

    def choose(self, context: DraftContext) -> DraftAsset:
        category_candidates: list[tuple[float, float, DraftAsset]] = []

        categories = sorted({a.category for a in context.available_assets})
        for category in categories:
            if not context.category_needed(category):
                continue
            available = sorted(
                (a for a in context.available_assets if a.category == category),
                key=lambda a: a.value,
                reverse=True,
            )
            if not available:
                continue

            best = available[0]
            followers = available[1 : 1 + self.window]
            follower_level = fmean(a.value for a in followers) if followers else best.value
            cliff = max(0.0, best.value - follower_level)
            replacement = context.replacement_levels.get(category, 0.0)
            vorp = best.value - replacement
            total = vorp + self.weight * cliff
            category_candidates.append((total, best.value, best))

        if not category_candidates:
            raise RuntimeError("No eligible asset remains for this roster")

        return max(category_candidates, key=lambda item: (item[0], item[1], item[2].name))[2]
