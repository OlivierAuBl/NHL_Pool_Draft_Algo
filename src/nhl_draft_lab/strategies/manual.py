from __future__ import annotations

from nhl_draft_lab.models import DraftAsset, DraftContext


class ManualStrategy:
    name = "manual"

    def __init__(self, show_per_category: int = 5) -> None:
        self.show_per_category = show_per_category

    def choose(self, context: DraftContext) -> DraftAsset:
        eligible = context.eligible_assets()
        if not eligible:
            raise RuntimeError("No eligible asset remains for this roster")

        shown: list[DraftAsset] = []
        for category in ("F", "D", "G", "T"):
            if not context.category_needed(category):
                continue
            top = sorted(
                (a for a in eligible if a.category == category),
                key=lambda a: a.value,
                reverse=True,
            )[: self.show_per_category]
            shown.extend(top)

        shown = sorted(shown, key=lambda a: a.value, reverse=True)
        print(f"\nGM {context.gm_id} | pick #{context.overall_pick} | next in {context.picks_until_next} opponent picks")
        for i, asset in enumerate(shown, start=1):
            print(f"  {i:2d}. [{asset.category}] {asset.name:28s} {asset.value:7.2f} {asset.nhl_team}")

        while True:
            raw = input("Choose displayed number: ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(shown):
                    return shown[idx]
            except ValueError:
                pass
            print("Invalid choice.")
