from __future__ import annotations

from typing import Protocol

from nhl_draft_lab.models import DraftAsset, DraftContext


class Strategy(Protocol):
    name: str

    def choose(self, context: DraftContext) -> DraftAsset:
        ...
