from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Mapping

from nhl_draft_lab.data.projections import Projection
from nhl_draft_lab.draft.engine import run_draft
from nhl_draft_lab.models import RosterConfig
from nhl_draft_lab.strategies.base import Strategy


@dataclass(frozen=True)
class MonteCarloSummary:
    gm_id: int
    mean_points: float
    std_points: float
    win_rate: float
    top3_rate: float


def run_monte_carlo(
    projections: list[Projection],
    gm_count: int,
    roster_config: RosterConfig,
    strategies: Mapping[int, Strategy],
    iterations: int = 10_000,
    seed: int | None = 42,
) -> list[MonteCarloSummary]:
    """V0 Monte Carlo.

    Draft choices are made once using projection means. Then each iteration samples
    realized fantasy points from N(mean, stddev), truncated at zero.

    This is intentionally simple. Future extensions can redraw draft rankings,
    model injury/games-played separately, correlate linemates/teams, or randomize
    opponent behavior.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    draft = run_draft(
        assets=[p.asset for p in projections],
        gm_count=gm_count,
        roster_config=roster_config,
        strategies=strategies,
    )
    by_id = {p.asset.entity_id: p for p in projections}
    rng = random.Random(seed)

    scores: dict[int, list[float]] = defaultdict(list)
    wins: dict[int, int] = defaultdict(int)
    top3: dict[int, int] = defaultdict(int)

    for _ in range(iterations):
        iteration_scores: dict[int, float] = {}
        for gm_id, roster in draft.rosters.items():
            total = 0.0
            for asset in roster:
                projection = by_id[asset.entity_id]
                total += max(0.0, rng.gauss(projection.mean, projection.stddev))
            iteration_scores[gm_id] = total
            scores[gm_id].append(total)

        ordered = sorted(iteration_scores.items(), key=lambda kv: kv[1], reverse=True)
        if ordered:
            best = ordered[0][1]
            winners = [gm for gm, score in ordered if score == best]
            for gm in winners:
                wins[gm] += 1 / len(winners)
            for gm, _score in ordered[:3]:
                top3[gm] += 1

    summaries: list[MonteCarloSummary] = []
    for gm_id in range(1, gm_count + 1):
        gm_scores = scores[gm_id]
        summaries.append(MonteCarloSummary(
            gm_id=gm_id,
            mean_points=fmean(gm_scores),
            std_points=pstdev(gm_scores),
            win_rate=float(wins[gm_id]) / iterations,
            top3_rate=float(top3[gm_id]) / iterations,
        ))
    return summaries
