# Projection sources and team-stacking experiments

This document records the high-level conclusions from the 2024-2025 and
2025-2026 historical experiments. Generated CSV files remain under `output/`
and are intentionally not versioned.

## Experimental setup

- 16-GM snake draft, with rosters of 10 F, 3 D, 2 G and 1 T.
- One focal GM is evaluated from every draft slot.
- Rosters are scored with realised season points.
- The controlled field uses Pool Pro raw VOR unless a field experiment says
  otherwise.
- The team-aware rule only changes a pick when candidates are within 3 VOR
  points of the best available player.
- `stack` favours the candidate whose NHL team is already most represented on
  the roster; `diversify` does the reverse.
- Remaining ties are resolved by VOR, projected value and player name.

All reported results reproduced the raw-VOR baseline, produced valid rosters
and contained no duplicate picks.

## Projection sources and consensus

| Season | Best single source | Trimmed mean | High-level conclusion |
| --- | --- | --- | --- |
| 2024-2025 | FantasyUnknown: 1063.6 mean points, mean rank 1.00 | 1072.5 mean points, mean rank 1.00, 16/16 wins | The trimmed consensus beat every individual source. FantasyUnknown's edition is not confirmed, so its result is exploratory. |
| 2025-2026 | Scott Cullen: 1015.8 mean points, mean rank 2.00 | 998.4 mean points, mean rank 3.44 | The trimmed consensus beat the average individual source, but not the best source. |

Across the two seasons, min-max-trimmed mean is the most defensible general
baseline: it dominated in 2024-2025 and remained competitive in 2025-2026.
This does not establish that it will always beat the strongest individual
projection source.

The 2024-2025 source snapshots do not contain verifiable publication dates.
That season is therefore an exploratory historical reconstruction rather than
a strict point-in-time backtest. Hashtag goalies were excluded because OTL was
not compatible, and all team-source columns used the shared previous-season
fallback. In 2025-2026, Hockey Magazine teams used the same fallback.

## Three-point team tiebreaks

The most stable result appears when the focal GM uses the trimmed-mean
projection:

| Season | Stack: points delta | Stack: rank delta | Diversify: points delta | Diversify: rank delta |
| --- | ---: | ---: | ---: | ---: |
| 2024-2025 | +31.2 | 0.00 | -4.6 | 0.00 |
| 2025-2026 | +13.6 | -0.31 | -15.7 | +1.81 |

A negative rank delta is an improvement. Team stacking was positive in both
seasons for the trimmed consensus, whereas forced diversification was not.

The single-source results were much less stable. For example, stacking helped
ESPN by 28.8 points in 2024-2025 but hurt it by 20.8 in 2025-2026. In
2024-2025 it hurt Scott Cullen by 52.1 points and Pool Pro by 28.6; in
2025-2026 it helped Scott Cullen by 23.6. The rule should therefore not be
treated as universally beneficial independently of the projection source.

## Does the quality of the concentrated team matter?

The 2024-2025 evidence supports that hypothesis:

- correlation between actual NHL standings points and stacking value: +0.432;
- correlation between the team's top-five realised fantasy strength and
  stacking value: +0.519;
- teams in the top eight by realised fantasy strength: +23.7 points on
  average;
- teams ranked 9-16: -21.8; teams ranked 17-24: -29.9.

Examples follow the same pattern: Winnipeg, Toronto and Tampa Bay stacks were
generally helpful, while Utah and Pittsburgh stacks were harmful.

The relationship did not repeat in 2025-2026: both correlations were near
zero. Colorado and Washington produced strong gains, but there were also
large negative cases on otherwise strong teams. Detailed roster-swap
reconciliation showed that the exact marginal players added and removed
explain the result better than team quality alone.

The practical conclusion is narrower than the original hypothesis: stacking
can capture correlated upside, and a weak-team concentration can be harmful,
but a team-level label is not sufficient. Any production rule must use
information available before the draft, such as projected team offence and
the projected value of the actual players involved. Actual standings or final
player points would leak the answer into the strategy.

## Inverting the field

The experiment was also run with all 15 opponents using the stacking rule
while the focal GM retained raw VOR, and with the focal/field roles reversed.
The values below aggregate the tested single sources and trimmed mean.

| Season | Focal raw vs 15 stacking: points / rank delta | Focal stacking vs 15 raw: points / rank delta |
| --- | ---: | ---: |
| 2024-2025 | -2.0 / +0.09 | -3.9 / +1.03 |
| 2025-2026 | +16.0 / -1.17 | +18.6 / -1.19 |

Changing the other 15 GMs materially changes which players reach the focal
GM. The 2024-2025 aggregate effect was roughly neutral to negative and highly
source-dependent. In 2025-2026, both the focal stacker and the raw-VOR focal GM
benefited in their respective altered fields. This confirms that the tiebreak
cannot be evaluated only as an isolated ranking adjustment; it also reshapes
the draft board.

## Decision and next test

For now:

1. Keep min-max-trimmed mean as the robust multi-source baseline.
2. Retain stacking within 3 VOR points as an experimental tiebreak for that
   consensus method.
3. Do not adopt forced diversification or universal stacking across every
   source.
4. Before production use, test a pre-draft quality gate based only on projected
   team offence and candidate-level value, then validate it out of sample.

The exact reproduction commands are documented in the
[README](README.md#5-compare-projection-sources-and-consensus-methods).
