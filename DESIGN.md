# Design — Stage 1 only

## Objective

Separate two problems that must not be mixed yet:

1. **Draft policy:** if the true end-of-season values were known, how should we draft?
2. Forecasting those values.

This repository currently solves only problem 1.

## Perfect-information assumption

`DraftAsset.value` is the realised final pool score from a historical NHL season, optionally normalised to 84 games.

Thus a bad Stage-1 result cannot be blamed on projection error. It is a property of the selection rule itself.

## Current strategy ladder

1. `BasicStrategy`: raw points only.
2. `VorpStrategy`: fixed category replacement level.
3. `PlateauStrategy`: fixed VORP plus local curve-shape information.

No snake-timing / wait-loss term is included in Plateau V0. That should be introduced only as a distinct next hypothesis, because otherwise we would not know whether an improvement came from curve shape or from draft-slot timing.

## Evaluation design

For each season and each draft slot:

- choose one focal strategy;
- choose one opponent strategy used by all other GMs;
- run the deterministic snake draft;
- record focal points, final rank, gap to winner and margin to field mean.

Across seasons and slots, summarize:

- mean rank;
- win rate;
- top-3 rate;
- mean points;
- mean gap to winner;
- mean margin versus field.

Pairwise opponent models matter because a policy can exploit Basic opponents but perform differently against VORP opponents.

## Next Stage-1 hypotheses, not yet implemented

- Plateau parameter calibration (`window`, `weight`).
- Snake-aware wait loss based on the actual number of picks until the GM's next turn.
- Treat consecutive turn picks as a joint decision near the snake edges.
- Counterfactual pick regret: after a draft, test alternative choices at selected decision points while holding an explicit continuation policy fixed.

Only after those are understood should Stage 2 projections be introduced.
