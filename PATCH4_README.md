# Patch 4 — Printable Tier Pressure

Stage 1 remains perfect-information / historical. This patch adds a new strategy and a printable tier score sheet.

## Core idea

Each fixed VORP tier receives exactly three pre-draft scores:

- `SHORT` — default horizon: 4 picks before your next turn
- `MID` — default horizon: 14 picks
- `LONG` — default horizon: 28 picks

The scores never change during the draft. When a tier is exhausted, move to the next tier row.

For a tier:

`Score = median tier VORP + strength * gap_to_next_tier * exhaustion_risk`

The exhaustion risk is precomputed. It uses:

1. tier size;
2. the SHORT / MID / LONG horizon;
3. roster slot proportions (`10F / 3D / 2G / 1T` by default);
4. the tier's median VORP relative to competing tiers at a similar draft depth.

If competing tier values are equal, estimated pick demand falls back to roster proportions. If (for example) a 2-player D tier at VORP ~80 is competing with F tiers around 65–70, the D demand share rises sharply.

No opponent roster state is used. The whole sheet can therefore be printed before the draft.

## New files

- `src/nhl_draft_lab/pressure.py`
- `src/nhl_draft_lab/strategies/tier_pressure.py`
- `tests/test_tier_pressure.py`

## Replaced files

- `src/nhl_draft_lab/strategies/factory.py`
- `src/nhl_draft_lab/backtest.py`
- `src/nhl_draft_lab/launcher.py`

## Generate the printable sheet

PowerShell:

```powershell
nhl-draft tier-scores `
  --db data/nhl_history.sqlite `
  --season 20242025 `
  --output output/tier_scores_20242025.csv
```

The useful live columns are:

`category | tier | players | tier_value | size | gap_to_next | SHORT | MID | LONG`

The CSV also contains `estimated_pick_share` and `market_pick` for diagnostics, but they do not need to be printed.

## Backtest TierPressure

```powershell
nhl-draft backtest `
  --db data/nhl_history.sqlite `
  --focal-strategies tier_vorp tier_pressure `
  --opponent-strategies tier_vorp `
  --diagnostics `
  --output-dir output/tier_pressure
```

`tier_pressure` is diagnosed against `tier_vorp`, not against raw VORP.

## Calibration parameters

```text
--tier-pressure-short-horizon 4
--tier-pressure-mid-horizon 14
--tier-pressure-long-horizon 28
--tier-pressure-temperature 10
--tier-pressure-strength 1.0
```

`temperature` controls how strongly relative tier quality changes expected pick demand. Lower = more sensitive to value gaps.

`strength` scales only the pressure bonus. This is the cleanest first calibration knob. Suggested Stage-1 grid:

```text
0.25, 0.50, 0.75, 1.00
```

Do not select the final value from a single season. Use all historical seasons, then revisit after the historical projection backtest.

## Current validation

- 23 unit tests pass.
- `tier-scores` smoke-tested on 2024-25.
- `tier_pressure` completed a full 15-GM / 16-round 2024-25 backtest.
- One-season smoke results suggest `strength=1.0` is too aggressive; `0.25` was much closer to TierVORP. This is only a smoke result, not a calibration conclusion.
