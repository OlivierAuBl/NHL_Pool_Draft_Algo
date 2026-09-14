# Patch 5 — capped Tier Pressure

This patch replaces the previous multiplicative `pressure-strength` experiment with a bounded pressure bonus.

## Formula

For each fixed pre-draft tier and each printable horizon:

```text
raw_pressure   = gap_to_next_tier * P(tier exhausted before return)
pressure_bonus = min(raw_pressure, cap * max(tier_median_vorp, 0))
score           = tier_median_vorp + pressure_bonus
```

The intent is that pressure can break a close tier decision, but cannot make a clearly inferior tier leapfrog a much better tier just because it has a large cliff.

Defaults:

```text
SHORT = 4 picks
MID   = 14 picks
LONG  = 24 picks
cap   = 0.03   # 3% of tier median VORP
```

`cap=0` is exactly TierVORP behaviour.

## Files to replace

Copy the patch files over the matching repo paths:

```text
src/nhl_draft_lab/pressure.py
src/nhl_draft_lab/backtest.py
src/nhl_draft_lab/launcher.py
src/nhl_draft_lab/strategies/factory.py
tests/test_tier_pressure.py
```

## Generate the printable sheet

PowerShell:

```powershell
nhl-draft tier-scores `
  --db data/nhl_history.sqlite `
  --season 20242025 `
  --tier-pressure-cap 0.03 `
  --output output/tier_scores_20242025.csv
```

The useful live columns remain deliberately small:

```text
category | tier | players | tier_value | size | gap_to_next | SHORT | MID | LONG
```

## Backtest one cap

```powershell
nhl-draft backtest `
  --db data/nhl_history.sqlite `
  --focal-strategies tier_pressure `
  --opponent-strategies tier_vorp `
  --tier-pressure-cap 0.03 `
  --diagnostics `
  --output-dir output/tier_pressure_cap_003
```

## Grid: 0%, 1%, 2%, 3%, 5%

Paste directly in PowerShell:

```powershell
$caps = 0.00, 0.01, 0.02, 0.03, 0.05

foreach ($cap in $caps) {
    Write-Host "=== TierPressure cap = $cap ==="

    nhl-draft backtest `
        --db data/nhl_history.sqlite `
        --focal-strategies tier_pressure `
        --opponent-strategies tier_vorp `
        --diagnostics `
        --tier-pressure-short-horizon 4 `
        --tier-pressure-mid-horizon 14 `
        --tier-pressure-long-horizon 24 `
        --tier-pressure-cap $cap `
        --output-dir "output/tier_pressure_cap_grid/cap_$cap"
}
```

## Validation performed

- `25 passed`
- `cap=0` matches TierVORP choice in the unit test
- pressure bonus never exceeds `cap * tier_value`
- a clearly inferior tier cannot leapfrog solely from a large cliff in the test fixture
- a close tier decision can still be reversed by pressure
- smoke backtest completed for 2024-25 with 15 GM and roster 10F / 3D / 2G / 1T
