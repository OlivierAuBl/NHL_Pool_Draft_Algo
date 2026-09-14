# V1.0 component projections

V1.0 is an auditable historical baseline. It keeps the frozen published-projection V0 unchanged.

## Model contract

For skaters:

```text
projected_points = projected_ppg × projected_gp
```

Both components are recency-weighted means of the last three available seasons by default. Power-play points per game and shooting percentage are emitted as diagnostic features, but they do not alter the prediction until a backtest demonstrates an incremental gain.

For goalies:

```text
projected_starts = 82 × projected_start_share × availability
projected_points = projected_starts × (
    2 × wins_per_start
  + 3 × shutouts_per_start
  + 1 × ot_losses_per_start
  + 1 × goals_per_start
  + 1 × assists_per_start
)
```

`games_started` is used when the NHL source supplies it. Otherwise the output explicitly reports `games_played_fallback`. Availability defaults to `1.0`; V1.0 does not pretend that injury availability can be identified separately from historical workload.

Team assets use recency-weighted win and OTL rates. The joint team–goalie feedback model is intentionally deferred until the component baselines can be backtested.

## Manual context CSV

Only `entity_id` is required. One row per entity is allowed.

Context columns such as these are copied to the output without changing the projection:

- `projected_line`
- `projected_pp_unit`
- `future_linemate_quality`
- `injury_note`
- `source_note`

Supported explicit overrides are:

- `projected_gp_override`
- `projected_ppg_override`
- `availability` (absolute season availability for skaters; multiplier on workload share for goalies)
- `projected_start_share_override`
- `projected_win_rate_override`
- `projected_shutout_rate_override`
- `projected_otl_rate_override`

All rate and availability overrides must be between 0 and 1. GP overrides are clipped to the configured season length. Manual role data therefore remains distinguishable from statistically estimated effects.

## Example

```csv
entity_id,projected_line,projected_pp_unit,future_linemate_quality,availability,injury_note
8478402,1,1,0.92,,Top line and PP1
8476945,,,,0.75,Expected to miss roughly one quarter of the season
```

## CLI

```bash
nhl-draft project-v1 \
  --db data/nhl_history.sqlite \
  --history-seasons 20222023 20232024 20242025 \
  --manual-context data/v1_manual_context.csv \
  --output output/v1/projections.csv
```

The generated CSV contains `entity_id`, `name`, `category`, `projected_points` and `stddev_points`, so it can be read by the existing projection loader.

## Evaluation against frozen V0

```bash
nhl-draft evaluate-v1 \
  --v0-master output/v0_full_analysis/02_projection_master.csv \
  --v1 output/v1/projections.csv \
  --output-dir output/v1/evaluation
```

The evaluation uses V0's master table as the target-season universe. Historical players absent from that table are discarded. Active players without usable history retain their V0 projection as an explicit fallback.

The four reported model rows distinguish:

- V0 on the exact subset covered by V1;
- V1 historical components on that same subset;
- V1 plus V0 fallback on the complete active universe;
- V0 on the complete active universe.

This prevents coverage differences from being mistaken for model improvements. Skater GP and PPG errors are exported separately in `04_v1_skater_component_errors.csv`.

`06_v1_draft_zone_metrics.csv` restricts the main diagnostic to a broad
draft-relevant zone. The default cutoffs are F200, D75, G45 and T25, leaving a
buffer around the assets likely to be selected without including the full
historical depth. Ranks are based on the preseason V0 projection, not realised
results. Defensemen ranked in the top 6 and top 9 are also reported separately
so their signal is not diluted by lower-value defensemen.

The pool format and defense windows are configurable:

```bash
nhl-draft evaluate-v1 \
  --v0-master output/v0_full_analysis/02_projection_master.csv \
  --v1 output/v1/projections.csv \
  --draft-counts F=200,D=75,G=45,T=25 \
  --defense-focus-ranks 6 9
```

`07_v1_blend_grid.csv` evaluates the convex blend
`V0 + w × (V1 - V0)` from 0% to 100% V1 by default. Results are reported for
each draft-relevant category and the D top-6/top-9 windows. The lowest-MAE
weight is marked as `is_best_mae_in_sample`; this is an exploratory diagnostic,
not a production coefficient selected from one backtest season.

`08_v1_draft_zone_disagreements.csv` lists every covered asset in the broad
draft zone, ordered by the absolute difference between V0 and V1. It includes
the realised result and identifies which baseline was closer.
