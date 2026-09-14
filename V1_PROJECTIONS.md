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
