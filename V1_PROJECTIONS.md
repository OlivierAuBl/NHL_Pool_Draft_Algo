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

For hand-maintained context, `name` may be used instead of `entity_id` when it
matches exactly one projected asset. This keeps notes usable before NHL IDs
have been looked up.

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

## V0-PPG × V1-GP hybrid

The evaluator also tests a skater-only hybrid:

```text
V0 implied PPG = V0 projected points / 84
hybrid skater points = V0 implied PPG × V1 projected GP
```

For a forward without V1 history, projected GP defaults to 50; a defenseman
without history defaults to 60. Goalies and teams remain at V0 because the
hybrid only tests the skater PPG/GP decomposition. These assumptions are
configurable with `--v0-reference-games`, `--rookie-gp` and
`--rookie-defense-gp`. The draft-ready output is written to
`09_v0_ppg_v1_gp_projections.csv`.

`10_v1_gp_correction_grid.csv` then tests partial availability corrections for
skaters with history:

```text
projection(alpha) = V0 + alpha × (full hybrid - V0)
```

The default grid runs from `alpha=0` to `alpha=1` by tenths. The 50-GP forward
and 60-GP defense defaults remain fixed at every alpha, which isolates the
incremental value of historical GP. The lowest-MAE weight is an in-sample
diagnostic only and is not automatically used as a production coefficient.

## Weighted GP candidate

The current candidate applies the best broad draft-zone weights from the
diagnostic grid: 30% of the historical-GP correction for forwards and 60% for
defensemen. No-history defaults remain 50 GP for F and 60 GP for D; goalies and
teams remain at V0. The weights can be changed with `--forward-gp-weight` and
`--defense-gp-weight`.

`11_v1_weighted_gp_candidate.csv` follows the projection-loader contract and
can therefore be supplied directly to the draft engine. It remains explicitly
an in-sample candidate until it is validated on additional seasons.

## Draft-outcome evaluation

`evaluate-draft-v1` first runs an all-V0 reference draft. It then repeats the
draft once per slot, with only that focal GM using the weighted-GP candidate
while every opponent continues to use V0. The strategy, roster rules, GM count
and frozen player universe stay constant. Selected rosters are scored with
actual season points.

```bash
nhl-draft evaluate-draft-v1 \
  --v0-master output/v0_full_analysis/02_projection_master.csv \
  --candidate output/v1/evaluation/11_v1_weighted_gp_candidate.csv \
  --strategies vorp tier_vorp \
  --gms 15 \
  --roster F=10,D=3,G=2,T=1 \
  --output-dir output/v1/draft_evaluation
```

The exports contain aggregate results, every draft-slot result, every pick,
paired candidate-minus-V0 deltas by slot, category-level attribution and a
projection-match audit. A positive actual-points delta favours the candidate;
a negative rank or gap-to-winner delta favours the candidate. The category
output makes it possible to verify whether a gain comes from F, D, G or T.
Each paired delta therefore measures the competitive benefit of giving the
candidate projections to one GM, rather than comparing two leagues where
everyone has the same information. These are deterministic historical
backtests, not independent trials, and the candidate's GP weights remain
in-sample for the evaluated season.

`08_candidate_roster_swap_summary.csv` explains each category delta through
the actual points added and removed. It also reports the candidate-minus-V0
movement in the first round and overall pick used for that category; a negative
movement means the candidate drafted the category earlier. The player-level
ledger is exported to `09_candidate_roster_swaps.csv`, where signed actual
contributions reconcile to the summary delta.
