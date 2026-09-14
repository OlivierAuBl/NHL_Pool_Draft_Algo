import pandas as pd
import numpy as np
from pathlib import Path

OUTDIR = Path("output/v0_projection_tiers")
OUTDIR.mkdir(parents=True, exist_ok=True)

SKATER_PATH = Path("data/Prediction_Skater_20252026.csv")
GOALIE_PATH = Path("data/Prediction_goalies_20252026.csv")
TEAM_PATH = Path("data/Prediction_Team_20252026.csv")
DB_PATH = Path("data/nhl_pool_analysis.sqlite")

GM_COUNT = 15
ROSTER = {"F": 10, "D": 3, "G": 2, "T": 1}

TIER_RELATIVE_WIDTH = 0.05
TIER_MAX_SIZE = 5
SUPERSTAR_MAX_SIZE = 3
SUPERSTAR_TIERS = 2

SKATER_SOURCES = [
    "ScottCullen", "CBS", "LineupExpert", "ESPN",
    "PoolPro", "HockeyMag", "Hastag", "PoolExpert",
]
GOALIE_SOURCES = [
    "calc_Hashtag_Score", "calc_HockeyMag_Score",
    "calc_LineupExpert_Score", "calc_PoolPro_Score",
    "calc_ScottCullen_Score", "calc_ESPN_Score",
    "calc_CBS_Score", "calc_PoolExpert_Score",
]
TEAM_SOURCES = [
    "ScottCullen", "CBS", "LineupExpert", "ESPN",
    "PoolPro", "PoolExpert", "Hastag",
]

sk = pd.read_csv(SKATER_PATH)
go = pd.read_csv(GOALIE_PATH)
tm = pd.read_csv(TEAM_PATH)

# Deduplicate skaters, preferring a valid F/D position when duplicate rows exist.
sk["_valid_pos"] = sk["Pos"].isin(["F", "D"]).astype(int)
sk = (
    sk.sort_values(["NHLID", "_valid_pos"], ascending=[True, False])
      .drop_duplicates("NHLID", keep="first")
      .drop(columns="_valid_pos")
      .copy()
)

# Position fixes for rows that were recorded as Pos=0 in the source.
# These are factual position mappings only; no performance/history information is used.
POSITION_FIX = {
    8485702: "F",  # Max Shabanov
    8483455: "F",  # Isaac Howard
    8483525: "F",  # Danila Yurov
    8476455: "F",  # Gabriel Landeskog
    8485402: "F",  # Michael Misa
    8481721: "F",  # Arseny Gritsyuk
    8482100: "D",  # Alexander Nikishin
    8484798: "D",  # Zeev Buium
    8475169: "F",  # Evander Kane
    8473604: "F",  # Jonathan Toews
}
sk["category"] = sk.apply(
    lambda r: r["Pos"] if r["Pos"] in ("F", "D") else POSITION_FIX.get(int(r["NHLID"]), r["Pos"]),
    axis=1,
)

def clean_projection_values(row, source_cols, injury_col=None):
    """
    V0 rules:
      - #N/A / blanks -> missing
      - 0 with no injury flag -> treated as source non-coverage
      - 0 with injury flag -> retained as a real zero projection
    """
    injury = ""
    if injury_col and pd.notna(row.get(injury_col)):
        injury = str(row[injury_col]).strip()

    vals = []
    used_sources = []
    ignored_zero_sources = []

    for col in source_cols:
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        value = float(value)

        if value == 0 and not injury:
            ignored_zero_sources.append(col)
            continue

        vals.append(value)
        used_sources.append(col)

    return vals, used_sources, ignored_zero_sources

def add_consensus(df, source_cols, injury_col=None):
    values = []
    used = []
    ignored = []
    for _, row in df.iterrows():
        v, u, i = clean_projection_values(row, source_cols, injury_col)
        values.append(v)
        used.append(u)
        ignored.append(i)

    out = df.copy()
    out["projection_median"] = [float(np.median(v)) if v else np.nan for v in values]
    out["projection_mean"] = [float(np.mean(v)) if v else np.nan for v in values]
    out["projection_min"] = [float(np.min(v)) if v else np.nan for v in values]
    out["projection_max"] = [float(np.max(v)) if v else np.nan for v in values]
    out["projection_mad"] = [
        float(np.median(np.abs(np.array(v) - np.median(v)))) if v else np.nan
        for v in values
    ]
    out["n_sources"] = [len(v) for v in values]
    out["sources_used"] = [" | ".join(u) for u in used]
    out["ignored_zero_sources"] = [" | ".join(i) for i in ignored]
    return out

sk = add_consensus(sk, SKATER_SOURCES, "DFO_Injury")
go = add_consensus(go, GOALIE_SOURCES, "DFO_Injury")
tm = add_consensus(tm, TEAM_SOURCES, None)

go["category"] = "G"
tm["category"] = "T"

# Non-scoring annotations: they do NOT affect V0 projections or tier placement.
manual_notes = {
    "Nikita Kucherov": "User manually rejected an implausibly low ESPN forecast; ESPN is already #N/A in consolidated CSV.",
    "Aleksander Barkov": "Known at draft: ACL preseason, out for season. Annotation only in V0; no projection adjustment.",
    "Matthew Tkachuk": "Known at draft: injured, long absence possible/uncertain; later confirmed out until January. Annotation only in V0.",
    "Zach Hyman": "Known at draft: preseason injury, roughly 1.5 months expected absence. Annotation only in V0.",
    "Mats Zuccarello": "Known at draft: preseason injury, roughly 1.5 months expected absence. Annotation only in V0.",
    "Logan Cooley": "Serious injury occurred during season; no pre-draft adjustment in V0.",
    "Jared McCann": "In-season injury / team underperformance; no pre-draft adjustment in V0.",
    "J.T. Miller": "In-season injury / team underperformance; no pre-draft adjustment in V0.",
}
sk["manual_note"] = sk["FullName"].map(manual_notes).fillna("")

# PoolPro Top is deliberately excluded from the central consensus.
sk["projection_ceiling_poolpro"] = pd.to_numeric(sk["PoolPro_Top"], errors="coerce")
go["projection_ceiling_poolpro"] = pd.to_numeric(go["calc_PoolPro_Top_Score"], errors="coerce")
tm["projection_ceiling_poolpro"] = np.nan

# Build replacement levels from projected medians.
category_frames = {
    "F": sk[sk["category"] == "F"].copy(),
    "D": sk[sk["category"] == "D"].copy(),
    "G": go.copy(),
    "T": tm.copy(),
}

replacement = {}
for cat, frame in category_frames.items():
    ranked = frame.dropna(subset=["projection_median"]).sort_values(
        "projection_median", ascending=False
    )
    replacement_rank = GM_COUNT * ROSTER[cat]
    replacement[cat] = float(ranked.iloc[replacement_rank - 1]["projection_median"])

def build_tiers(frame, category, id_col, name_col):
    ranked = frame.dropna(subset=["projection_median"]).copy()
    ranked["projected_vorp"] = ranked["projection_median"] - replacement[category]
    ranked = ranked.sort_values(
        ["projected_vorp", "projection_median", name_col],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    tier_rows = []
    player_rows = []
    cursor = 0

    while cursor < len(ranked):
        tier_no = len(tier_rows) + 1
        leader = ranked.iloc[cursor]
        leader_vorp = float(leader["projected_vorp"])
        denom = max(abs(float(leader["projection_median"])), 1e-9)
        cap = SUPERSTAR_MAX_SIZE if tier_no <= SUPERSTAR_TIERS else TIER_MAX_SIZE

        member_idx = [cursor]
        cursor += 1

        while cursor < len(ranked) and len(member_idx) < cap:
            candidate = ranked.iloc[cursor]
            relative_gap = (
                leader_vorp - float(candidate["projected_vorp"])
            ) / denom
            if relative_gap > TIER_RELATIVE_WIDTH:
                break
            member_idx.append(cursor)
            cursor += 1

        members = ranked.iloc[member_idx].copy()
        tier_value = float(members["projected_vorp"].median())

        tier_rows.append({
            "category": category,
            "tier": tier_no,
            "size": len(members),
            "tier_value_median_projected_vorp": tier_value,
            "leader_projected_vorp": float(members["projected_vorp"].max()),
            "min_projected_vorp": float(members["projected_vorp"].min()),
            "median_projection_points": float(members["projection_median"].median()),
            "replacement_level": replacement[category],
            "members": " | ".join(members[name_col].astype(str)),
        })

        members["tier"] = tier_no
        members["tier_value_median_projected_vorp"] = tier_value
        player_rows.append(members)

    tiers = pd.DataFrame(tier_rows)
    players = pd.concat(player_rows, ignore_index=True)
    return tiers, players

tier_frames = []
player_frames = []

for cat, frame, id_col, name_col in [
    ("F", category_frames["F"], "NHLID", "FullName"),
    ("D", category_frames["D"], "NHLID", "FullName"),
    ("G", category_frames["G"], "NHLID", "FullName"),
    ("T", category_frames["T"], "Team", "Team"),
]:
    tier_df, player_df = build_tiers(frame, cat, id_col, name_col)
    tier_frames.append(tier_df)
    player_frames.append(player_df)

tiers = pd.concat(tier_frames, ignore_index=True)
players = pd.concat(player_frames, ignore_index=True)

# Global tier priority = structural TierVORP ordering across categories.
tiers["global_priority"] = (
    tiers["tier_value_median_projected_vorp"]
    .rank(method="first", ascending=False)
    .astype(int)
)
tiers = tiers.sort_values(
    ["tier_value_median_projected_vorp", "category", "tier"],
    ascending=[False, True, True],
).reset_index(drop=True)

# Keep a convenient "draft-relevant" report through each category's replacement tier.
replacement_tier_no = {}
for cat in ["F", "D", "G", "T"]:
    cat_players = players[players["category"] == cat].sort_values(
        "projection_median", ascending=False
    )
    replacement_rank = GM_COUNT * ROSTER[cat]
    replacement_player = cat_players.iloc[replacement_rank - 1]
    replacement_tier_no[cat] = int(replacement_player["tier"])

draft_relevant = tiers[
    tiers.apply(lambda r: r["tier"] <= replacement_tier_no[r["category"]], axis=1)
].copy()

# Player-level export with consistent useful columns.
player_output_cols = [
    "category", "tier", "tier_value_median_projected_vorp",
    "NHLID", "FullName", "Team",
    "projection_median", "projected_vorp", "projection_mean",
    "projection_mad", "projection_min", "projection_max", "n_sources",
    "projection_ceiling_poolpro", "sources_used", "ignored_zero_sources",
    "manual_note",
]
# Team rows have different identifying columns.
players_out = players.copy()
if "NHLID" not in players_out.columns:
    players_out["NHLID"] = np.nan
if "FullName" not in players_out.columns:
    players_out["FullName"] = np.nan
if "Team" not in players_out.columns:
    players_out["Team"] = np.nan

for col in player_output_cols:
    if col not in players_out.columns:
        players_out[col] = np.nan

players_out = players_out[player_output_cols].sort_values(
    ["category", "tier", "projection_median"],
    ascending=[True, True, False],
)

tiers_path = OUTDIR / "v0_tiers_20252026.csv"
players_path = OUTDIR / "v0_players_20252026.csv"
draft_path = OUTDIR / "v0_draft_relevant_tiers_20252026.csv"

tiers.to_csv(tiers_path, index=False)
players_out.to_csv(players_path, index=False)
draft_relevant.to_csv(draft_path, index=False)

print("Replacement levels:", replacement)
print("Replacement tier numbers:", replacement_tier_no)
print()
print("Top global tiers:")
print(
    draft_relevant[
        ["global_priority", "category", "tier", "size",
         "tier_value_median_projected_vorp", "median_projection_points", "members"]
    ].head(25).to_string(index=False)
)
print()
print("Files created:")
print(tiers_path)
print(players_path)
print(draft_path)
