from nhl_draft_lab.draft.engine import replacement_levels, run_draft
from nhl_draft_lab.models import DraftAsset, RosterConfig
from nhl_draft_lab.strategies.basic import BasicStrategy


def test_replacement_level_matches_gms_times_slots():
    assets = [DraftAsset(str(i), f"F{i}", "F", 100 - i) for i in range(1, 11)]
    levels = replacement_levels(assets, gm_count=2, roster_config=RosterConfig({"F": 3}))
    # 2 GMs * 3 F = F6 => value 94.
    assert levels["F"] == 94


def test_engine_fills_rosters():
    assets = [
        *[DraftAsset(f"f{i}", f"F{i}", "F", 100 - i) for i in range(1, 9)],
        *[DraftAsset(f"d{i}", f"D{i}", "D", 50 - i) for i in range(1, 5)],
    ]
    config = RosterConfig({"F": 2, "D": 1})
    strategies = {1: BasicStrategy(), 2: BasicStrategy()}
    result = run_draft(assets, 2, config, strategies)
    assert all(len(roster) == 3 for roster in result.rosters.values())
    assert len(result.picks) == 6
