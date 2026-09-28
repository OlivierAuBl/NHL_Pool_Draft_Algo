from __future__ import annotations

import pandas as pd

from nhl_draft_lab.historical_projection_master import (
    _history_identity_map,
    _projection_rows,
    _resolve_player_id,
    normalize_name,
)


def test_normalize_name_removes_accents_and_punctuation():
    assert normalize_name("Marc-André Fleury") == "marc andre fleury"


def test_unresolved_projection_can_match_a_unique_historical_name():
    history = pd.DataFrame(
        [{"category": "F", "name": "Alex Kerfoot", "entity_id": 8477962}]
    )
    row = pd.Series(
        {
            "category": "F",
            "canonical_name": "Alex Kerfoot",
            "nhl_player_id": pd.NA,
        }
    )

    resolved, method = _resolve_player_id(row, _history_identity_map(history))

    assert resolved == 8477962
    assert method == "exact_history_name"


def test_projection_rows_use_compatible_goalie_scores_and_derived_poolpro():
    projections = pd.DataFrame(
        [
            {
                "source": "espn",
                "entity_id": "nhl:1",
                "nhl_player_id": 1,
                "canonical_name": "Goalie One",
                "category": "G",
                "team_code": "AAA",
                "points": pd.NA,
                "goalie_components_points": 75,
            },
            {
                "source": "hashtag",
                "entity_id": "nhl:2",
                "nhl_player_id": 2,
                "canonical_name": "Goalie Two",
                "category": "G",
                "team_code": "BBB",
                "points": pd.NA,
                "goalie_components_points": pd.NA,
            },
        ]
    )
    derived = pd.DataFrame(
        [
            {
                "source": "local_poolpro",
                "entity_id": "nhl:2",
                "nhl_player_id": 2,
                "canonical_name": "Goalie Two",
                "category": "G",
                "team_code": "BBB",
                "projection_value": 64,
            }
        ]
    )

    rows = _projection_rows(projections, derived)

    assert rows[["source", "projection_value"]].to_records(index=False).tolist() == [
        ("espn", 75.0),
        ("poolpro", 64.0),
    ]
