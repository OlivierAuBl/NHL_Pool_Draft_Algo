import sqlite3

import pandas as pd

from nhl_draft_lab.data.repository import load_history, write_rankings


def test_write_rankings_adds_new_feature_columns_to_existing_database(tmp_path):
    db_path = tmp_path / "history.sqlite"
    old = pd.DataFrame([{
        "season_id": 20232024,
        "category": "F",
        "entity_id": 1,
        "name": "Old row",
        "games_played": 82,
    }])
    write_rankings(db_path, old)

    new = old.assign(
        season_id=20242025,
        name="New row",
        games_started=0,
        power_play_points=20,
    )
    write_rankings(db_path, new)

    history = load_history(db_path)
    assert set(history["season_id"]) == {20232024, 20242025}
    assert "games_started" in history
    assert history.loc[history["season_id"] == 20242025, "power_play_points"].iloc[0] == 20

    with sqlite3.connect(db_path) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(rankings)")}
    assert {"games_started", "power_play_points"} <= columns
