from nhl_draft_lab.draft.engine import snake_order


def test_snake_order_three_gms_three_rounds():
    assert snake_order(3, 3) == [1, 2, 3, 3, 2, 1, 1, 2, 3]
