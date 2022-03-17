from pricing_models.trinomial_tree import topm


def test_tomp_call():
    opt_price = topm(100, 110, 0.1666, 0.025, 0.2, 60, 'C')
    assert opt_price == 0.3123935238660019


def test_tomp_put():
    opt_price = topm(100, 90, 0.1666, 0.025, 0.2, 60, 'P')
    assert opt_price == 0.32069787875063316