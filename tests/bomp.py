from pricing_models.bomp import bomp


def test_bomp_call():
    option_price = bomp(38, 40, 0.25, 0.025, 0.2, 15, "C")
    assert option_price == 0.8502384349728811


def test_bomp_put():
    option_price = bomp(45, 40, 0.25, 0.025, 0.25, 30, "P")
    assert option_price == 0.45419070657825167
