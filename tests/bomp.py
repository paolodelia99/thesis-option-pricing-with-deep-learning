from pricing_models.bomp import bomp


def test_bomp_call():
    option_matrix = bomp(38, 40, 0.25, 0.025, 0.2, 15, "C")
    assert option_matrix[15, 0] == 15.97384311294811
    assert option_matrix[15, 15] == 0.0
    assert option_matrix[0, 0] == 0.8502384349728811


def test_bomp_put():
    option_matrix = bomp(45, 40, 0.25, 0.025, 0.25, 30, "P")
    assert option_matrix[0, 30] == 0.0
    assert option_matrix[30, 30] == 17.308071401978
    assert option_matrix[18, 18] == 10.0595043276781
