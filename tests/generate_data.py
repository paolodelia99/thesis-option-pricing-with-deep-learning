from pricing_models.generate_data import mc_option_data_geo


def test_geo_mc_data():
    mc_geo_option_data = mc_option_data_geo(100, 30, 'C')
    assert len(mc_geo_option_data) == 10000
