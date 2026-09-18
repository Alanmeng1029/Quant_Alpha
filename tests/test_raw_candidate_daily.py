from scripts.produce_raw_candidate_factors import FACTOR_SET, INDEXES


def test_raw_candidate_daily_contract_is_fixed():
    assert FACTOR_SET == "o2o_raw_daily60_minute45_csi500_csi1000_v1"
    assert INDEXES == ("000905.SH", "000852.SH")
