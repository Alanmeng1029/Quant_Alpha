from scripts.produce_raw_candidate_factors import FACTOR_SET, INDEXES


def test_raw_candidate_daily_contract_is_fixed():
    assert FACTOR_SET == "o2o_raw_daily60_minute45_csi300_csi500_csi1000_v2"
    assert INDEXES == ("000300.SH", "000905.SH", "000852.SH")
