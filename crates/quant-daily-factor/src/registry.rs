pub const FACTOR_NAMES: [&str; 60] = [
    "gtja_alpha159_raw_v1",
    "gtja_alpha158_raw_v1",
    "wq_alpha088_raw_v1",
    "gtja_alpha140_raw_v1",
    "gtja_alpha042_raw_v1",
    "wq_alpha040_raw_v1",
    "gtja_alpha187_raw_v1",
    "gtja_alpha175_raw_v1",
    "gtja_alpha189_raw_v1",
    "gtja_alpha167_raw_v1",
    "gtja_alpha161_raw_v1",
    "gtja_alpha174_raw_v1",
    "gtja_alpha070_raw_v1",
    "gtja_alpha165_raw_v1",
    "gtja_alpha185_raw_v1",
    "gtja_alpha114_raw_v1",
    "wq_alpha083_raw_v1",
    "gtja_alpha129_raw_v1",
    "gtja_alpha183_raw_v1",
    "gtja_alpha160_raw_v1",
    "gtja_alpha164_raw_v1",
    "gtja_alpha093_raw_v1",
    "gtja_alpha121_raw_v1",
    "gtja_alpha010_raw_v1",
    "gtja_alpha126_raw_v1",
    "gtja_alpha173_raw_v1",
    "gtja_alpha153_raw_v1",
    "gtja_alpha124_raw_v1",
    "wq_alpha041_raw_v1",
    "wq_alpha005_raw_v1",
    "gtja_alpha095_raw_v1",
    "gtja_alpha041_raw_v1",
    "wq_alpha077_raw_v1",
    "gtja_alpha120_raw_v1",
    "wq_alpha042_raw_v1",
    "gtja_alpha091_raw_v1",
    "gtja_alpha108_raw_v1",
    "gtja_alpha127_raw_v1",
    "gtja_alpha098_raw_v1",
    "wq_alpha094_raw_v1",
    "wq_alpha026_raw_v1",
    "wq_alpha057_raw_v1",
    "gtja_alpha005_raw_v1",
    "gtja_alpha150_raw_v1",
    "wq_alpha071_raw_v1",
    "gtja_alpha119_raw_v1",
    "gtja_alpha132_raw_v1",
    "wq_alpha014_raw_v1",
    "wq_alpha006_raw_v1",
    "gtja_alpha136_raw_v1",
    "gtja_alpha139_raw_v1",
    "gtja_alpha148_raw_v1",
    "gtja_alpha076_raw_v1",
    "gtja_alpha154_raw_v1",
    "wq_alpha065_raw_v1",
    "gtja_alpha092_raw_v1",
    "gtja_alpha149_raw_v1",
    "wq_alpha098_raw_v1",
    "wq_alpha051_raw_v1",
    "gtja_alpha040_raw_v1",
];

pub fn parse_ids(contents: &str) -> Vec<&str> {
    contents
        .lines()
        .filter_map(|line| {
            let value = line.split('#').next()?.trim();
            (!value.is_empty()).then_some(value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn production_registry_is_unique_and_complete() {
        let mut values = FACTOR_NAMES.to_vec();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), 60);
    }
}
