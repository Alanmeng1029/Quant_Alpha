//! Core24 factor registry: names, parameters, and formula documentation.
//!
//! Every formula is stated here once and mirrored into the root manifest so the
//! produced dataset is self-describing.  All factors are close-after-session
//! values: they only use minute bars of their own trade date plus strictly older
//! calendar days for rolling baselines.
use serde::{Deserialize, Serialize};

pub const SESSION_BARS: usize = 241;
pub const MORNING_LAST: usize = 120; // minute_index of the 11:30 bar
pub const TAIL_BARS: usize = 30; // last 30 minutes: index 211..=240
pub const SMART_WINDOW_DAYS: usize = 10; // includes the signal day
pub const BASELINE_DAYS: usize = 20; // strictly excludes the signal day
pub const BUCKETS_5MIN: usize = 49; // (0..=240)/5

/// Minimum number of stocked observations inside a rolling window before the
/// baseline-dependent factor is emitted instead of null.  Windows are aligned
/// to the market calendar, so a stock suspended for part of the window simply
/// contributes fewer observations; the window itself never shifts.
pub const MIN_SMART_PRIOR_OBS: usize = 5; // of 9 prior days
pub const MIN_BASELINE_OBS: usize = 10; // of 20 prior days

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SmartKey {
    /// |r| / volume^beta
    PowerBeta,
    /// |r| / ln(1 + volume)
    LogVolume,
}

/// The five smart-money S series; b025 p15/p20 share one sort order, so the
/// per-day sort state is keyed by `sort_id`.
#[derive(Clone, Copy, Debug)]
pub struct SmartVariant {
    pub name: &'static str,
    pub sort_id: usize,
    pub beta: f64,
    pub key: SmartKey,
    pub volume_fraction: f64,
}

pub const SMART_VARIANTS: [SmartVariant; 5] = [
    SmartVariant {
        name: "mf_smart_q_b010_w10_p20",
        sort_id: 0,
        beta: 0.10,
        key: SmartKey::PowerBeta,
        volume_fraction: 0.20,
    },
    SmartVariant {
        name: "mf_smart_q_b025_w10_p20",
        sort_id: 1,
        beta: 0.25,
        key: SmartKey::PowerBeta,
        volume_fraction: 0.20,
    },
    SmartVariant {
        name: "mf_smart_q_b050_w10_p20",
        sort_id: 2,
        beta: 0.50,
        key: SmartKey::PowerBeta,
        volume_fraction: 0.20,
    },
    SmartVariant {
        name: "mf_smart_q_logv_w10_p20",
        sort_id: 3,
        beta: 0.50,
        key: SmartKey::LogVolume,
        volume_fraction: 0.20,
    },
    SmartVariant {
        name: "mf_smart_q_b025_w10_p15",
        sort_id: 1,
        beta: 0.25,
        key: SmartKey::PowerBeta,
        volume_fraction: 0.15,
    },
];

/// Column order of the wide output.  Index into `[Option<f64>; N_FACTORS]`.
pub const FACTOR_NAMES: [&str; 24] = [
    "mf_smart_q_b010_w10_p20",
    "mf_smart_q_b025_w10_p20",
    "mf_smart_q_b050_w10_p20",
    "mf_smart_q_logv_w10_p20",
    "mf_smart_q_b025_w10_p15",
    "mf_close_to_vwap",
    "mf_tail30_return",
    "mf_tail30_amount_share",
    "mf_tail30_vwap_to_day_vwap",
    "mf_pm_minus_am_return",
    "mf_realized_volatility",
    "mf_downside_semivariance_ratio",
    "mf_realized_skewness",
    "mf_realized_kurtosis",
    "mf_trend_efficiency",
    "mf_max_intraday_drawdown",
    "mf_amount_hhi",
    "mf_amount_entropy",
    "mf_top10bar_amount_share",
    "mf_tail30_amount_z20",
    "mf_amount_curve_rmse_w20",
    "mf_price_impact_b050",
    "mf_return_amount_corr",
    "mf_five_minute_amount_excess_w20",
];

pub const N_FACTORS: usize = FACTOR_NAMES.len();

// Slot layout inside the per-stock-day value array.
pub const F_SMART_BASE: usize = 0; // slots 0..5
pub const F_CLOSE_TO_VWAP: usize = 5;
pub const F_TAIL30_RETURN: usize = 6;
pub const F_TAIL30_AMOUNT_SHARE: usize = 7;
pub const F_TAIL30_VWAP_TO_DAY: usize = 8;
pub const F_PM_MINUS_AM: usize = 9;
pub const F_REALIZED_VOL: usize = 10;
pub const F_SEMIVAR_RATIO: usize = 11;
pub const F_REALIZED_SKEW: usize = 12;
pub const F_REALIZED_KURT: usize = 13;
pub const F_TREND_EFFICIENCY: usize = 14;
pub const F_MAX_DRAWDOWN: usize = 15;
pub const F_AMOUNT_HHI: usize = 16;
pub const F_AMOUNT_ENTROPY: usize = 17;
pub const F_TOP10_SHARE: usize = 18;
pub const F_TAIL30_Z20: usize = 19;
pub const F_CURVE_RMSE_W20: usize = 20;
pub const F_PRICE_IMPACT_B050: usize = 21;
pub const F_RETURN_AMOUNT_CORR: usize = 22;
pub const F_FIVE_MIN_EXCESS_W20: usize = 23;

#[derive(Serialize, Deserialize, Clone)]
pub struct FactorFormula {
    pub name: &'static str,
    pub formula: String,
}

/// Formula strings embedded into the root manifest verbatim.
pub fn formulas() -> Vec<FactorFormula> {
    vec![
        FactorFormula { name: FACTOR_NAMES[0], formula: "Q=S_d/mean(S over [d-9,d] calendar window incl. signal day, >=5 of 9 prior days observed); S=smart_vwap/day_vwap; smart set = bars sorted by |r|/volume^0.10 desc (tie: minute_index asc) until cumulative volume >= 20% of day volume, crossing bar included; zero-volume bars excluded; r_0=ln(close_0/open_0), r_t=ln(close_t/close_{t-1})".into() },
        FactorFormula { name: FACTOR_NAMES[1], formula: "as mf_smart_q_b010_w10_p20 with beta=0.25".into() },
        FactorFormula { name: FACTOR_NAMES[2], formula: "as mf_smart_q_b010_w10_p20 with beta=0.50".into() },
        FactorFormula { name: FACTOR_NAMES[3], formula: "as mf_smart_q_b010_w10_p20 with sort key |r|/ln(1+volume_share)".into() },
        FactorFormula { name: FACTOR_NAMES[4], formula: "as mf_smart_q_b025_w10_p20 with 15% volume threshold; shares the beta=0.25 sort order (same sort_id)".into() },
        FactorFormula { name: FACTOR_NAMES[5], formula: "close_240/day_vwap - 1; day_vwap = sum(amount_cny)/sum(volume_share) over all bars".into() },
        FactorFormula { name: FACTOR_NAMES[6], formula: "ln(close_240/close_210); intraday only, never crosses overnight".into() },
        FactorFormula { name: FACTOR_NAMES[7], formula: "sum(amount, minute 211..240)/sum(amount, whole day)".into() },
        FactorFormula { name: FACTOR_NAMES[8], formula: "tail30_vwap/day_vwap - 1; tail30_vwap = sum(amount, minute 211..240)/sum(volume_share, minute 211..240)".into() },
        FactorFormula { name: FACTOR_NAMES[9], formula: "ln(close_240/close_120) - ln(close_120/close_0); minute 120 is the 11:30 bar, lunch break is intraday".into() },
        FactorFormula { name: FACTOR_NAMES[10], formula: "sqrt(sum(r_t^2, t=1..240))".into() },
        FactorFormula { name: FACTOR_NAMES[11], formula: "sum(min(r_t,0)^2)/sum(r_t^2)".into() },
        FactorFormula { name: FACTOR_NAMES[12], formula: "centered sample skewness of r_1..r_240 = m3/m2^1.5".into() },
        FactorFormula { name: FACTOR_NAMES[13], formula: "centered sample kurtosis of r_1..r_240 = m4/m2^2 (Pearson, not excess)".into() },
        FactorFormula { name: FACTOR_NAMES[14], formula: "|close_240-close_0|/sum(|close_t-close_{t-1}|, t=1..240) (Kaufman efficiency on the close path)".into() },
        FactorFormula { name: FACTOR_NAMES[15], formula: "min_t(close_t/running_max(close_0..close_t)-1) over all 241 close points; always <= 0".into() },
        FactorFormula { name: FACTOR_NAMES[16], formula: "sum(s_t^2) with s_t = amount_t/day_amount over all 241 bars".into() },
        FactorFormula { name: FACTOR_NAMES[17], formula: "-sum(s_t*ln(s_t), s_t>0)/ln(241), Shannon entropy of bar amount shares normalized to [0,1]".into() },
        FactorFormula { name: FACTOR_NAMES[18], formula: "sum of the 10 largest bar amounts (ties: lower minute_index wins)/day_amount".into() },
        FactorFormula { name: FACTOR_NAMES[19], formula: "(tail30_amount_share_d - mean_baseline)/sample_std_baseline; baseline over calendar [d-20,d-1], >=10 observed days, std>1e-12".into() },
        FactorFormula { name: FACTOR_NAMES[20], formula: "sqrt(mean_j((share_d,j - mean_baseline_j)^2)) over 241 per-bar amount shares; baseline per bar over calendar [d-20,d-1], >=10 observed days".into() },
        FactorFormula { name: FACTOR_NAMES[21], formula: "sum(|r_t|, t=1..240)/sqrt(day volume_share); beta=0.5 price impact in volume units".into() },
        FactorFormula { name: FACTOR_NAMES[22], formula: "Pearson corr(r_t, amount_t), t=1..240; null when either variance is 0".into() },
        FactorFormula { name: FACTOR_NAMES[23], formula: "max_b(bucket_share_d,b - mean_baseline_b) over 49 five-minute buckets (minute_index/5); baseline per bucket over calendar [d-20,d-1], >=10 observed days".into() },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_variant_names_align_with_factor_slots_and_shared_sort() {
        for (index, variant) in SMART_VARIANTS.iter().enumerate() {
            assert_eq!(FACTOR_NAMES[F_SMART_BASE + index], variant.name);
        }
        // b025 p15/p20 must share one sort order; every other variant owns one.
        assert_eq!(SMART_VARIANTS[1].sort_id, SMART_VARIANTS[4].sort_id);
        assert_ne!(SMART_VARIANTS[0].sort_id, SMART_VARIANTS[1].sort_id);
        assert_ne!(SMART_VARIANTS[2].sort_id, SMART_VARIANTS[3].sort_id);
    }

    #[test]
    fn factor_names_are_unique() {
        let mut names = FACTOR_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), N_FACTORS);
    }
}
