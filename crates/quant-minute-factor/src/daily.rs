//! Per-stock-day factor math.
//!
//! `compute_daily` derives everything that only needs this trade date's bars,
//! including the five smart-money S series.  `finalize` adds the rolling-window
//! factors (smart Q, 20-day baselines) on top of `state::RollingState`.
//!
//! All functions are deterministic and free of interior mutability so the same
//! inputs always produce bit-identical outputs regardless of thread count.
use crate::loader::Bar;
use crate::schema::*;

fn finite(value: f64) -> Option<f64> {
    if value.is_finite() { Some(value) } else { None }
}

/// Log returns.  `r[0] = ln(close_0/open_0)` captures the opening auction bar;
/// the rest are consecutive close-to-close log returns within the same session,
/// so the lunch break (minute 120 -> 121) is intraday and nothing crosses
/// overnight.
pub fn log_returns(bars: &[Bar; SESSION_BARS]) -> [f64; SESSION_BARS] {
    let mut returns = [0.0_f64; SESSION_BARS];
    returns[0] = (bars[0].close / bars[0].open).ln();
    for t in 1..SESSION_BARS {
        returns[t] = (bars[t].close / bars[t - 1].close).ln();
    }
    returns
}

/// Everything computable from a single day's bars.
pub struct StockDayRaw {
    /// Smart-money S = smart_vwap/day_vwap per `SMART_VARIANTS` entry.
    pub s: [Option<f64>; SMART_VARIANTS.len()],
    pub tail_share: Option<f64>,
    /// Per-bar amount shares; feeds the w20 curve RMSE baseline.
    pub curve: Option<Box<[f64; SESSION_BARS]>>,
    /// 5-minute bucket amount shares; feeds the bucket excess baseline.
    pub buckets: Option<Box<[f64; BUCKETS_5MIN]>>,
    /// Baseline-free factor values keyed by the `F_*` slot constants.
    pub values: [Option<f64>; N_FACTORS],
}

/// Sort bars once per smart-money key definition: descending sort key, ties
/// broken by ascending minute_index so the order is a deterministic total
/// order.  Variants that share a `sort_id` reuse the same order.
fn smart_order(
    bars: &[Bar; SESSION_BARS],
    returns: &[f64; SESSION_BARS],
    variant: &SmartVariant,
) -> Vec<usize> {
    let mut order: Vec<(usize, f64)> = Vec::with_capacity(SESSION_BARS);
    for (t, bar) in bars.iter().enumerate() {
        if bar.volume_share <= 0.0 {
            continue;
        }
        let magnitude = returns[t].abs();
        let key = match variant.key {
            SmartKey::PowerBeta => magnitude / bar.volume_share.powf(variant.beta),
            SmartKey::LogVolume => magnitude / (1.0 + bar.volume_share).ln(),
        };
        if key.is_finite() {
            order.push((t, key));
        }
    }
    order.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| bars[left.0].minute_index.cmp(&bars[right.0].minute_index))
    });
    order.into_iter().map(|(t, _)| t).collect()
}

/// Walk a sorted bar order until cumulative volume reaches the fraction of the
/// day's volume, including the crossing bar, then compute S = smart VWAP over
/// day VWAP.
fn smart_prefix_vwap(
    order: &[usize],
    bars: &[Bar; SESSION_BARS],
    volume_fraction: f64,
) -> Option<f64> {
    let day_volume: f64 = bars.iter().map(|bar| bar.volume_share).sum();
    if !(day_volume > 0.0) || order.is_empty() {
        return None;
    }
    let threshold = day_volume * volume_fraction;
    let mut cumulative = 0.0_f64;
    let mut amount = 0.0_f64;
    let mut volume = 0.0_f64;
    for &t in order {
        let bar = &bars[t];
        cumulative += bar.volume_share;
        amount += bar.amount;
        volume += bar.volume_share;
        if cumulative >= threshold {
            break;
        }
    }
    if volume <= 0.0 {
        return None;
    }
    Some((amount / volume) / (day_amount_total(bars) / day_volume))
}

fn day_amount_total(bars: &[Bar; SESSION_BARS]) -> f64 {
    bars.iter().map(|bar| bar.amount).sum()
}

fn pearson(pairs: &[(f64, f64)]) -> Option<f64> {
    let n = pairs.len() as f64;
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if variance_x <= 0.0 || variance_y <= 0.0 {
        return None;
    }
    finite(covariance / (variance_x * variance_y).sqrt())
}

/// All single-day computations.  Bars must be validated (see `loader`).
pub fn compute_daily(bars: &[Bar; SESSION_BARS]) -> StockDayRaw {
    let returns = log_returns(bars);
    let day_amount = day_amount_total(bars);
    let day_volume: f64 = bars.iter().map(|bar| bar.volume_share).sum();
    let day_vwap = if day_volume > 0.0 {
        Some(day_amount / day_volume)
    } else {
        None
    };

    let mut s: [Option<f64>; SMART_VARIANTS.len()] = [None; SMART_VARIANTS.len()];
    {
        // Bars are sorted once per key definition; variants sharing a sort_id
        // (b025 p15/p20) reuse that order and differ only in the volume
        // threshold at which the crossing bar is absorbed.
        let mut orders: Vec<Option<Vec<usize>>> =
            (0..SMART_VARIANTS.iter().map(|v| v.sort_id).max().unwrap_or(0) + 1)
                .map(|_| None)
                .collect();
        for (slot, variant) in SMART_VARIANTS.iter().enumerate() {
            let order =
                orders[variant.sort_id].get_or_insert_with(|| smart_order(bars, &returns, variant));
            s[slot] = smart_prefix_vwap(order, bars, variant.volume_fraction);
        }
    }

    let mut values: [Option<f64>; N_FACTORS] = [None; N_FACTORS];

    if let (Some(vwap), last_close) = (day_vwap, bars[SESSION_BARS - 1].close) {
        values[F_CLOSE_TO_VWAP] = finite(last_close / vwap - 1.0);
    }

    values[F_TAIL30_RETURN] = {
        let base = bars[SESSION_BARS - 1 - TAIL_BARS].close;
        finite((bars[SESSION_BARS - 1].close / base).ln())
    };

    if day_amount > 0.0 {
        let tail_amount: f64 = bars[SESSION_BARS - TAIL_BARS..]
            .iter()
            .map(|bar| bar.amount)
            .sum();
        values[F_TAIL30_AMOUNT_SHARE] = finite(tail_amount / day_amount);
        if let Some(vwap) = day_vwap {
            let tail_volume: f64 = bars[SESSION_BARS - TAIL_BARS..]
                .iter()
                .map(|bar| bar.volume_share)
                .sum();
            if tail_volume > 0.0 {
                values[F_TAIL30_VWAP_TO_DAY] = finite((tail_amount / tail_volume) / vwap - 1.0);
            }
        }
    }

    {
        let am = (bars[MORNING_LAST].close / bars[0].close).ln();
        let pm = (bars[SESSION_BARS - 1].close / bars[MORNING_LAST].close).ln();
        values[F_PM_MINUS_AM] = finite(pm - am);
    }

    {
        let mut sum_sq = 0.0_f64;
        let mut down_sq = 0.0_f64;
        for &r in &returns[1..] {
            sum_sq += r * r;
            if r < 0.0 {
                down_sq += r * r;
            }
        }
        values[F_REALIZED_VOL] = finite(sum_sq.sqrt());
        if sum_sq > 0.0 {
            values[F_SEMIVAR_RATIO] = finite(down_sq / sum_sq);
        }
    }

    {
        let n = (SESSION_BARS - 1) as f64;
        let mean = returns[1..].iter().sum::<f64>() / n;
        let (mut m2, mut m3, mut m4) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &r in &returns[1..] {
            let d = r - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        m2 /= n;
        m3 /= n;
        m4 /= n;
        if m2 > 1e-300 {
            values[F_REALIZED_SKEW] = finite(m3 / m2.powf(1.5));
            values[F_REALIZED_KURT] = finite(m4 / (m2 * m2));
        }
    }

    {
        let mut path = 0.0_f64;
        for t in 1..SESSION_BARS {
            path += (bars[t].close - bars[t - 1].close).abs();
        }
        if path > 0.0 {
            let net = (bars[SESSION_BARS - 1].close - bars[0].close).abs();
            values[F_TREND_EFFICIENCY] = finite(net / path);
        }
    }

    {
        let mut peak = bars[0].close;
        let mut worst = 0.0_f64;
        for bar in bars.iter() {
            peak = peak.max(bar.close);
            worst = worst.min(bar.close / peak - 1.0);
        }
        values[F_MAX_DRAWDOWN] = finite(worst);
    }

    if day_amount > 0.0 {
        let mut hhi = 0.0_f64;
        let mut entropy = 0.0_f64;
        let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(SESSION_BARS);
        for (t, bar) in bars.iter().enumerate() {
            if bar.amount > 0.0 {
                let share = bar.amount / day_amount;
                hhi += share * share;
                entropy -= share * share.ln();
                ranked.push((t, bar.amount));
            }
        }
        values[F_AMOUNT_HHI] = finite(hhi);
        values[F_AMOUNT_ENTROPY] = finite(entropy / (SESSION_BARS as f64).ln());
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        let top10: f64 = ranked.iter().take(10).map(|(_, amount)| amount).sum();
        values[F_TOP10_SHARE] = finite(top10 / day_amount);
    }

    if day_volume > 0.0 {
        let absolute: f64 = returns[1..].iter().map(|r| r.abs()).sum();
        values[F_PRICE_IMPACT_B050] = finite(absolute / day_volume.sqrt());
    }

    {
        let pairs: Vec<(f64, f64)> = returns[1..]
            .iter()
            .zip(bars[1..].iter())
            .map(|(r, bar)| (*r, bar.amount))
            .collect();
        values[F_RETURN_AMOUNT_CORR] = pearson(&pairs);
    }

    let curve = if day_amount > 0.0 {
        let mut shares = Box::new([0.0_f64; SESSION_BARS]);
        for (t, bar) in bars.iter().enumerate() {
            shares[t] = bar.amount / day_amount;
        }
        Some(shares)
    } else {
        None
    };

    let buckets = if day_amount > 0.0 {
        let mut bucket_shares = Box::new([0.0_f64; BUCKETS_5MIN]);
        for (t, bar) in bars.iter().enumerate() {
            bucket_shares[t / 5] += bar.amount;
        }
        for share in bucket_shares.iter_mut() {
            *share /= day_amount;
        }
        Some(bucket_shares)
    } else {
        None
    };

    StockDayRaw {
        s,
        tail_share: values[F_TAIL30_AMOUNT_SHARE],
        curve,
        buckets,
        values,
    }
}

/// Fill the rolling-window factor slots.  `day_index` positions the stock in
/// the market calendar so suspensions leave true holes instead of compressing
/// the windows.
pub fn finalize(
    code: &str,
    day_index: u32,
    raw: &StockDayRaw,
    state: &crate::state::RollingState,
) -> [Option<f64>; N_FACTORS] {
    let mut values = raw.values;
    for (slot, _) in SMART_VARIANTS.iter().enumerate() {
        values[F_SMART_BASE + slot] = raw.s[slot]
            .and_then(|s_today| state.smart_window_mean(code, slot, day_index, s_today))
            .and_then(|mean| finite(raw.s[slot].unwrap() / mean));
    }
    values[F_TAIL30_Z20] = raw
        .tail_share
        .and_then(|x| state.tail_z20(code, day_index, x))
        .and_then(finite);
    values[F_CURVE_RMSE_W20] = raw
        .curve
        .as_deref()
        .and_then(|curve| state.curve_baseline_rmse(code, day_index, curve))
        .and_then(finite);
    values[F_FIVE_MIN_EXCESS_W20] = raw
        .buckets
        .as_deref()
        .and_then(|buckets| state.bucket_baseline_excess(code, day_index, buckets))
        .and_then(finite);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_bars(price: f64, volume: i64, amount: f64) -> Box<[Bar; SESSION_BARS]> {
        Box::new(std::array::from_fn(|t| Bar {
            minute_index: t as u8,
            open: price,
            high: price,
            low: price,
            close: price,
            volume_share: volume as f64,
            amount,
        }))
    }

    #[test]
    fn first_return_uses_open_of_first_bar_and_lunch_is_intraday() {
        let mut bars = constant_bars(10.0, 100, 1_000.0);
        bars[0].open = 9.0;
        bars[0].close = 9.9; // r_0 = ln(1.1)
        bars[119].close = 10.0;
        bars[120].close = 11.0; // am ends +10%
        bars[121].close = 12.0;
        bars[240].close = 13.2; // pm +10%
        let returns = log_returns(&bars);
        assert!((returns[0] - (9.9_f64 / 9.0).ln()).abs() < 1e-12);
        assert!((returns[1] - (10.0_f64 / 9.9).ln()).abs() < 1e-12);
        let raw = compute_daily(&bars);
        let pm_minus_am = raw.values[F_PM_MINUS_AM].unwrap();
        // am = ln(11/9.9) after the close path... verify against direct formula
        let am = (11.0_f64 / 9.9).ln();
        let pm = (13.2_f64 / 11.0).ln();
        assert!((pm_minus_am - (pm - am)).abs() < 1e-12);
    }

    #[test]
    fn zero_volume_bars_are_excluded_from_vwap_and_smart_sort() {
        let mut bars = constant_bars(10.0, 0, 0.0);
        for (t, bar) in bars.iter_mut().enumerate() {
            if (10..20).contains(&t) {
                bar.volume_share = 100.0;
                bar.amount = 1_000.0 + t as f64;
                bar.close = 10.0 + (t as f64 - 10.0) * 0.1;
                bar.open = bar.close;
            }
        }
        let raw = compute_daily(&bars);
        // day vwap comes only from the 10 traded bars
        let day_amount: f64 = bars.iter().map(|bar| bar.amount).sum();
        let vwap = day_amount / (100.0 * 10.0);
        let close_to_vwap = raw.values[F_CLOSE_TO_VWAP].unwrap();
        assert!((close_to_vwap - (bars[240].close / vwap - 1.0)).abs() < 1e-12);
        // smart S must exist with only 10 traded bars
        assert!(raw.s.iter().all(|s| s.is_some()));
    }

    #[test]
    fn smart_threshold_includes_crossing_bar_with_stable_ties() {
        // 3 bars with volumes 60/30/10, threshold 20% of 100 = 20:
        // sorted desc by |r|/volume^0.25.  Give bar 2 (volume 10) the largest
        // return so it sorts first; crossing bar must be included whole.
        let mut bars = constant_bars(10.0, 0, 0.0);
        bars[0].volume_share = 60.0;
        bars[0].amount = 600.0;
        bars[1].volume_share = 30.0;
        bars[1].amount = 300.0;
        bars[2].volume_share = 10.0;
        bars[2].amount = 105.0;
        bars[0].close = 10.0;
        bars[1].close = 10.01;
        bars[2].close = 10.0 + 0.5; // biggest move
        bars[0].open = 10.0;
        bars[1].open = 10.0;
        bars[2].open = 10.0;
        let returns = log_returns(&bars);
        let variant = &SMART_VARIANTS[1]; // b025 p20
        let order = smart_order(&bars, &returns, variant);
        // Sorted order is bar2, bar1, bar0; the 20% threshold of 100 shares is
        // crossed only after absorbing bar 1 (10 + 30 = 40 >= 20).
        let s = smart_prefix_vwap(&order, &bars, 0.20).unwrap();
        let expected = ((105.0 + 300.0) / (10.0 + 30.0)) / (1_005.0 / 100.0);
        assert!((s - expected).abs() < 1e-12);
        // The 15% threshold of 100 shares still needs bar 1 (10 < 15).
        let s15 = smart_prefix_vwap(&order, &bars, 0.15).unwrap();
        assert!((s15 - expected).abs() < 1e-12);
    }

    #[test]
    fn smart_ties_break_by_minute_index_ascending() {
        let mut bars = constant_bars(10.0, 0, 0.0);
        // bars 0 and 1 have identical positive sort keys (same |r|, same
        // volume): r_0 = ln(10.1/10) and r_1 = ln(10.201/10.1) are both ln(1.01).
        bars[0].volume_share = 15.0;
        bars[0].amount = 150.0;
        bars[0].open = 10.0;
        bars[0].close = 10.1;
        bars[1].volume_share = 15.0;
        bars[1].amount = 160.0;
        bars[1].close = 10.1 * 1.01;
        bars[2].volume_share = 70.0;
        bars[2].amount = 700.0;
        bars[2].close = bars[1].close; // zero return, sorts last
        let returns = log_returns(&bars);
        let order = smart_order(&bars, &returns, &SMART_VARIANTS[1]);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
        // threshold 20% of 100 = 20: bar 0 alone (15) does not cross,
        // bar 0 + bar 1 (30) does; the tie order guarantees both included.
        let s = smart_prefix_vwap(&order, &bars, 0.20).unwrap();
        let expected = (150.0 + 160.0) / 30.0 / (1_010.0 / 100.0);
        assert!((s - expected).abs() < 1e-12);
    }

    #[test]
    fn moments_entropy_hhi_and_top10_on_synthetic_day() {
        let mut bars = constant_bars(10.0, 100, 1_000.0);
        let high = 10.1_f64;
        let low = 9.9_f64;
        for (t, bar) in bars.iter_mut().enumerate() {
            bar.close = if t % 2 == 0 { high } else { low };
        }
        bars[0].open = high;
        let raw = compute_daily(&bars);
        // every move is exactly +/- ln(low/high)
        let r = (low / high).ln();
        assert!((raw.values[F_REALIZED_VOL].unwrap() - (240.0_f64 * r * r).sqrt()).abs() < 1e-9);
        assert!((raw.values[F_SEMIVAR_RATIO].unwrap() - 0.5).abs() < 1e-12);
        assert!(raw.values[F_REALIZED_SKEW].unwrap().abs() < 1e-9);
        assert!((raw.values[F_REALIZED_KURT].unwrap() - 1.0).abs() < 1e-9);
        // uniform amounts: hhi = 241*(1/241)^2, entropy normalized = 1
        assert!((raw.values[F_AMOUNT_HHI].unwrap() - 1.0_f64 / 241.0).abs() < 1e-12);
        assert!((raw.values[F_AMOUNT_ENTROPY].unwrap() - 1.0).abs() < 1e-12);
        // top10 = 10/241
        assert!((raw.values[F_TOP10_SHARE].unwrap() - 10.0 / 241.0).abs() < 1e-12);
        // flat-ish trend: efficiency = |net|/path
        let net = (bars[240].close - bars[0].close).abs();
        let path: f64 = (1..SESSION_BARS)
            .map(|t| (bars[t].close - bars[t - 1].close).abs())
            .sum();
        assert!((raw.values[F_TREND_EFFICIENCY].unwrap() - net / path).abs() < 1e-12);
        // drawdown of the alternating path is exactly one down-leg
        let worst_leg = low / high - 1.0;
        assert!((raw.values[F_MAX_DRAWDOWN].unwrap() - worst_leg).abs() < 1e-9);
        // returns and amounts uncorrelated by construction (r alternates, amount fixed)
        assert!(raw.values[F_RETURN_AMOUNT_CORR].is_none());
        assert!(raw.values[F_PRICE_IMPACT_B050].is_some());
        assert!(raw.curve.is_some());
        assert!(raw.buckets.is_some());
    }

    #[test]
    fn five_minute_buckets_cover_all_241_bars() {
        let bars = constant_bars(10.0, 100, 241.0);
        let raw = compute_daily(&bars);
        let buckets = raw.buckets.unwrap();
        let total: f64 = buckets.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((buckets[0] - 5.0 / 241.0).abs() < 1e-12);
        assert!((buckets[48] - 1.0 / 241.0).abs() < 1e-12);
    }
}
