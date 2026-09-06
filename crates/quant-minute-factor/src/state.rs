//! Calendar-aligned rolling state for the smart-money and 20-day baselines.
//!
//! Every observation is stamped with its market-calendar position (`day_index`).
//! A window over `[d-N, d-1]` reads only slots whose stamp falls inside the
//! range, so a stock suspended for part of the window simply contributes fewer
//! observations - the window itself never shifts or compresses.
use std::collections::HashMap;

use crate::daily::StockDayRaw;
use crate::schema::{
    BASELINE_DAYS, BUCKETS_5MIN, MIN_BASELINE_OBS, MIN_SMART_PRIOR_OBS, SESSION_BARS,
    SMART_VARIANTS,
};

#[derive(Debug)]
struct Stamped<T> {
    day: u32,
    value: T,
}

/// Fixed-capacity ring where slot `day % N` holds the observation written for
/// that calendar position.
#[derive(Debug)]
struct Ring<T> {
    slots: Vec<Option<Stamped<T>>>,
}

impl<T> Ring<T> {
    fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| None).collect(),
        }
    }

    fn write(&mut self, day: u32, value: T) {
        let capacity = self.slots.len();
        self.slots[day as usize % capacity] = Some(Stamped { day, value });
    }

    fn in_window(&self, day: u32, lookback: usize) -> impl Iterator<Item = &Stamped<T>> {
        let lower = day.saturating_sub(lookback as u32);
        self.slots.iter().filter_map(move |slot| {
            slot.as_ref()
                .filter(|stamped| stamped.day >= lower && stamped.day < day)
        })
    }
}

#[derive(Debug)]
struct StockState {
    /// One ring per smart sort series, holding the prior 9 days' S values.
    smart: Vec<Ring<Option<f64>>>,
    tail_share: Ring<Option<f64>>,
    curve: Ring<Option<Box<[f64; SESSION_BARS]>>>,
    buckets: Ring<Option<Box<[f64; BUCKETS_5MIN]>>>,
}

impl StockState {
    fn new() -> Self {
        Self {
            smart: (0..SMART_VARIANTS.len())
                .map(|_| Ring::new(SMART_WINDOW_LOOKBACK))
                .collect(),
            tail_share: Ring::new(BASELINE_DAYS),
            curve: Ring::new(BASELINE_DAYS),
            buckets: Ring::new(BASELINE_DAYS),
        }
    }
}

/// Prior-day depth of the smart-money window (signal day excluded here; it is
/// passed to the mean directly).
const SMART_WINDOW_LOOKBACK: usize = 9;

#[derive(Debug)]
pub struct RollingState {
    stocks: HashMap<String, StockState>,
}

impl RollingState {
    pub fn new() -> Self {
        Self {
            stocks: HashMap::new(),
        }
    }

    /// Record one stock-day's rolling inputs.  Call after the day's factors
    /// are finalized; never for suspended (absent) stocks - the missing day
    /// must stay a hole.
    pub fn advance(&mut self, day: u32, code: &str, raw: &StockDayRaw) {
        let state = self
            .stocks
            .entry(code.to_string())
            .or_insert_with(StockState::new);
        for (slot, s) in raw.s.iter().enumerate() {
            state.smart[slot].write(day, *s);
        }
        state.tail_share.write(day, raw.tail_share);
        if let Some(curve) = &raw.curve {
            state.curve.write(day, Some(curve.clone()));
        }
        if let Some(buckets) = &raw.buckets {
            state.buckets.write(day, Some(buckets.clone()));
        }
    }

    /// Mean of one smart-money S series over `[d-9, d]` including the signal
    /// day.  Requires at least `MIN_SMART_PRIOR_OBS` observed prior days.
    pub fn smart_window_mean(&self, code: &str, slot: usize, day: u32, today: f64) -> Option<f64> {
        let state = self.stocks.get(code)?;
        let ring = state.smart.get(slot)?;
        let mut sum = today;
        let mut count = 1_usize;
        for stamped in ring.in_window(day, SMART_WINDOW_LOOKBACK) {
            if let Some(value) = stamped.value {
                sum += value;
                count += 1;
            }
        }
        if count - 1 < MIN_SMART_PRIOR_OBS {
            return None;
        }
        let mean = sum / count as f64;
        (mean.is_finite() && mean > 0.0).then_some(mean)
    }

    fn scalar_baseline(&self, code: &str, day: u32) -> Option<(f64, f64)> {
        let state = self.stocks.get(code)?;
        let values: Vec<f64> = state
            .tail_share
            .in_window(day, BASELINE_DAYS)
            .filter_map(|stamped| stamped.value)
            .collect();
        moments(&values).filter(|_| values.len() >= MIN_BASELINE_OBS)
    }

    /// `(x - mean)/std` of the tail-30 amount share against its 20-day
    /// baseline (sample std, strictly excluding the signal day).
    pub fn tail_z20(&self, code: &str, day: u32, today: f64) -> Option<f64> {
        let (mean, std) = self.scalar_baseline(code, day)?;
        Some((today - mean) / std)
    }

    /// RMSE of today's 241-bar amount-share curve against the per-bar mean
    /// curve over `[d-20, d-1]`.
    pub fn curve_baseline_rmse(
        &self,
        code: &str,
        day: u32,
        today: &[f64; SESSION_BARS],
    ) -> Option<f64> {
        let state = self.stocks.get(code)?;
        let mut sum = vec![0.0_f64; SESSION_BARS];
        let mut count = 0_usize;
        for stamped in state.curve.in_window(day, BASELINE_DAYS) {
            for (accumulator, value) in sum
                .iter_mut()
                .zip(stamped.value.iter().flat_map(|window| window.iter()))
            {
                *accumulator += value;
            }
            count += 1;
        }
        if count < MIN_BASELINE_OBS {
            return None;
        }
        let count = count as f64;
        let mut squared = 0.0_f64;
        for (today_share, baseline_sum) in today.iter().zip(sum.iter()) {
            let deviation = today_share - baseline_sum / count;
            squared += deviation * deviation;
        }
        Some((squared / SESSION_BARS as f64).sqrt())
    }

    /// Largest 5-minute bucket share gap against the per-bucket baseline mean.
    pub fn bucket_baseline_excess(
        &self,
        code: &str,
        day: u32,
        today: &[f64; BUCKETS_5MIN],
    ) -> Option<f64> {
        let state = self.stocks.get(code)?;
        let mut sum = vec![0.0_f64; BUCKETS_5MIN];
        let mut count = 0_usize;
        for stamped in state.buckets.in_window(day, BASELINE_DAYS) {
            for (accumulator, value) in sum
                .iter_mut()
                .zip(stamped.value.iter().flat_map(|window| window.iter()))
            {
                *accumulator += value;
            }
            count += 1;
        }
        if count < MIN_BASELINE_OBS {
            return None;
        }
        let count = count as f64;
        today
            .iter()
            .zip(sum.iter())
            .map(|(today_share, baseline_sum)| today_share - baseline_sum / count)
            .fold(None::<f64>, |acc, gap| {
                Some(acc.map_or(gap, |best: f64| best.max(gap)))
            })
    }
}

/// `(mean, sample_std)`; `None` when the variance is degenerate.
fn moments(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    if !variance.is_finite() || variance <= 1e-24 {
        return None;
    }
    Some((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_with_s(s: Option<f64>) -> StockDayRaw {
        StockDayRaw {
            s: std::array::from_fn(|_| s),
            tail_share: s,
            curve: None,
            buckets: None,
            values: [None; crate::schema::N_FACTORS],
        }
    }

    #[test]
    fn suspended_days_leave_holes_instead_of_compressing_windows() {
        let mut state = RollingState::new();
        // observations at days 0..4 only, then silence until day 30
        for day in 0..5_u32 {
            state.advance(day, "X", &raw_with_s(Some(1.0)));
        }
        state.advance(30, "X", &raw_with_s(Some(1.0)));
        // window for day 31 is [22, 30] on the calendar: only day 30 observed.
        let mean = state.smart_window_mean("X", 0, 31, 2.0);
        assert!(mean.is_none(), "1 prior observation is below the minimum");
        // a stock observed every day reaches the threshold
        let mut dense = RollingState::new();
        for day in 0..9_u32 {
            dense.advance(day, "Y", &raw_with_s(Some(1.0)));
        }
        let mean = dense.smart_window_mean("Y", 2, 9, 2.0).unwrap();
        assert!((mean - (2.0 + 9.0) / 10.0).abs() < 1e-12);
        // distinct S series stay distinct: slot 4 saw a different history
        let mut split = RollingState::new();
        for day in 0..9_u32 {
            let mut raw = raw_with_s(Some(1.0));
            raw.s[4] = Some(3.0);
            split.advance(day, "W", &raw);
        }
        let slot4 = split.smart_window_mean("W", 4, 9, 3.0).unwrap();
        let slot0 = split.smart_window_mean("W", 0, 9, 9.0).unwrap();
        assert!((slot4 - (3.0 + 9.0 * 3.0) / 10.0).abs() < 1e-12);
        assert!((slot0 - (9.0 + 9.0) / 10.0).abs() < 1e-12);
    }

    #[test]
    fn ring_holds_exactly_the_baseline_window_when_read_before_write() {
        // Production order: day d's baselines are read before day d is written,
        // so the ring still holds [d-20, d-1] at read time.
        let mut state = RollingState::new();
        for day in 0..24_u32 {
            state.advance(day, "Z", &raw_with_s(Some(day as f64)));
        }
        // Ring slots iterate in slot order, not day order: compare as a set.
        let mut values = state.stocks["Z"]
            .tail_share
            .in_window(24, BASELINE_DAYS)
            .filter_map(|stamped| stamped.value)
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values.len(), 20);
        assert_eq!(values[0], 4.0);
        assert_eq!(*values.last().unwrap(), 23.0);

        // Writing day 24 evicts day 4; the window for day 25 slides forward.
        state.advance(24, "Z", &raw_with_s(Some(24.0)));
        let mut values = state.stocks["Z"]
            .tail_share
            .in_window(25, BASELINE_DAYS)
            .filter_map(|stamped| stamped.value)
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values.len(), 20);
        assert_eq!(values[0], 5.0);
        assert_eq!(*values.last().unwrap(), 24.0);
    }

    #[test]
    fn null_days_are_observed_but_not_counted() {
        // days 5..8 are observed with null S (no volume); day 9 is not yet
        // written because baselines for day 9 are read before it advances.
        let mut state = RollingState::new();
        for day in 0..9_u32 {
            state.advance(
                day,
                "A",
                &raw_with_s(if day < 5 { Some(1.0) } else { None }),
            );
        }
        // 5 of the 9 prior days carry a value: >= MIN_SMART_PRIOR_OBS(5)
        let mean = state.smart_window_mean("A", 1, 9, 1.5).unwrap();
        assert!((mean - (1.5 + 5.0) / 6.0).abs() < 1e-12);
        // one fewer observation fails the threshold
        let mut state2 = RollingState::new();
        for day in 0..9_u32 {
            state2.advance(
                day,
                "B",
                &raw_with_s(if day < 4 { Some(1.0) } else { None }),
            );
        }
        assert!(state2.smart_window_mean("B", 1, 9, 1.5).is_none());
    }

    #[test]
    fn tail_z20_uses_sample_std_excluding_signal_day() {
        let mut state = RollingState::new();
        for day in 0..20_u32 {
            state.advance(day, "A", &raw_with_s(Some(day as f64)));
        }
        // baseline for day 20 = values 0..19
        let mean = 9.5;
        let variance: f64 = (0..20).map(|v| (v as f64 - mean).powi(2)).sum::<f64>() / 19.0;
        let z = state.tail_z20("A", 20, 20.0).unwrap();
        assert!((z - (20.0 - mean) / variance.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn curve_rmse_and_bucket_excess_math() {
        let mut state = RollingState::new();
        for day in 0..20_u32 {
            let mut raw = raw_with_s(Some(1.0));
            let mut curve = Box::new([0.0_f64; SESSION_BARS]);
            curve[0] = 0.5;
            curve[1] = 0.5;
            raw.curve = Some(curve);
            let mut buckets = Box::new([0.0_f64; BUCKETS_5MIN]);
            buckets[3] = 1.0;
            raw.buckets = Some(buckets);
            state.advance(day, "A", &raw);
        }
        let mut today = Box::new([0.0_f64; SESSION_BARS]);
        today[0] = 0.6;
        today[1] = 0.4;
        let rmse = state.curve_baseline_rmse("A", 20, &today).unwrap();
        let expected = ((0.1_f64.powi(2) + 0.1_f64.powi(2)) / SESSION_BARS as f64).sqrt();
        assert!((rmse - expected).abs() < 1e-12);

        let mut today_buckets = Box::new([0.0_f64; BUCKETS_5MIN]);
        today_buckets[7] = 0.9;
        let excess = state
            .bucket_baseline_excess("A", 20, &today_buckets)
            .unwrap();
        assert!((excess - 0.9).abs() < 1e-12);
    }
}
