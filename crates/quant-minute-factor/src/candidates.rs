//! OHLCV-only candidate factor registry and causal daily producer.
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::{
    daily, loader,
    manifest::{self, DayStatus, Manifest},
    pipeline,
    schema::{FactorFormula, SESSION_BARS},
    writer::{self, DynamicWideRow},
    BuildArgs,
};

pub const N: usize = 45;
pub const NAMES: [&str; N] = [
    "mf_cpv_price_volume_level_w20",
    "mf_cpv_dprice_dvolume_pp_w20",
    "mf_cpv_dprice_dvolume_pm_w20",
    "mf_cpv_dprice_dvolume_mp_w20",
    "mf_cpv_dprice_dvolume_mm_w20",
    "mf_cpv_dvolume_lead_dprice_pp_w20",
    "mf_cpv_dvolume_lead_dprice_pm_w20",
    "mf_cpv_dvolume_lead_dprice_mp_w20",
    "mf_cpv_dvolume_lead_dprice_mm_w20",
    "mf_volume_up_return_std",
    "mf_volume_up_positive_return_std",
    "mf_cpv_segment_std_w20",
    "mf_cpv_segment_last30_w20",
    "mf_high_price_vol_ratio_w20",
    "mf_low_price_vol_ratio_w20",
    "mf_smart_q_pooled_b025_w10_p20",
    "mf_minute_amount_seasonality_return_corr_w20",
    "mf_amount_share_skewness",
    "mf_amount_share_kurtosis",
    "mf_return_volume_corr_w20",
    "mf_return_amount_corr_w20",
    "mf_price_dvolume_corr_w20",
    "mf_return_damount_corr_w20",
    "mf_amihud_intraday_w20",
    "mf_amount_periodicity_peak_share_w20",
    "mf_amount_periodicity_band_power_w20",
    "mf_chip_return_bin_std",
    "mf_chip_return_bin_skewness",
    "mf_chip_return_bin_kurtosis",
    "mf_chip_top3_return_bin_share",
    "mf_chip_return_q80",
    "mf_segment_return_dispersion",
    "mf_segment_return_lag1_autocorr",
    "mf_open30_amount_share",
    "mf_open30_tail30_amount_ratio",
    "mf_am_pm_amount_ratio",
    "mf_open30_amount_z20",
    "mf_tail30_amount_to_segment_median",
    "mf_tail30_open30_rv_ratio",
    "mf_pm_am_rv_ratio",
    "mf_tail30_rv_z20",
    "mf_tail30_open30_amihud_ratio",
    "mf_tail30_day_amihud_ratio",
    "mf_tail30_minus_open30_return",
    "mf_tail30_minus_open30_ra_corr",
];
const W20: [usize; 18] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 19, 20, 21, 22, 23,
];
const EPS: f64 = 1e-12;

#[derive(Clone)]
struct Raw {
    values: [Option<f64>; N],
    shares: Option<Box<[f64; SESSION_BARS]>>,
    bars: Box<[loader::Bar; SESSION_BARS]>,
}
#[derive(Default)]
struct State {
    stocks: HashMap<String, VecDeque<Raw>>,
}

fn finite(x: f64) -> Option<f64> {
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}
fn pearson(v: &[(f64, f64)]) -> Option<f64> {
    if v.len() < 3 {
        return None;
    };
    let n = v.len() as f64;
    let (mx, my) = (
        v.iter().map(|x| x.0).sum::<f64>() / n,
        v.iter().map(|x| x.1).sum::<f64>() / n,
    );
    let (mut a, mut b, mut c) = (0., 0., 0.);
    for (x, y) in v {
        let (dx, dy) = (x - mx, y - my);
        a += dx * dy;
        b += dx * dx;
        c += dy * dy;
    }
    if b > 0. && c > 0. {
        finite(a / (b * c).sqrt())
    } else {
        None
    }
}
fn moments(v: &[f64]) -> (Option<f64>, Option<f64>) {
    if v.len() < 3 {
        return (None, None);
    }
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let (mut m2, mut m3, mut m4) = (0., 0., 0.);
    for x in v {
        let d = x - m;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    if m2 > EPS {
        (finite(m3 / m2.powf(1.5)), finite(m4 / (m2 * m2)))
    } else {
        (None, None)
    }
}
fn log_ratio(a: f64, b: f64) -> Option<f64> {
    if a >= 0. && b >= 0. {
        finite(((a + EPS) / (b + EPS)).ln())
    } else {
        None
    }
}
fn seg_range(s: usize) -> std::ops::RangeInclusive<usize> {
    match s {
        0 => 0..=29,
        1 => 30..=59,
        2 => 60..=89,
        3 => 90..=120,
        4 => 121..=150,
        5 => 151..=180,
        6 => 181..=210,
        _ => 211..=240,
    }
}
fn avg(v: impl Iterator<Item = f64>) -> Option<f64> {
    let x: Vec<f64> = v.filter(|x| x.is_finite()).collect();
    if x.is_empty() {
        None
    } else {
        Some(x.iter().sum::<f64>() / x.len() as f64)
    }
}

fn raw(bars: Box<[loader::Bar; SESSION_BARS]>) -> Raw {
    let r = daily::log_returns(&bars);
    let amount: f64 = bars.iter().map(|b| b.amount).sum();
    let volume: f64 = bars.iter().map(|b| b.volume_share).sum();
    let mut x = [None; N];
    let mut dp = Vec::new();
    let mut dv = Vec::new();
    for t in 1..SESSION_BARS {
        dp.push(bars[t].close - bars[t - 1].close);
        dv.push(bars[t].volume_share - bars[t - 1].volume_share);
    }
    x[0] = pearson(
        &(0..SESSION_BARS)
            .map(|t| (bars[t].close, bars[t].volume_share))
            .collect::<Vec<_>>(),
    );
    for q in 0..4 {
        let pairs: Vec<_> = (0..dp.len())
            .filter(|&t| ((dp[t] >= 0.) as usize) * 2 + ((dv[t] >= 0.) as usize) == q)
            .map(|t| (dp[t], dv[t]))
            .collect();
        x[1 + q] = pearson(&pairs);
    }
    for q in 0..4 {
        let pairs: Vec<_> = (0..SESSION_BARS - 2)
            .filter(|&t| ((dv[t] >= 0.) as usize) * 2 + ((dp[t + 1] >= 0.) as usize) == q)
            .map(|t| (dv[t], dp[t + 1]))
            .collect();
        x[5 + q] = pearson(&pairs);
    }
    let vm = volume / SESSION_BARS as f64;
    let vs = (bars
        .iter()
        .map(|b| (b.volume_share - vm).powi(2))
        .sum::<f64>()
        / SESSION_BARS as f64)
        .sqrt();
    let up: Vec<f64> = (1..SESSION_BARS)
        .filter(|&t| bars[t].volume_share > vm + vs)
        .map(|t| r[t])
        .collect();
    let pos: Vec<f64> = up.iter().copied().filter(|z| *z > 0.).collect();
    x[9] = avg(up.iter().map(|z| {
        let m = up.iter().sum::<f64>() / up.len().max(1) as f64;
        (z - m).powi(2)
    }))
    .map(|z| -z.sqrt());
    x[10] = avg(pos.iter().map(|z| {
        let m = pos.iter().sum::<f64>() / pos.len().max(1) as f64;
        (z - m).powi(2)
    }))
    .map(|z| -z.sqrt());
    let mut sc = [None; 8];
    for s in 0..8 {
        let q = seg_range(s);
        sc[s] = pearson(
            &q.clone()
                .filter(|&t| t > 0)
                .map(|t| (r[t], bars[t].amount))
                .collect::<Vec<_>>(),
        );
    }
    let ss: Vec<f64> = sc.iter().flatten().copied().collect();
    if !ss.is_empty() {
        let m = ss.iter().sum::<f64>() / ss.len() as f64;
        x[11] = finite((ss.iter().map(|z| (z - m).powi(2)).sum::<f64>() / ss.len() as f64).sqrt());
    }
    x[12] = sc[7];
    let rv5: Vec<f64> = (0..SESSION_BARS)
        .map(|t| {
            let a = t.saturating_sub(4);
            (a..=t).map(|j| r[j] * r[j]).sum::<f64>().sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..SESSION_BARS).collect();
    order.sort_by(|a, b| bars[*a].close.partial_cmp(&bars[*b].close).unwrap());
    let k = 49;
    let all = rv5.iter().sum::<f64>() / SESSION_BARS as f64;
    x[13] = finite(order.iter().rev().take(k).map(|&i| rv5[i]).sum::<f64>() / k as f64 / all);
    x[14] = finite(order.iter().take(k).map(|&i| rv5[i]).sum::<f64>() / k as f64 / all);
    x[19] = pearson(
        &(1..SESSION_BARS)
            .map(|t| (r[t], bars[t].volume_share))
            .collect::<Vec<_>>(),
    );
    x[20] = pearson(
        &(1..SESSION_BARS)
            .map(|t| (r[t], bars[t].amount))
            .collect::<Vec<_>>(),
    );
    x[21] = pearson(
        &(1..SESSION_BARS)
            .map(|t| {
                (
                    bars[t].close,
                    bars[t].volume_share - bars[t - 1].volume_share,
                )
            })
            .collect::<Vec<_>>(),
    );
    x[22] = pearson(
        &(1..SESSION_BARS)
            .map(|t| (r[t], bars[t].amount - bars[t - 1].amount))
            .collect::<Vec<_>>(),
    );
    x[23] = avg((1..SESSION_BARS)
        .filter(|&t| bars[t].amount > 0.)
        .map(|t| r[t].abs() / bars[t].amount));
    let sr: Vec<f64> = (0..8)
        .map(|s| {
            let q = seg_range(s);
            (bars[*q.end()].close / bars[*q.start()].close).ln()
        })
        .collect();
    let sm = sr.iter().sum::<f64>() / 8.;
    x[31] = finite((sr.iter().map(|z| (z - sm).powi(2)).sum::<f64>() / 8.).sqrt());
    x[32] = pearson(&(0..7).map(|s| (sr[s], sr[s + 1])).collect::<Vec<_>>());
    let seg_rv: Vec<f64> = (0..8)
        .map(|s| {
            seg_range(s)
                .filter(|&t| t > 0)
                .map(|t| r[t] * r[t])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let seg_am: Vec<f64> = (0..8)
        .map(|s| {
            avg(seg_range(s)
                .filter(|&t| t > 0 && bars[t].amount > 0.)
                .map(|t| r[t].abs() / bars[t].amount))
            .unwrap_or(f64::NAN)
        })
        .collect();
    x[38] = log_ratio(seg_rv[7], seg_rv[0]);
    x[39] = log_ratio(
        (seg_rv[4].powi(2) + seg_rv[5].powi(2) + seg_rv[6].powi(2) + seg_rv[7].powi(2)).sqrt(),
        (seg_rv[0].powi(2) + seg_rv[1].powi(2) + seg_rv[2].powi(2) + seg_rv[3].powi(2)).sqrt(),
    );
    x[41] = log_ratio(seg_am[7], seg_am[0]);
    x[42] = log_ratio(
        seg_am[7],
        avg((1..SESSION_BARS)
            .filter(|&t| bars[t].amount > 0.)
            .map(|t| r[t].abs() / bars[t].amount))
        .unwrap_or(f64::NAN),
    );
    x[43] = finite(sr[7] - sr[0]);
    x[44] = match (sc[7], sc[0]) {
        (Some(a), Some(b)) => finite(a.atanh() - b.atanh()),
        _ => None,
    };
    if amount > 0. {
        let shares = Box::new(std::array::from_fn(|t| bars[t].amount / amount));
        let sv: Vec<f64> = shares.iter().copied().collect();
        let (sk, ku) = moments(&sv);
        x[17] = sk;
        x[18] = ku;
        x[33] = Some(shares[0..30].iter().sum());
        let seg_a: Vec<f64> = (0..8)
            .map(|s| seg_range(s).map(|t| bars[t].amount).sum())
            .collect();
        x[34] = log_ratio(seg_a[0], seg_a[7]);
        x[35] = log_ratio(
            seg_a[0] + seg_a[1] + seg_a[2] + seg_a[3],
            seg_a[4] + seg_a[5] + seg_a[6] + seg_a[7],
        );
        let mut m = seg_a.clone();
        m.sort_by(|a, b| a.partial_cmp(b).unwrap());
        x[37] = log_ratio(seg_a[7], m[3]);
        let bins = 22usize;
        let mut chip = vec![0.; bins];
        for t in 0..SESSION_BARS {
            let cr = (bars[t].close / bars[0].open).ln();
            let bi = ((cr * 100.0).floor() as isize + 11).clamp(0, 21) as usize;
            chip[bi] += bars[t].amount / amount;
        }
        let (csk, cku) = moments(&chip);
        let cm = chip.iter().sum::<f64>() / bins as f64;
        x[26] = Some((chip.iter().map(|z| (z - cm).powi(2)).sum::<f64>() / bins as f64).sqrt());
        x[27] = csk;
        x[28] = cku;
        let mut cp = chip.clone();
        cp.sort_by(|a, b| b.partial_cmp(a).unwrap());
        x[29] = Some(cp.iter().take(3).sum());
        let mut p = 0.;
        for i in 0..bins {
            p += chip[i];
            if p >= 0.8 {
                x[30] = Some((i as f64 - 11.0) * 0.01);
                break;
            }
        }
        return Raw {
            values: x,
            shares: Some(shares),
            bars,
        };
    }
    Raw {
        values: x,
        shares: None,
        bars,
    }
}

fn hist_mean(h: &VecDeque<Raw>, slot: usize) -> Option<f64> {
    let v: Vec<f64> = h.iter().filter_map(|r| r.values[slot]).collect();
    if v.len() < 10 {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}
fn hist_z(h: &VecDeque<Raw>, slot: usize, current: Option<f64>) -> Option<f64> {
    let x = current?;
    let v: Vec<f64> = h.iter().filter_map(|r| r.values[slot]).collect();
    if v.len() < 10 {
        return None;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|z| (z - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt();
    if sd > EPS {
        finite((x - m) / sd)
    } else {
        None
    }
}
fn finalize(mut raw: Raw, h: &VecDeque<Raw>) -> [Option<f64>; N] {
    raw.values[36] = hist_z(h, 33, raw.values[33]);
    raw.values[40] = hist_z(h, 38, raw.values[38]);
    for i in W20 {
        raw.values[i] = hist_mean(h, i);
    } // strictly prior-day rolling means
    if h.len() >= 5 {
        let mut pool = Vec::new();
        for q in h.iter().rev().take(9) {
            pool.extend(q.bars.iter().copied());
        }
        pool.extend(raw.bars.iter().copied());
        pool.retain(|b| b.volume_share > 0.);
        pool.sort_by(|a, b| {
            let ra = (a.close / a.open).ln().abs() / a.volume_share.powf(0.25);
            let rb = (b.close / b.open).ln().abs() / b.volume_share.powf(0.25);
            rb.partial_cmp(&ra).unwrap()
        });
        let tv: f64 = pool.iter().map(|b| b.volume_share).sum();
        let mut v = 0.;
        let mut a = 0.;
        for b in &pool {
            v += b.volume_share;
            a += b.amount;
            if v >= tv * 0.2 {
                break;
            }
        }
        let av: f64 = pool.iter().map(|b| b.amount).sum();
        raw.values[15] = if v > 0. && tv > 0. && av > 0. {
            finite((a / v) / (av / tv))
        } else {
            None
        };
    }
    if let Some(sh) = &raw.shares {
        let profiles: Vec<_> = h.iter().filter_map(|z| z.shares.as_ref()).collect();
        if profiles.len() >= 10 {
            let base: [f64; SESSION_BARS] = std::array::from_fn(|t| {
                profiles.iter().map(|p| p[t]).sum::<f64>() / profiles.len() as f64
            });
            let rr = daily::log_returns(&raw.bars);
            raw.values[16] = pearson(
                &(0..SESSION_BARS)
                    .map(|t| (rr[t], sh[t] / (base[t] + EPS)))
                    .collect::<Vec<_>>(),
            );
            let z: Vec<f64> = (0..SESSION_BARS).map(|t| sh[t] - base[t]).collect();
            let mean = z.iter().sum::<f64>() / z.len() as f64;
            // Pre-registered low/mid-frequency grid.  The minute-volume
            // U-shape has already been removed; restricting to 1..=30
            // avoids spending most production time on noisy Nyquist bins.
            let power: Vec<f64> = (1..=30)
                .map(|k| {
                    let (re, im) = (0..SESSION_BARS).fold((0., 0.), |(a, b), t| {
                        let q =
                            2. * std::f64::consts::PI * k as f64 * t as f64 / SESSION_BARS as f64;
                        (a + (z[t] - mean) * q.cos(), b + (z[t] - mean) * q.sin())
                    });
                    re * re + im * im
                })
                .collect();
            let total = power.iter().sum::<f64>();
            if total > EPS {
                raw.values[24] = finite(power.iter().copied().fold(0., f64::max) / total);
                raw.values[25] = finite(power[4..20].iter().sum::<f64>() / total);
            }
        }
    }
    raw.values
}

pub fn formulas() -> Vec<FactorFormula> {
    NAMES
        .iter()
        .map(|&n| FactorFormula {
            name: n,
            formula: "OHLCV candidate factor; exact causal parameters recorded in source registry"
                .into(),
        })
        .collect()
}
pub fn run(args: BuildArgs) -> Result<PathBuf> {
    if args.start > args.end {
        bail!("--start must not exceed --end")
    };
    if args.threads_per_job == 0 {
        bail!("--threads-per-job must be positive")
    }
    std::fs::create_dir_all(args.output.join("_staging"))?;
    let catalog = args.catalog.canonicalize()?;
    let root = args.minute_root.canonicalize()?;
    let output = args.output;
    let p = manifest::default_parameters(
        args.block_days,
        args.jobs,
        args.threads_per_job,
        args.memory_limit_mb,
    );
    let mut src = BTreeMap::new();
    src.insert("catalog".into(), catalog.display().to_string());
    src.insert("minute_root".into(), root.display().to_string());
    src.insert(
        "factor_inputs".into(),
        "open,close,volume_share,amount_cny".into(),
    );
    let mp = output.join("manifest.json");
    let mut man = if mp.exists() {
        let m: Manifest = serde_json::from_str(&std::fs::read_to_string(&mp)?)?;
        if m.factor_set != "ohlcv_candidates_v1" {
            bail!("incompatible manifest")
        };
        m
    } else {
        Manifest::new_named("ohlcv_candidates_v1", p, src, formulas())
    };
    let ctx = Arc::new(pipeline::load_market_context(
        &catalog,
        &args.start,
        &args.end,
        args.memory_limit_mb,
        &args.index_codes,
        args.raw_eligible_universe,
    )?);
    let blocks = pipeline::plan_blocks(&ctx.calendar, &args.start, &args.end, args.block_days, 20);
    if blocks.is_empty() {
        bail!("no market dates")
    }
    if args.replace {
        for b in &blocks {
            for day in b.target_begin..=b.target_end {
                man.dates.remove(&ctx.calendar[day]);
            }
        }
    }
    let manifest = Arc::new(Mutex::new(man));
    let blocks = Arc::new(blocks);
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..args.jobs {
            let ctx = Arc::clone(&ctx);
            let manifest = Arc::clone(&manifest);
            let blocks = Arc::clone(&blocks);
            let next = Arc::clone(&next);
            let root = root.clone();
            let output = output.clone();
            let failures = &failures;
            scope.spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(args.threads_per_job)
                    .build()
                    .expect("candidate rayon pool");
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= blocks.len() {
                        break;
                    }
                    if let Err(e) = run_block(&pool, &ctx, &manifest, &root, &output, &blocks[i]) {
                        failures.lock().unwrap().push(format!("block {i}: {e:#}"));
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    {
        let guard = manifest.lock().unwrap();
        guard.save(&mp)?;
    }
    if !failures.is_empty() {
        bail!("candidate build failed: {}", failures.join("; "));
    }
    Ok(output)
}

fn candidate_day_file(output: &Path, date: &str) -> PathBuf {
    output
        .join(format!("year={}", &date[..4]))
        .join(format!("{date}.parquet"))
}
fn run_block(
    pool: &rayon::ThreadPool,
    ctx: &pipeline::MarketContext,
    manifest: &Arc<Mutex<Manifest>>,
    root: &Path,
    output: &Path,
    block: &pipeline::Block,
) -> Result<()> {
    let mut state = State::default();
    for di in block.warmup_begin..=block.target_end {
        let date = &ctx.calendar[di];
        let target = di >= block.target_begin;
        let universe_ok = !target || ctx.universe.contains_key(date);
        let done = {
            let m = manifest.lock().unwrap();
            target
                && universe_ok
                && m.date_complete(date) == Some(true)
                && candidate_day_file(output, date).is_file()
        };
        let write = target && universe_ok && !done;
        if let Some(day) = loader::load_day(root, date)? {
            let computed: Vec<(String, Raw)> = pool.install(|| {
                day.stocks
                    .into_par_iter()
                    .map(|(c, b)| (c, raw(b)))
                    .collect()
            });
            let mut rows = Vec::new();
            for (code, r) in computed {
                let h = state.stocks.entry(code.clone()).or_default();
                let values = finalize(r.clone(), h);
                if write && ctx.contains(date, &code) {
                    rows.push(DynamicWideRow {
                        ts_code: code.clone(),
                        values: values.to_vec(),
                    });
                }
                h.push_back(r);
                if h.len() > 20 {
                    h.pop_front();
                }
            }
            if write {
                rows.sort_by(|a, b| a.ts_code.cmp(&b.ts_code));
                writer::write_day_dynamic(output, date, &NAMES, &rows)?;
                let mut m = manifest.lock().unwrap();
                m.record(
                    date,
                    DayStatus {
                        status: "ok".into(),
                        rows: rows.len(),
                        excluded: BTreeMap::new(),
                        elapsed_seconds: 0.,
                    },
                );
                m.save(&output.join("manifest.json"))?;
            }
        } else if write {
            let mut m = manifest.lock().unwrap();
            m.record(
                date,
                DayStatus {
                    status: "missing_partition".into(),
                    rows: 0,
                    excluded: BTreeMap::new(),
                    elapsed_seconds: 0.,
                },
            );
            m.save(&output.join("manifest.json"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registry_has_45_unique_columns() {
        let mut names = NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), N);
        assert_eq!(formulas().len(), N);
    }
    #[test]
    fn segment_boundaries_cover_session_once() {
        let mut seen = [0u8; SESSION_BARS];
        for s in 0..8 {
            for t in seg_range(s) {
                seen[t] += 1;
            }
        }
        assert!(seen.iter().all(|n| *n == 1));
    }
}
