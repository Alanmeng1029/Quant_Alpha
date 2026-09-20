//! `dos_minute_v1`: faithful Rust replication of the 31 DolphinDB minute-bar
//! daily factor scripts under `2.分钟K线因子/*.dos`.
//!
//! Column mapping: `LastPx→close`, `OpenPx→open`, `HighPx→high`, `LowPx→low`,
//! `Volume→volume_share`, `Amount→amount` (amount_cny).  Factor 1's
//! `TradeMoney = LastPx*Volume` is reproduced as `close*volume_share`.
//!
//! Time window: the .dos job template filters `time between 09:30 and 14:57`
//! which maps to `minute_index 0..=STD_END(237)`; factors 10/12/19 used
//! `endTime = 15:00` and therefore read the full session (`0..=240`), with
//! 10/19 further restricting the regression sample to rows 5..=239
//! (`having rank(TradeTime) between 5:239`).
//!
//! Cross-day windows replicate DolphinDB row-based `mavg/mstd` semantics
//! (partial windows, no calendar alignment).  Documented deviations from the
//! scripts: (a) `deltas`/`percentChange` that the scripts evaluate without
//! `context by` across stock boundaries (factors 2, 16, 17, 18) are computed
//! per stock-day instead; (b) ties in `aggrTopN`-style argmin/argmax resolve
//! to the earliest bar; see `docs/research/dolphindb_minute_factor_replication.md`.
//!
//! Blocks are single-threaded because factors 5/8/10/29 consume same-day
//! cross-sectional information.
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::{
    BuildArgs, loader,
    manifest::{self, DayStatus, Manifest},
    pipeline,
    schema::{FactorFormula, SESSION_BARS},
    writer::{self, DynamicWideRow},
};

const N: usize = 31;
/// Last bar of the 09:30–14:57 window (minute_index of the 14:57 bar).
const STD_END: usize = 237;
/// Last bar of the full session (15:00), used by factors 10/12/19.
const FULL_END: usize = 240;
/// Factor 10/19 regression sample: `rank(TradeTime) between 5:239` over the
/// full session maps to minute_index 5..=239.
const REG_LO: usize = 5;
const REG_HI: usize = 239;
/// Cross-day window for factors 3/13 (20-day same-minute stats) and 5 (20-day
/// residual mavg).
const W20: usize = 20;
/// Factor 29's per-stock 10-day std window.
const W10: usize = 10;
const EPS: f64 = 1e-12;
const T_DF: f64 = 240.0;
const WARMUP_DAYS: usize = 25;

pub const NAMES: [&str; N] = [
    "mf_dos_illiq_shortcut",
    "mf_dos_positive_consist_volume",
    "mf_dos_corr_ret_lag_adj_amount",
    "mf_dos_vol_tide_ratio",
    "mf_dos_fall_center_dev",
    "mf_dos_consist_volume",
    "mf_dos_vol_prop_entropy",
    "mf_dos_patv",
    "mf_dos_flash_volatility",
    "mf_dos_noon_canopy_alpha",
    "mf_dos_single_vol_prop_entropy",
    "mf_dos_resilience_cov",
    "mf_dos_corr_lag_ret_adj_amount",
    "mf_dos_flash_returns",
    "mf_dos_peak_climbing_cov",
    "mf_dos_corr_ret_amount",
    "mf_dos_corr_ret_lag_amount",
    "mf_dos_corr_lag_ret_amount",
    "mf_dos_dawn_fog_vol_persist",
    "mf_dos_prop_t_dis",
    "mf_dos_prop_normal_dis",
    "mf_dos_prop_naive_act",
    "mf_dos_prop_uniform_dis",
    "mf_dos_volume_peak_count",
    "mf_dos_ratio_fuzziness_amount",
    "mf_dos_ratio_fuzziness_volume",
    "mf_dos_p_dis_vol",
    "mf_dos_b_dis_vol",
    "mf_dos_adj_fuzziness_diff",
    "mf_dos_vsa_ratio",
    "mf_dos_ratio_fuzziness_amt_corr",
];

const I_ILLIQ: usize = 0;
const I_POS_CONSIST: usize = 1;
const I_CORR_RET_LAG_ADJ: usize = 2;
const I_VOL_TIDE: usize = 3;
const I_FALL_CENTER: usize = 4;
const I_CONSIST: usize = 5;
const I_ENTROPY: usize = 6;
const I_PATV: usize = 7;
const I_FLASH_VOL: usize = 8;
const I_NOON: usize = 9;
const I_SINGLE_ENTROPY: usize = 10;
const I_RESILIENCE: usize = 11;
const I_CORR_LAG_RET_ADJ: usize = 12;
const I_FLASH_RET: usize = 13;
const I_PEAK_CLIMB: usize = 14;
const I_CORR_RET_AMT: usize = 15;
const I_CORR_RET_LAG_AMT: usize = 16;
const I_CORR_LAG_RET_AMT: usize = 17;
const I_DAWN_FOG: usize = 18;
const I_PROP_T: usize = 19;
const I_PROP_NORMAL: usize = 20;
const I_PROP_NAIVE: usize = 21;
const I_PROP_UNIFORM: usize = 22;
const I_PEAK_COUNT: usize = 23;
const I_FUZZ_AMT: usize = 24;
const I_FUZZ_VOL: usize = 25;
const I_PDIS: usize = 26;
const I_BDIS: usize = 27;
const I_ADJ_FUZZ: usize = 28;
const I_VSA: usize = 29;
const I_FUZZ_AMT_CORR: usize = 30;

// ---------------------------------------------------------------------------
// Generic statistics (DolphinDB semantics: std = sample, covar = population,
// corr = Pearson dropping null pairs, kurtosis = biased excess m4/m2^2 - 3).
// ---------------------------------------------------------------------------

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn sample_std(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    Some((v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt())
}

fn covar_pop(p: &[(f64, f64)]) -> Option<f64> {
    if p.is_empty() {
        return None;
    }
    let n = p.len() as f64;
    let (mx, my) = (
        p.iter().map(|x| x.0).sum::<f64>() / n,
        p.iter().map(|x| x.1).sum::<f64>() / n,
    );
    finite(p.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / n)
}

fn pearson(p: &[(f64, f64)]) -> Option<f64> {
    if p.len() < 2 {
        return None;
    }
    let n = p.len() as f64;
    let (mx, my) = (
        p.iter().map(|x| x.0).sum::<f64>() / n,
        p.iter().map(|x| x.1).sum::<f64>() / n,
    );
    let (mut cv, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (x, y) in p {
        let (dx, dy) = (x - mx, y - my);
        cv += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    (vx > EPS && vy > EPS).then(|| finite(cv / (vx * vy).sqrt()))?
}

fn kurtosis_excess(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &x in v {
        let d = x - m;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;
    (m2 > EPS).then(|| finite(m4 / (m2 * m2) - 3.0))?
}

/// Wall-clock milliseconds of the bar's time label; bar 120 is 11:30 and bar
/// 121 is 13:01, so the lunch break spans 91 minutes between adjacent bars.
fn time_ms(t: usize) -> i64 {
    let minutes = if t <= 120 { 570 + t } else { 660 + t };
    minutes as i64 * 60_000
}

// ---------------------------------------------------------------------------
// Distributions: normal CDF via A&S 7.1.26 erf (|err| <= 1.5e-7) and Student-t
// CDF via the regularized incomplete beta (Lentz continued fraction).
// ---------------------------------------------------------------------------

fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    s * (1.0 - poly * (-z * z).exp())
}

fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn lgamma(x: f64) -> f64 {
    // Lanczos approximation (g = 7, 9 terms); only used for a, b >= 0.5.
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Lentz continued fraction for the incomplete beta (DLMF 8.17.7).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPSB: f64 = 3e-14;
    const FPMIN: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c: f64 = 1.0;
    let mut d: f64 = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let (m, m2) = (m as f64, (2 * m) as f64);
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -((a + m) * (qab + m) * x) / ((qam + m2 + 1.0) * (a + m2 + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPSB {
            break;
        }
    }
    h
}

fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = lgamma(a + b) - lgamma(a) - lgamma(b);
    let front = (a * x.ln() + b * (1.0 - x).ln() + lbeta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

fn student_t_cdf(df: f64, t: f64) -> f64 {
    let x = df / (df + t * t);
    let ib = inc_beta(df / 2.0, 0.5, x);
    if t > 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
}

// ---------------------------------------------------------------------------
// OLS with intercept: solves the normal equations after rms-scaling each
// regressor (t stats and F are scale invariant) and returns coefficient
// t statistics plus the overall ANOVA F.  Rows with any null y/x are dropped
// (DolphinDB ols behaviour).
// ---------------------------------------------------------------------------

fn gauss_solve(a: Vec<Vec<f64>>, b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut a = a;
    let mut x = b.to_vec();
    let scale = a.iter().flatten().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if !(scale > 0.0) {
        return None;
    }
    for col in 0..n {
        let piv =
            (col..n).max_by(|&u, &v| a[u][col].abs().partial_cmp(&a[v][col].abs()).unwrap())?;
        if a[piv][col].abs() < scale * 1e-13 {
            return None;
        }
        a.swap(col, piv);
        x.swap(col, piv);
        for r in (col + 1)..n {
            let m = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= m * a[col][c];
            }
            x[r] -= m * x[col];
        }
    }
    for r in (0..n).rev() {
        for c in (r + 1)..n {
            x[r] -= a[r][c] * x[c];
        }
        x[r] /= a[r][r];
    }
    Some(x)
}

struct Ols {
    t: Vec<f64>,
    f: f64,
}

fn ols_with_stats(y: &[f64], xcols: &[Vec<f64>]) -> Option<Ols> {
    let (n, k) = (y.len(), xcols.len());
    let p = k + 1;
    if n <= p {
        return None;
    }
    // Rms-scale each regressor for conditioning; t/F are invariant.
    let scaled: Vec<Vec<f64>> = xcols
        .iter()
        .map(|c| {
            let rms = (c.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
            if !(rms > 0.0) {
                return c.clone();
            }
            c.iter().map(|x| x / rms).collect()
        })
        .collect();
    let mut a = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for i in 0..n {
        let mut row = vec![1.0; p];
        for (j, c) in scaled.iter().enumerate() {
            row[j + 1] = c[i];
        }
        for u in 0..p {
            for v in 0..p {
                a[u][v] += row[u] * row[v];
            }
            rhs[u] += row[u] * y[i];
        }
    }
    let beta = gauss_solve(a.clone(), &rhs)?;
    let ybar = y.iter().sum::<f64>() / n as f64;
    let (mut sse, mut ssr) = (0.0, 0.0);
    for i in 0..n {
        let mut yhat = beta[0];
        for (j, c) in scaled.iter().enumerate() {
            yhat += beta[j + 1] * c[i];
        }
        sse += (y[i] - yhat).powi(2);
        ssr += (yhat - ybar).powi(2);
    }
    let dof = (n - p) as f64;
    if !(sse > 1e-18) {
        return None; // perfect fit: t/F are infinite, not representable
    }
    let sigma2 = sse / dof;
    let mut t = Vec::with_capacity(p);
    for j in 0..p {
        let mut e = vec![0.0; p];
        e[j] = 1.0;
        let col = gauss_solve(a.clone(), &e)?;
        let se = (sigma2 * col[j]).sqrt();
        if !(se > 0.0) {
            return None;
        }
        t.push(beta[j] / se);
    }
    let f = (ssr / k as f64) / (sse / dof);
    Some(Ols { t, f })
}

// ---------------------------------------------------------------------------
// Per-factor computations.  All take the full 241-bar day plus the inclusive
// end index of the factor's time window.
// ---------------------------------------------------------------------------

/// 1/6/2: K线形态家族.
fn illiq_shortcut(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> Option<f64> {
    let mut s = 0.0;
    for t in 0..=t_end {
        let trade_money = b[t].close * b[t].volume_share;
        if trade_money > 0.0 {
            let path = 2.0 * (b[t].high - b[t].low) - (b[t].open - b[t].close).abs();
            s += path / trade_money;
        }
    }
    finite(s)
}

/// 6 (require_rising=false) and 2 (true).  `deltas(LastPx)` is grouped per
/// stock-day (the .dos evaluated it across stock boundaries).
fn consist_volume(
    b: &[loader::Bar; SESSION_BARS],
    t_end: usize,
    require_rising: bool,
) -> Option<f64> {
    let mut cond_vol = 0.0;
    let mut hit = false;
    for t in 0..=t_end {
        let consistent = (b[t].close - b[t].open).abs() <= 0.5 * (b[t].high - b[t].low).abs();
        if consistent && (!require_rising || t >= 1 && b[t].close > b[t - 1].close) {
            cond_vol += b[t].volume_share;
            hit = true;
        }
    }
    if !hit {
        return None; // DolphinDB: no resTemp group -> null
    }
    let total: f64 = (0..=t_end).map(|t| b[t].volume_share).sum();
    (total > 0.0).then(|| finite(cond_vol / total))?
}

/// 4: volume tide — centered 9-bar volume neighbourhood, peak, troughs.
fn vol_tide_ratio(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> Option<f64> {
    let n = t_end + 1;
    let mut ps = vec![0.0; n + 1];
    for t in 0..n {
        ps[t + 1] = ps[t] + b[t].volume_share;
    }
    // adj[t] = msum(Volume,9)[t+4] = sum over rows max(0,t-4)..=t+4 (partial at
    // the start); null when t+4 exceeds the window.
    let adj = |t: usize| -> Option<f64> {
        (t + 4 <= t_end).then(|| {
            let hi = (t + 5).min(n);
            let lo = t.saturating_sub(4);
            ps[hi] - ps[lo]
        })
    };
    let tide = (0..n)
        .filter_map(|t| adj(t).map(|a| (t, a)))
        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())?;
    let trough = |range: std::ops::Range<usize>| -> Option<(usize, f64)> {
        range
            .filter(|&t| t < n)
            .filter_map(|t| adj(t).map(|a| (t, a)))
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
    };
    let (rise_num, _) = trough(0..tide.0)?;
    let (fall_num, _) = trough(tide.0 + 1..n)?;
    let (rise_price, fall_price) = (b[rise_num].close, b[fall_num].close);
    let dt = (fall_num - rise_num) as f64;
    (dt != 0.0).then(|| finite(((rise_price / fall_price) - 1.0) / dt))?
}

/// 5: per-stock intraday gravity centers: wavg(t, ret) over ret>0 / ret<0.
fn gravity_centers(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> (Option<f64>, Option<f64>) {
    let (mut un, mut ud, mut dn, mut dd) = (0.0, 0.0, 0.0, 0.0);
    for t in 1..=t_end {
        let ret = (b[t].close - b[t - 1].close) / b[t - 1].close;
        if ret > 0.0 {
            un += t as f64 * ret;
            ud += ret;
        } else if ret < 0.0 {
            dn += t as f64 * ret;
            dd += ret;
        }
    }
    (
        (ud != 0.0).then(|| finite(un / ud)).flatten(),
        (dd != 0.0).then(|| finite(dn / dd)).flatten(),
    )
}

/// Cross-sectional OLS of down_g on up_g (factor 5): per-observation residual.
fn cross_section_residual(pairs: &[(f64, f64)]) -> Option<Vec<f64>> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let (mx, my) = (
        pairs.iter().map(|x| x.0).sum::<f64>() / n,
        pairs.iter().map(|x| x.1).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (x, y) in pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    if !(sxx > EPS) {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    Some(
        pairs
            .iter()
            .map(|(x, y)| y - (intercept + slope * x))
            .collect(),
    )
}

/// 7/11: amount-share entropy; `single` = Volume-share * price-share variant.
fn vol_prop_entropy(b: &[loader::Bar; SESSION_BARS], t_end: usize, single: bool) -> Option<f64> {
    let (mut amt, mut vol, mut px) = (0.0, 0.0, 0.0);
    for t in 0..=t_end {
        amt += b[t].amount;
        vol += b[t].volume_share;
        px += b[t].close;
    }
    if single && !(vol > 0.0 && px > 0.0) {
        return None;
    }
    if !single && !(amt > 0.0) {
        return None;
    }
    let mut h = 0.0;
    for t in 0..=t_end {
        let p = if single {
            (b[t].volume_share / vol) * (b[t].close / px)
        } else {
            b[t].amount / amt
        };
        if p > 0.0 {
            h += -p * p.ln();
        }
    }
    finite(h)
}

/// 8: per-minute Volume / mavg(Volume, 10) (partial windows), null when the
/// moving average is zero.
fn patv_vol_prop(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> Box<[Option<f64>]> {
    let mut out = vec![None; t_end + 1].into_boxed_slice();
    let mut ps = vec![0.0; t_end + 2];
    for t in 0..=t_end {
        ps[t + 1] = ps[t] + b[t].volume_share;
    }
    for t in 0..=t_end {
        let lo = t.saturating_sub(9);
        let m = (ps[t + 1] - ps[lo]) / (t + 1 - lo) as f64;
        if m > 0.0 {
            out[t] = finite(b[t].volume_share / m);
        }
    }
    out
}

/// 9 (volatility=true) / 14: flash family — mean of the chosen series over
/// bars whose volume delta exceeds mean+std of the delta.
fn flash_family(b: &[loader::Bar; SESSION_BARS], t_end: usize, volatility: bool) -> Option<f64> {
    let dvols: Vec<f64> = (1..=t_end)
        .map(|t| b[t].volume_share - b[t - 1].volume_share)
        .collect();
    let thr = mean(&dvols)? + sample_std(&dvols)?;
    let mut vals = Vec::new();
    for t in 1..=t_end {
        if b[t].volume_share - b[t - 1].volume_share > thr {
            if volatility {
                // adjVolatility[t] = mstd(ret,5)[t+4]: forward 5-bar std of
                // returns, valid while t+4 <= t_end.
                if t + 4 <= t_end {
                    let w: Vec<f64> = (t..=t + 4)
                        .map(|u| (b[u].close - b[u - 1].close) / b[u - 1].close)
                        .collect();
                    if let Some(s) = sample_std(&w) {
                        vals.push(s);
                    }
                }
            } else {
                vals.push((b[t].close - b[t - 1].close) / b[t - 1].close);
            }
        }
    }
    mean(&vals).and_then(finite)
}

/// 10/19: regression of ret on [ΔV, ΔV lag1..lag5] over rows 5..=239 of the
/// full session.  Returns (F, |t_intercept|, std of the 5 lagged t stats).
fn canopy_dawn(b: &[loader::Bar; SESSION_BARS]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let mut ys = Vec::new();
    let mut xs: Vec<Vec<f64>> = vec![Vec::new(); 6];
    for t in REG_LO..=REG_HI {
        let dvs: Option<Vec<f64>> = (0..=5)
            .map(|j| (t >= j + 1).then(|| b[t - j].volume_share - b[t - j - 1].volume_share))
            .collect();
        let ret = (t >= 1).then(|| (b[t].close - b[t - 1].close) / b[t - 1].close);
        if let (Some(ret), Some(dvs)) = (ret, dvs) {
            ys.push(ret);
            for (j, d) in dvs.into_iter().enumerate() {
                xs[j].push(d);
            }
        }
    }
    match ols_with_stats(&ys, &xs) {
        None => (None, None, None),
        Some(o) => {
            let dawn = sample_std(&o.t[2..7]).and_then(finite);
            (finite(o.f), finite(o.t[0].abs()), dawn)
        }
    }
}

/// 12 (thresholded=false, full session) / 15: covariance between
/// ret/adjVolatility and adjVolatility, where adjVolatility =
/// (rowStd/rowAvg)^2 over the 20-value OHLC window of the last 5 bars.
fn ohlc_cov(b: &[loader::Bar; SESSION_BARS], t_end: usize, thresholded: bool) -> Option<f64> {
    let mut advs: Vec<(usize, f64)> = Vec::new(); // (t, adjVolatility)
    for t in 4..=t_end {
        let mut v = [0.0; 20];
        for k in 0..5 {
            v[k] = b[t - 4 + k].high;
            v[5 + k] = b[t - 4 + k].open;
            v[10 + k] = b[t - 4 + k].low;
            v[15 + k] = b[t - 4 + k].close;
        }
        let m = mean(&v)?;
        let s = sample_std(&v)?;
        if m > 0.0 {
            advs.push((t, (s / m).powi(2)));
        }
    }
    let thr = if thresholded {
        let a: Vec<f64> = advs.iter().map(|x| x.1).collect();
        mean(&a)? + sample_std(&a)?
    } else {
        f64::NEG_INFINITY
    };
    let mut pairs = Vec::new();
    for (t, av) in advs {
        if av > thr && t >= 1 && av > 0.0 {
            let ret = (b[t].close - b[t - 1].close) / b[t - 1].close;
            pairs.push((ret / av, av));
        }
    }
    covar_pop(&pairs)
}

/// 16/17/18: corr(|log1p(ret)|, Amount) with optional one-bar lags on either
/// side.  Bar 0 has no same-day predecessor (the .dos leaked the previous
/// stock's last bar here); lags shift accordingly.
fn corr_absret_amount(
    b: &[loader::Bar; SESSION_BARS],
    t_end: usize,
    lag_ret: bool,
    lag_amount: bool,
) -> Option<f64> {
    let abs_log_ret =
        |t: usize| -> Option<f64> { (t >= 1).then(|| ((b[t].close / b[t - 1].close).ln()).abs()) };
    let mut pairs = Vec::new();
    for t in 0..=t_end {
        let a = if lag_ret {
            if t == 0 { None } else { abs_log_ret(t - 1) }
        } else {
            abs_log_ret(t)
        };
        let amt = if lag_amount {
            (t >= 1).then(|| b[t - 1].amount)
        } else {
            Some(b[t].amount)
        };
        if let (Some(a), Some(m)) = (a, amt) {
            pairs.push((a, m));
        }
    }
    pearson(&pairs)
}

/// 3/13: corr between |log1p(ret)| (optionally lagged) and the
/// 20-day-same-minute standardized amount (optionally lagged).  Bar 0 pairs
/// with the previous day's last window bar (same stock, repartitioned data).
fn corr_ret_lag_adj(
    b: &[loader::Bar; SESSION_BARS],
    t_end: usize,
    adj: &[Option<f64>],
    last: Option<&LastBar237>,
    lag_ret: bool,
    lag_amount: bool,
) -> Option<f64> {
    let abs_log_ret =
        |t: usize| -> Option<f64> { (t >= 1).then(|| ((b[t].close / b[t - 1].close).ln()).abs()) };
    let mut pairs = Vec::new();
    for t in 0..=t_end {
        let a = if lag_ret {
            match t {
                0 => last.and_then(|l| l.ret_abs_log),
                _ => abs_log_ret(t - 1),
            }
        } else {
            match t {
                0 => last
                    .filter(|l| l.close > 0.0)
                    .map(|l| ((b[0].close / l.close).ln()).abs()),
                _ => abs_log_ret(t),
            }
        };
        let m = if lag_amount {
            match t {
                0 => last.and_then(|l| l.adj_amount),
                _ => adj.get(t - 1).copied().flatten(),
            }
        } else {
            adj.get(t).copied().flatten()
        };
        if let (Some(a), Some(m)) = (a, m) {
            pairs.push((a, m));
        }
    }
    pearson(&pairs)
}

/// 20/21/22/23: active-share family — sum(Amount * w(x)) / sum(Amount) with
/// pluggable weight w over the standardized series x.  The denominator is the
/// full-window amount sum while the numerator only includes bars with a
/// defined weight (bar 0 never has one).
fn prop_active(b: &[loader::Bar; SESSION_BARS], t_end: usize, kind: u8) -> Option<f64> {
    let ret = |t: usize| -> Option<f64> {
        (t >= 1).then(|| (b[t].close - b[t - 1].close) / b[t - 1].close)
    };
    let den: f64 = (0..=t_end).map(|t| b[t].amount).sum();
    if !(den > 0.0) {
        return None;
    }
    let mut num = 0.0;
    match kind {
        0 | 2 => {
            // whole-day standardization: ret/std(ret) or Δclose/std(Δclose)
            let xs: Vec<f64> = if kind == 0 {
                (1..=t_end).filter_map(|t| ret(t)).collect()
            } else {
                (1..=t_end).map(|t| b[t].close - b[t - 1].close).collect()
            };
            let sd = sample_std(&xs)?;
            if !(sd > 0.0) {
                return None;
            }
            for t in 1..=t_end {
                let z = if kind == 0 {
                    ret(t)?
                } else {
                    b[t].close - b[t - 1].close
                };
                num += b[t].amount * student_t_cdf(T_DF, z / sd);
            }
        }
        _ => {
            for t in 1..=t_end {
                let w = match kind {
                    1 => ret(t).map(|r| phi(r * 1.96 / 0.1)),
                    _ => ret(t).map(|r| (r - 0.1) / 0.2),
                };
                if let Some(w) = w {
                    num += b[t].amount * w;
                }
            }
        }
    }
    finite(num / den)
}

/// 24: count of above-threshold volume bars whose gap to the previous such
/// bar exceeds one minute (the lunch break counts: 11:30 -> 13:01 = 91 min).
fn volume_peak_count(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> Option<f64> {
    let vols: Vec<f64> = (0..=t_end).map(|t| b[t].volume_share).collect();
    let thr = mean(&vols)? + sample_std(&vols)?;
    let peaks: Vec<usize> = (0..=t_end).filter(|&t| vols[t] > thr).collect();
    if peaks.is_empty() {
        return None; // DolphinDB: empty group -> no factor row
    }
    let mut count = 0;
    for w in peaks.windows(2) {
        if time_ms(w[1]) - time_ms(w[0]) > 60_000 {
            count += 1;
        }
    }
    finite(count as f64)
}

/// 25/26/29 stage 1: fuzziness = mstd(mstd(ret,5,5),5,5) with full-period
/// minimums; fog bars are those with fuzziness above its day mean.
fn fuzziness_series(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> Vec<Option<f64>> {
    let ret = |t: usize| -> Option<f64> {
        (t >= 1).then(|| (b[t].close - b[t - 1].close) / b[t - 1].close)
    };
    let mut inner = vec![None; t_end + 1];
    for j in 5..=t_end {
        let w: Vec<f64> = (j - 4..=j).filter_map(&ret).collect();
        if w.len() == 5 {
            inner[j] = sample_std(&w);
        }
    }
    let mut fz = vec![None; t_end + 1];
    for k in 9..=t_end {
        let w: Vec<f64> = (k - 4..=k).filter_map(|i| inner[i]).collect();
        if w.len() == 5 {
            fz[k] = sample_std(&w);
        }
    }
    fz
}

#[derive(Clone, Copy, PartialEq)]
enum FuzzKind {
    Amount,
    Volume,
    Diff,
}

fn ratio_fuzz(b: &[loader::Bar; SESSION_BARS], t_end: usize, kind: FuzzKind) -> Option<f64> {
    let fz = fuzziness_series(b, t_end);
    let vals: Vec<f64> = fz.iter().flatten().copied().collect();
    let thr = mean(&vals)?;
    let (mut fog_amt, mut fog_vol, mut fog_n) = (0.0, 0.0, 0.0);
    for t in 0..=t_end {
        if let Some(z) = fz[t] {
            if z > thr {
                fog_amt += b[t].amount;
                fog_vol += b[t].volume_share;
                fog_n += 1.0;
            }
        }
    }
    if fog_n == 0.0 {
        return None; // no fog bars -> no group row
    }
    let avg_amt: f64 = (0..=t_end).map(|t| b[t].amount).sum::<f64>() / (t_end + 1) as f64;
    let avg_vol: f64 = (0..=t_end).map(|t| b[t].volume_share).sum::<f64>() / (t_end + 1) as f64;
    match kind {
        FuzzKind::Amount => (avg_amt > 0.0).then(|| finite((fog_amt / fog_n) / avg_amt))?,
        FuzzKind::Volume => (avg_vol > 0.0).then(|| finite((fog_vol / fog_n) / avg_vol))?,
        FuzzKind::Diff => {
            if !(avg_vol > 0.0 && avg_amt > 0.0) {
                return None;
            }
            finite((fog_vol / fog_n) / avg_vol - (fog_amt / fog_n) / avg_amt)
        }
    }
}

/// 31: corr(fuzziness, Amount) over non-null fuzziness bars.
fn fuzz_amt_corr(b: &[loader::Bar; SESSION_BARS], t_end: usize) -> Option<f64> {
    let fz = fuzziness_series(b, t_end);
    let pairs: Vec<(f64, f64)> = (0..=t_end)
        .filter_map(|t| fz[t].map(|z| (z, b[t].amount)))
        .collect();
    pearson(&pairs)
}

/// 27/28/30: volume-support-area family.  Groups bars by close price, finds
/// the max-volume price (support), accumulates volume shares by proximity to
/// the support until 50% of the day's volume, then returns an area-vs-extreme
/// difference.  Factor 30's "close" is the last price in the price-sorted
/// group table, i.e. the day's highest minute close (replicated quirk).
fn support_value(
    b: &[loader::Bar; SESSION_BARS],
    t_end: usize,
    p_kind: bool,
    vsa: bool,
) -> Option<f64> {
    let mut groups: BTreeMap<u64, (f64, f64)> = BTreeMap::new(); // price -> (sum vol, max high / min low)
    for t in 0..=t_end {
        let key = b[t].close.to_bits();
        let e = groups.entry(key).or_insert((
            0.0,
            if p_kind {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
        ));
        e.0 += b[t].volume_share;
        e.1 = if p_kind {
            e.1.max(b[t].high)
        } else {
            e.1.min(b[t].low)
        };
    }
    let mut rows: Vec<(f64, f64, f64)> = groups
        .into_iter()
        .map(|(k, (vol, ext))| (f64::from_bits(k), vol, ext))
        .collect(); // BTreeMap iterates in bit order == ascending numeric order for positive floats
    let total: f64 = rows.iter().map(|r| r.1).sum();
    if !(total > 0.0) {
        return None;
    }
    let mut support_idx = 0;
    for i in 1..rows.len() {
        if rows[i].1 > rows[support_idx].1 {
            support_idx = i; // ties keep the earliest (lowest) price
        }
    }
    let support = rows[support_idx].0;
    rows.sort_by(|a, c| {
        (a.0 - support)
            .abs()
            .partial_cmp(&(c.0 - support).abs())
            .unwrap()
            .then(c.1.partial_cmp(&a.1).unwrap())
    });
    let mut cum = 0.0;
    let mut vsp = None;
    for (i, r) in rows.iter().enumerate() {
        cum += r.1 / total;
        if cum >= 0.5 {
            vsp = Some(i);
            break;
        }
    }
    let vsp = vsp?;
    let area = &rows[..=vsp];
    let area_min = area.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let area_max = area.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
    let extreme = rows.iter().map(|r| r.2).fold(
        if p_kind {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        },
        if p_kind { f64::max } else { f64::min },
    );
    if vsa {
        // 30: min area price - highest close of the day (last() over the
        // price-sorted groups).
        let day_max_close = rows.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
        finite(area_min - day_max_close)
    } else if p_kind {
        finite(area_min - extreme) // 27: support-area low - day high
    } else {
        finite(area_max - extreme) // 28: support-area high - day low
    }
}

// ---------------------------------------------------------------------------
// State: row-based deques replicating DolphinDB mavg/mstd over the stock's
// own daily series (suspension days leave holes; windows compress).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct LastBar237 {
    close: f64,
    /// Standardized amount at the last window bar, for the next day's lagged
    /// pairing of factor 3.
    adj_amount: Option<f64>,
    /// |log1p(ret)| at the last window bar, for factor 13's lagged return.
    ret_abs_log: Option<f64>,
}

#[derive(Default)]
struct StockHist {
    /// Strictly-prior daily amount curves (all 241 bars) for the 20-day
    /// same-minute statistics of factors 3/13.
    amounts: VecDeque<Box<[f64; SESSION_BARS]>>,
    /// Daily cross-sectional residuals of factor 5 (Option: hole days).
    residuals: VecDeque<Option<f64>>,
    /// Daily stage-1 values of factor 29.
    fuzz: VecDeque<Option<f64>>,
    last: Option<LastBar237>,
}

#[derive(Default)]
struct State {
    stocks: HashMap<String, StockHist>,
}

/// (Amount - mean)/sample-std against the same minute over the up-to-20
/// strictly-prior days (DolphinDB `prev(mavg(Amount,20))` / `prev(mstd)`),
/// with partial windows when history is shorter.
fn adj_amount_series(
    hist: &VecDeque<Box<[f64; SESSION_BARS]>>,
    b: &[loader::Bar; SESSION_BARS],
    t_end: usize,
) -> Vec<Option<f64>> {
    let mut out = vec![None; t_end + 1];
    let k = hist.len();
    if k < 2 {
        return out; // mstd of a single observation is null
    }
    for t in 0..=t_end {
        let xs: Vec<f64> = hist.iter().map(|c| c[t]).collect();
        let m = xs.iter().sum::<f64>() / k as f64;
        if let Some(s) = sample_std(&xs) {
            if s > 0.0 {
                out[t] = finite((b[t].amount - m) / s);
            }
        }
    }
    out
}

/// Everything computed per stock-day before any cross-sectional step.
struct DayInter {
    values: [Option<f64>; N],
    up_g: Option<f64>,
    down_g: Option<f64>,
    f_stat: Option<f64>,
    abs_t0: Option<f64>,
    fuzz_diff: Option<f64>,
    vol_prop: Box<[Option<f64>]>,
    close_237: f64,
    adj_amount_237: Option<f64>,
    ret_237_abs_log: Option<f64>,
    amount_curve: Box<[f64; SESSION_BARS]>,
}

fn compute_stock_day(b: &[loader::Bar; SESSION_BARS], hist: Option<&StockHist>) -> DayInter {
    let mut v = [None; N];
    v[I_ILLIQ] = illiq_shortcut(b, STD_END);
    v[I_POS_CONSIST] = consist_volume(b, STD_END, true);
    v[I_CONSIST] = consist_volume(b, STD_END, false);
    v[I_VOL_TIDE] = vol_tide_ratio(b, STD_END);
    v[I_ENTROPY] = vol_prop_entropy(b, STD_END, false);
    v[I_SINGLE_ENTROPY] = vol_prop_entropy(b, STD_END, true);
    v[I_FLASH_VOL] = flash_family(b, STD_END, true);
    v[I_FLASH_RET] = flash_family(b, STD_END, false);
    v[I_RESILIENCE] = ohlc_cov(b, FULL_END, false);
    v[I_PEAK_CLIMB] = ohlc_cov(b, STD_END, true);
    v[I_PROP_T] = prop_active(b, STD_END, 0);
    v[I_PROP_NORMAL] = prop_active(b, STD_END, 1);
    v[I_PROP_NAIVE] = prop_active(b, STD_END, 2);
    v[I_PROP_UNIFORM] = prop_active(b, STD_END, 3);
    v[I_PEAK_COUNT] = volume_peak_count(b, STD_END);
    v[I_FUZZ_AMT] = ratio_fuzz(b, STD_END, FuzzKind::Amount);
    v[I_FUZZ_VOL] = ratio_fuzz(b, STD_END, FuzzKind::Volume);
    v[I_PDIS] = support_value(b, STD_END, true, false);
    v[I_BDIS] = support_value(b, STD_END, false, false);
    v[I_VSA] = support_value(b, STD_END, true, true);
    v[I_FUZZ_AMT_CORR] = fuzz_amt_corr(b, STD_END);
    v[I_CORR_RET_AMT] = corr_absret_amount(b, STD_END, false, false);
    v[I_CORR_RET_LAG_AMT] = corr_absret_amount(b, STD_END, false, true);
    v[I_CORR_LAG_RET_AMT] = corr_absret_amount(b, STD_END, true, false);
    let (up_g, down_g) = gravity_centers(b, STD_END);
    let (f_stat, abs_t0, dawn) = canopy_dawn(b);
    v[I_DAWN_FOG] = dawn;
    let hist = hist.filter(|h| h.amounts.len() >= 2 || h.last.is_some());
    let (adj, adj_237) = match hist {
        Some(h) if h.amounts.len() >= 2 => {
            let adj = adj_amount_series(&h.amounts, b, STD_END);
            let a = adj[STD_END];
            (adj, a)
        }
        _ => (vec![None; STD_END + 1], None),
    };
    let last = hist.and_then(|h| h.last.as_ref());
    v[I_CORR_RET_LAG_ADJ] = corr_ret_lag_adj(b, STD_END, &adj, last, false, true);
    v[I_CORR_LAG_RET_ADJ] = corr_ret_lag_adj(b, STD_END, &adj, last, true, false);
    let ret_237_abs = ((b[STD_END].close / b[STD_END - 1].close).ln()).abs();
    DayInter {
        values: v,
        up_g,
        down_g,
        f_stat,
        abs_t0,
        fuzz_diff: ratio_fuzz(b, STD_END, FuzzKind::Diff),
        vol_prop: patv_vol_prop(b, STD_END),
        close_237: b[STD_END].close,
        adj_amount_237: adj_237,
        ret_237_abs_log: finite(ret_237_abs),
        amount_curve: Box::new(std::array::from_fn(|t| b[t].amount)),
    }
}

// ---------------------------------------------------------------------------
// Build pipeline (single-threaded blocks, same skeleton as candidates_v3).
// ---------------------------------------------------------------------------

pub fn formulas() -> Vec<FactorFormula> {
    const F: [&str; N] = [
        "dos#1 illiqShortCut: sum((2*(high-low)-abs(open-close))/(close*volume)) over 09:30-14:57",
        "dos#2 positiveConsistVolume: sum(volume | |close-open|<=0.5*|high-low| and close>prev close)/sum(volume); deltas grouped per stock-day (script bug fixed)",
        "dos#3 corrRetLagAdjAmount: corr(|log1p(pctChange(close))|, prev((amount-mean)/std)) vs same-minute 20-day prior stats; bar 0 pairs with previous day's 14:57 bar",
        "dos#4 volTideRatio: ((risePrice/fallPrice)-1)/(fallNum-riseNum) over centered 9-bar volume neighbourhood troughs around the peak",
        "dos#5 fallCenterDev: 20-day mavg of the daily cross-sectional OLS residual of down-centroid on up-centroid (wavg(minute, ret))",
        "dos#6 consistVolume: sum(volume | |close-open|<=0.5*|high-low|)/sum(volume)",
        "dos#7 volPropEntropy: sum(-p*ln p), p = minute amount / day amount",
        "dos#8 PATV: mean/std + kurtosis of the per-minute cross-sectional percent rank (min-tie, (0,1]) of volume/mavg(volume,10)",
        "dos#9 flashVolatility: mean of forward 5-bar return std over bars with deltas(volume) > mean+std",
        "dos#10 NoonCanopyAlpha: sign by F > cross-sectional mean(F) times |t| of intercept in ols(ret ~ [dVol, dVol lag1..5]) over bars 5..239 of the full session",
        "dos#11 singleVolPropEntropy: sum(-p*ln p), p = (volume share)*(close share)",
        "dos#12 resilienceCov: covar(ret/adjVol, adjVol), adjVol=(std/mean)^2 of the 20-value OHLC window (last 5 bars), full session",
        "dos#13 corrLagRetAdjAmount: corr(prev(|log1p(pctChange(close))|), (amount-mean)/std) vs same-minute 20-day prior stats",
        "dos#14 flashReturns: mean(ret) over bars with deltas(volume) > mean+std",
        "dos#15 peakClimbingCov: covar(ret/adjVol, adjVol) restricted to adjVol > mean+std of adjVol",
        "dos#16 corrRetAmount: corr(|log1p(pctChange(close))|, amount)",
        "dos#17 corrRetLagAmount: corr(|log1p(pctChange(close))|, prev(amount)); bar 0 null (script bug fixed)",
        "dos#18 corrLagRetAmount: corr(prev(|log1p(pctChange(close))|), amount); bar 0 null (script bug fixed)",
        "dos#19 DawnFogVolPersist: std of the t stats of the 5 lagged dVolume regressors in ols(ret ~ [dVol, dVol lag1..5]) over bars 5..239 of the full session",
        "dos#20 propTDis: sum(amount*cdfStudent(240, ret/std(ret)))/sum(amount)",
        "dos#21 propNormalDis: sum(amount*Phi(ret*1.96/0.1))/sum(amount)",
        "dos#22 propNaiveAct: sum(amount*cdfStudent(240, (close-prev close)/std(close diff)))/sum(amount)",
        "dos#23 propUniformDis: sum(amount*((ret-0.1)/0.2))/sum(amount), unclamped as in the script",
        "dos#24 volumePeakCount: count of above-(mean+std)-volume bars whose gap to the previous such bar exceeds 1 minute (lunch break counts)",
        "dos#25 ratioFuzzinessAmount: mean(amount | fuzziness>avg fuzziness)/avg(amount), fuzziness=mstd(mstd(ret,5,5),5,5)",
        "dos#26 ratioFuzzinessVolume: mean(volume | fuzziness>avg fuzziness)/avg(volume)",
        "dos#27 pDisVol: min close in the 50%-volume support area around the max-volume price - day high",
        "dos#28 bDisVol: max close in the support area - day low",
        "dos#29 adjFuzzinessDiff: (vol ratio - amount ratio on fog bars), negatives scaled by the 10-day std then by s1/s2 cross-sectionally",
        "dos#30 vsaRatio: min close in the support area - highest close (script's last() over price-sorted groups)",
        "dos#31 ratioFuzzinessAmtCorr: corr(fuzziness, amount)",
    ];
    NAMES
        .iter()
        .zip(F.iter())
        .map(|(n, f)| FactorFormula {
            name: n,
            formula: f.to_string(),
        })
        .collect()
}

pub fn run(args: BuildArgs) -> Result<PathBuf> {
    if args.start > args.end {
        bail!("--start must not exceed --end");
    }
    if args.threads_per_job != 1 {
        bail!(
            "dos_minute_v1 requires --threads-per-job 1 because each block is serial and cross-sectional"
        );
    }
    if args.jobs == 0 || args.block_days == 0 {
        bail!("jobs and block-days must be positive");
    }
    std::fs::create_dir_all(args.output.join("_staging"))?;
    let catalog = args.catalog.canonicalize()?;
    let root = args.minute_root.canonicalize()?;
    let output = args.output;
    let ctx = Arc::new(pipeline::load_market_context(
        &catalog,
        "1900-01-01",
        &args.end,
        args.memory_limit_mb,
        &args.index_codes,
        args.raw_eligible_universe,
    )?);
    let blocks = Arc::new(pipeline::plan_blocks(
        &ctx.calendar,
        &args.start,
        &args.end,
        args.block_days,
        WARMUP_DAYS,
    ));
    if blocks.is_empty() {
        bail!("no market dates");
    }
    let mut src = BTreeMap::new();
    src.insert("catalog".into(), catalog.display().to_string());
    src.insert("minute_root".into(), root.display().to_string());
    src.insert(
        "universe".into(),
        format!("dynamic {}", args.index_codes.join(" union ")),
    );
    src.insert(
        "source_scripts".into(),
        "2.分钟K线因子/*.dos (DolphinDB), see docs/research/dolphindb_minute_factor_replication.md"
            .into(),
    );
    let mp = output.join("manifest.json");
    let mut p = manifest::default_parameters(args.block_days, args.jobs, 1, args.memory_limit_mb);
    p.warmup_days = WARMUP_DAYS;
    let mut m = if mp.exists() {
        let mut x: Manifest = serde_json::from_str(&std::fs::read_to_string(&mp)?)?;
        if x.factor_set != "dos_minute_v1" {
            bail!("incompatible manifest");
        }
        x.formulas = formulas();
        x
    } else {
        Manifest::new_named("dos_minute_v1", p, src, formulas())
    };
    if args.replace {
        for b in blocks.iter() {
            for d in b.target_begin..=b.target_end {
                m.dates.remove(&ctx.calendar[d]);
            }
        }
    }
    let manifest = Arc::new(Mutex::new(m));
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..args.jobs {
            let ctx = ctx.clone();
            let blocks = blocks.clone();
            let manifest = manifest.clone();
            let next = next.clone();
            let root = root.clone();
            let output = output.clone();
            let failures = &failures;
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= blocks.len() {
                        break;
                    }
                    if let Err(e) = run_block(&ctx, &manifest, &root, &output, &blocks[i]) {
                        failures.lock().unwrap().push(format!("block {i}: {e:#}"));
                        break;
                    }
                }
            });
        }
    });
    let f = failures.into_inner().unwrap();
    {
        let g = manifest.lock().unwrap();
        g.save(&mp)?;
    }
    if !f.is_empty() {
        bail!("dos_minute_v1 build failed: {}", f.join("; "));
    }
    write_quality(&output, &manifest.lock().unwrap())?;
    Ok(output)
}

fn run_block(
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
        let universe = ctx.universe.get(date);
        let done = {
            let m = manifest.lock().unwrap();
            target
                && universe.is_some()
                && m.date_complete(date) == Some(true)
                && day_file(output, date).is_file()
        };
        let Some(day) = loader::load_day(root, date)? else {
            if target && !done {
                record(manifest, output, date, "missing_partition", 0)?
            }
            continue;
        };
        let mut inters: Vec<(String, DayInter)> = day
            .stocks
            .iter()
            .map(|(c, b)| (c.clone(), compute_stock_day(b, state.stocks.get(c))))
            .collect();
        // ---- factor 8: minute-by-minute cross-sectional percent ranks ----
        let n_stocks = inters.len();
        let mut ranks: Vec<Vec<Option<f64>>> = vec![vec![None; STD_END + 1]; n_stocks];
        for t in 0..=STD_END {
            let mut idx: Vec<usize> = (0..n_stocks)
                .filter(|&i| inters[i].1.vol_prop[t].is_some())
                .collect();
            if idx.is_empty() {
                continue;
            }
            idx.sort_by(|&a, &b| {
                inters[a].1.vol_prop[t]
                    .unwrap()
                    .partial_cmp(&inters[b].1.vol_prop[t].unwrap())
                    .unwrap()
            });
            let n = idx.len();
            let mut s = 0;
            while s < n {
                let mut e = s + 1;
                while e < n && inters[idx[s]].1.vol_prop[t] == inters[idx[e]].1.vol_prop[t] {
                    e += 1;
                }
                let rank = (s + 1) as f64 / n as f64; // min tie rank / n in (0,1]
                for &i in &idx[s..e] {
                    ranks[i][t] = Some(rank);
                }
                s = e;
            }
        }
        for (i, (_, inter)) in inters.iter_mut().enumerate() {
            let vals: Vec<f64> = ranks[i].iter().flatten().copied().collect();
            let q = match (mean(&vals), sample_std(&vals)) {
                (Some(m), Some(s)) if s > EPS => m / s,
                _ => continue,
            };
            inter.values[I_PATV] = kurtosis_excess(&vals).and_then(|k| finite(q + k));
        }
        // ---- factor 5: cross-sectional OLS residual + 20-day mavg ----
        let cs: Vec<usize> = (0..n_stocks)
            .filter(|&i| inters[i].1.up_g.is_some() && inters[i].1.down_g.is_some())
            .collect();
        let pairs: Vec<(f64, f64)> = cs
            .iter()
            .map(|&i| (inters[i].1.up_g.unwrap(), inters[i].1.down_g.unwrap()))
            .collect();
        let residuals = cross_section_residual(&pairs);
        for (j, &i) in cs.iter().enumerate() {
            let today = residuals.as_ref().map(|r| r[j]);
            let hist = state
                .stocks
                .get(&inters[i].0)
                .map(|h| h.residuals.iter().flatten().copied().collect())
                .unwrap_or_default();
            let mut w: Vec<f64> = hist;
            if let Some(r) = today {
                w.push(r);
            }
            inters[i].1.values[I_FALL_CENTER] = mean(&w).and_then(finite);
        }
        // ---- factor 10: cross-sectional mean of F ----
        let fs: Vec<f64> = inters.iter().filter_map(|x| x.1.f_stat).collect();
        let avg_f = mean(&fs);
        if let Some(avg) = avg_f {
            for (_, inter) in inters.iter_mut() {
                inter.values[I_NOON] = match (inter.f_stat, inter.abs_t0) {
                    (Some(f), Some(t)) if f > avg => Some(t),
                    (Some(_), Some(t)) => Some(-t),
                    _ => None,
                };
            }
        }
        // ---- factor 29: stage-2 cross-sectional adjustment ----
        {
            let mut adjs: Vec<Option<f64>> = Vec::with_capacity(n_stocks);
            let mut s1 = 0.0;
            let mut has_neg_raw = false;
            for (code, inter) in inters.iter() {
                match inter.fuzz_diff {
                    Some(v) if v < 0.0 => {
                        s1 += v;
                        has_neg_raw = true;
                    }
                    _ => {}
                }
                let h = state.stocks.get(code);
                let adj = match inter.fuzz_diff {
                    None => None,
                    Some(v) if v >= 0.0 => Some(v),
                    Some(v) => {
                        let mut w: Vec<f64> = h
                            .map(|h| h.fuzz.iter().flatten().copied().collect())
                            .unwrap_or_default();
                        w.push(v);
                        sample_std(&w)
                            .filter(|s| *s > 0.0)
                            .and_then(|s| finite(v / s))
                    }
                };
                adjs.push(adj);
            }
            let mut s2 = 0.0;
            let mut has_neg_adj = false;
            for &a in &adjs {
                if let Some(v) = a {
                    if v < 0.0 {
                        s2 += v;
                        has_neg_adj = true;
                    }
                }
            }
            let s = if has_neg_raw && has_neg_adj && s2 != 0.0 {
                finite(s1 / s2)
            } else {
                None
            };
            for (i, (_, inter)) in inters.iter_mut().enumerate() {
                inter.values[I_ADJ_FUZZ] = match adjs[i] {
                    Some(v) if v < 0.0 => s.and_then(|s| finite(v * s)),
                    other => other,
                };
            }
        }
        // ---- write target day ----
        if target && !done && universe.is_some() {
            let rows: Vec<DynamicWideRow> = inters
                .iter()
                .filter(|(c, _)| universe.is_some_and(|u| u.contains(c)))
                .map(|(c, inter)| DynamicWideRow {
                    ts_code: c.clone(),
                    values: inter.values.to_vec(),
                })
                .collect();
            // inters follows day.stocks order, which the loader sorts by ts_code.
            writer::write_day_dynamic(output, date, &NAMES, &rows)?;
            record(manifest, output, date, "ok", rows.len())?;
        } else if target && !done {
            record(manifest, output, date, "no_universe", 0)?;
        }
        // ---- advance state strictly after every factor consumed it ----
        let mut resid_by_idx = vec![None; n_stocks];
        for (j, &i) in cs.iter().enumerate() {
            resid_by_idx[i] = residuals.as_ref().map(|r| r[j]);
        }
        for (i, (code, inter)) in inters.into_iter().enumerate() {
            let h = state.stocks.entry(code).or_default();
            h.amounts.push_back(inter.amount_curve);
            while h.amounts.len() > W20 {
                h.amounts.pop_front();
            }
            h.residuals.push_back(resid_by_idx[i]);
            while h.residuals.len() > W20 {
                h.residuals.pop_front();
            }
            h.fuzz.push_back(inter.fuzz_diff);
            while h.fuzz.len() > W10 {
                h.fuzz.pop_front();
            }
            h.last = Some(LastBar237 {
                close: inter.close_237,
                adj_amount: inter.adj_amount_237,
                ret_abs_log: inter.ret_237_abs_log,
            });
        }
    }
    Ok(())
}

fn day_file(root: &Path, date: &str) -> PathBuf {
    root.join(format!("year={}", &date[..4]))
        .join(format!("{date}.parquet"))
}

fn record(m: &Arc<Mutex<Manifest>>, output: &Path, d: &str, s: &str, n: usize) -> Result<()> {
    let mut x = m.lock().unwrap();
    x.record(
        d,
        DayStatus {
            status: s.into(),
            rows: n,
            excluded: BTreeMap::new(),
            elapsed_seconds: 0.,
        },
    );
    x.save(&output.join("manifest.json"))?;
    Ok(())
}

fn write_quality(root: &Path, m: &Manifest) -> Result<()> {
    use arrow::array::Array;
    let mut ok = 0;
    let mut rows = 0;
    for s in m.dates.values() {
        if s.status == "ok" {
            ok += 1;
            rows += s.rows;
        }
    }
    let mut non_null = [0usize; N];
    let mut nulls = [0usize; N];
    let mut lo = [f64::INFINITY; N];
    let mut hi = [f64::NEG_INFINITY; N];
    let mut files = Vec::new();
    for y in std::fs::read_dir(root)? {
        let y = y?.path();
        if !y.is_dir()
            || !y
                .file_name()
                .is_some_and(|x| x.to_string_lossy().starts_with("year="))
        {
            continue;
        }
        for f in std::fs::read_dir(y)? {
            let p = f?.path();
            if p.extension().is_some_and(|x| x == "parquet") {
                files.push(p);
            }
        }
    }
    files.sort();
    for p in files {
        let file = std::fs::File::open(p)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;
        for batch in reader {
            let batch = batch?;
            for k in 0..N {
                let a = batch
                    .column_by_name(NAMES[k])
                    .context("missing dos_minute_v1 quality column")?
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .context("dos_minute_v1 quality type")?;
                for i in 0..a.len() {
                    if a.is_null(i) {
                        nulls[k] += 1;
                    } else {
                        let x = a.value(i);
                        non_null[k] += 1;
                        lo[k] = lo[k].min(x);
                        hi[k] = hi[k].max(x);
                    }
                }
            }
        }
    }
    let factors: Vec<_> = (0..N)
        .map(|k| {
            serde_json::json!({
                "name": NAMES[k],
                "non_null": non_null[k],
                "null": nulls[k],
                "min": (non_null[k] > 0).then_some(lo[k]),
                "max": (non_null[k] > 0).then_some(hi[k]),
                "all_null": non_null[k] == 0,
            })
        })
        .collect();
    std::fs::write(
        root.join("quality_report.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"factor_set":"dos_minute_v1","dates_ok":ok,"rows":rows,"factor_count":N,"factors":factors}),
        )?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(
        t: usize,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        vol: f64,
        amt: f64,
    ) -> loader::Bar {
        loader::Bar {
            minute_index: t as u8,
            open,
            high,
            low,
            close,
            volume_share: vol,
            amount: amt,
        }
    }

    /// Constant-price bars with per-minute volume/amount closures.
    fn synth<T, U>(_t_end: usize, close: T, vol: U) -> Box<[loader::Bar; SESSION_BARS]>
    where
        T: Fn(usize) -> f64,
        U: Fn(usize) -> f64,
    {
        Box::new(std::array::from_fn(|t| {
            let c = close(t);
            let v = vol(t);
            bar(t, c, c, c, c, v, c * v)
        }))
    }

    #[test]
    fn names_are_unique() {
        let mut s = NAMES.to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), N);
    }

    #[test]
    fn phi_matches_known_quantiles() {
        assert!((phi(0.0) - 0.5).abs() < 1e-9);
        assert!((phi(1.959964) - 0.975).abs() < 1e-5);
        assert!((phi(-1.959964) - 0.025).abs() < 1e-5);
        assert!((phi(3.0) - 0.99865).abs() < 1e-5);
    }

    #[test]
    fn student_t_cdf_agrees_with_normal_for_df_240() {
        assert!((student_t_cdf(240.0, 0.0) - 0.5).abs() < 1e-12);
        for x in [-3.0, -1.0, 1.0, 2.5, 3.0] {
            assert!((student_t_cdf(240.0, x) - phi(x)).abs() < 5e-3, "x={x}");
        }
        // symmetric
        assert!((student_t_cdf(240.0, 1.3) + student_t_cdf(240.0, -1.3) - 1.0).abs() < 1e-12);
        // scipy references: t.cdf(1.96, 240) = 0.9744235118, t.cdf(1, 240) = 0.8408411659
        assert!((student_t_cdf(240.0, 1.96) - 0.9744235118).abs() < 1e-9);
        assert!((student_t_cdf(240.0, 1.0) - 0.8408411659).abs() < 1e-9);
        assert!((student_t_cdf(240.0, 3.0) - 0.9985078336).abs() < 1e-9);
    }

    #[test]
    fn ols_residuals_are_orthogonal_and_recovers_exact_solution() {
        // y in span(X) plus a fixed perturbation keeps SSE > 0; the test
        // recomputes F and the intercept t stat from first principles via an
        // independent normal-equation solve.
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).sin()).collect();
        let x4: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).cos()).collect();
        let x5: Vec<f64> = (0..n).map(|i| (i as f64 * 2.1).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 + 3.0 * x1[i] - 1.5 * x2[i] + 1e-3 * ((i * 7 % 13) as f64 - 6.0))
            .collect();
        let o = ols_with_stats(
            &y,
            &[x1.clone(), x2.clone(), x3.clone(), x4.clone(), x5.clone()],
        )
        .unwrap();
        assert_eq!(o.t.len(), 6);
        let p = 6;
        let mut a = vec![vec![0.0_f64; p]; p];
        let mut rhs = vec![0.0_f64; p];
        for i in 0..n {
            let row = [1.0, x1[i], x2[i], x3[i], x4[i], x5[i]];
            for u in 0..p {
                for v in 0..p {
                    a[u][v] += row[u] * row[v];
                }
                rhs[u] += row[u] * y[i];
            }
        }
        let beta = gauss_solve(a.clone(), &rhs).unwrap();
        let ybar = y.iter().sum::<f64>() / n as f64;
        let (mut sse, mut ssr) = (0.0, 0.0);
        for i in 0..n {
            let mut yhat = beta[0];
            for (j, c) in [x1.clone(), x2.clone(), x3.clone(), x4.clone(), x5.clone()]
                .iter()
                .enumerate()
            {
                yhat += beta[j + 1] * c[i];
            }
            sse += (y[i] - yhat).powi(2);
            ssr += (yhat - ybar).powi(2);
        }
        let dof = (n - p) as f64;
        let f_expected = (ssr / 5.0) / (sse / dof);
        assert!((o.f - f_expected).abs() < 1e-8 * f_expected.abs().max(1.0));
        // intercept t stat from first principles
        let mut e = vec![0.0_f64; p];
        e[0] = 1.0;
        let inv_col = gauss_solve(a, &e).unwrap();
        let se = (sse / dof * inv_col[0]).sqrt();
        assert!((o.t[0] - beta[0] / se).abs() < 1e-8 * o.t[0].abs().max(1.0));
        assert!(o.t[0].abs() > 5.0);
        assert!(o.f > 100.0);
    }

    #[test]
    fn entropy_matches_hand_computation() {
        // Two bars with amounts 60/40 -> -0.6 ln0.6 - 0.4 ln0.4
        let bars = synth(1, |_t| 10.0, |t| if t % 2 == 0 { 60.0 } else { 40.0 });
        let v = vol_prop_entropy(&bars, 1, false).unwrap();
        let expected = -(0.6_f64 * 0.6_f64.ln() + 0.4_f64 * 0.4_f64.ln());
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn illiq_shortcut_hand_case() {
        // bar with high-low = 1, open-close = 0 -> path 2; trade money = close*5
        let mut bars = synth(2, |t| 10.0 + t as f64, |_| 5.0);
        for b in bars.iter_mut().take(3) {
            b.high = b.close + 0.5;
            b.low = b.close - 0.5;
            b.open = b.close;
        }
        // t_end = 2 covers three bars: closes 10, 11, 12
        let expected = 2.0 / (10.0 * 5.0) + 2.0 / (11.0 * 5.0) + 2.0 / (12.0 * 5.0);
        assert!((illiq_shortcut(&bars, 2).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn consist_volume_filters_and_ratio() {
        // bar0: |c-o| = 0 <= 0.5*0 with h=l -> qualifies (volume 3)
        // bar1: |c-o| = 1 > 0.5*0 -> no; bar2: |c-o|=2 <= 0.5*8=4 -> yes and rising
        let bars = Box::new(std::array::from_fn(|t| match t {
            0 => bar(0, 10.0, 10.0, 10.0, 10.0, 3.0, 30.0),
            1 => bar(1, 10.0, 10.0, 10.0, 11.0, 4.0, 44.0),
            2 => bar(2, 10.0, 14.0, 6.0, 12.0, 5.0, 60.0),
            _ => bar(t, 12.0, 12.0, 12.0, 12.0, 1.0, 12.0),
        }));
        assert!((consist_volume(&bars, 2, false).unwrap() - 8.0 / 12.0).abs() < 1e-12);
        // rising-only: bar0 (no prev) excluded, bar2 close 12 > 11 included
        assert!((consist_volume(&bars, 2, true).unwrap() - 5.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn vol_tide_ratio_single_spike() {
        // Single volume spike at t=10; adjVol = 1 on 6..=14, 0 elsewhere.
        // Peak = first max at 6; troughs at 0 and 15.
        let bars = synth(20, |t| 10.0 + t as f64, |t| if t == 10 { 1.0 } else { 0.0 });
        let v = vol_tide_ratio(&bars, 20).unwrap();
        let expected = ((10.0_f64 / 25.0_f64) - 1.0) / 15.0;
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn volume_peak_count_counts_lunch_gap() {
        let bars = synth(
            STD_END,
            |_| 10.0,
            |t| {
                if matches!(t, 10 | 15 | 120 | 121) {
                    100.0
                } else {
                    1.0
                }
            },
        );
        // gaps 10->15 (5min), 15->120 (105min), 120->121 (91min lunch) all > 1min
        assert_eq!(volume_peak_count(&bars, STD_END).unwrap(), 3.0);
    }

    #[test]
    fn prop_uniform_dis_exact_arithmetic() {
        // +5% per-bar returns with growing amounts; the weight is the constant
        // (0.05-0.1)/0.2, so value = w * sum_{t>=1} amount / sum_{t>=0} amount.
        let t_end = 10;
        let mut prev = 10.0_f64;
        let bars = Box::new(std::array::from_fn(|t| {
            let c = if t == 0 { 10.0 } else { prev * 1.05 };
            prev = c;
            bar(t, c, c, c, c, 100.0, c * 100.0)
        }));
        let v = prop_active(&bars, t_end, 3).unwrap();
        let den: f64 = (0..=t_end).map(|t| bars[t].amount).sum();
        let mut num = 0.0;
        for t in 1..=t_end {
            let r = (bars[t].close - bars[t - 1].close) / bars[t - 1].close;
            num += bars[t].amount * (r - 0.1) / 0.2;
        }
        assert!((v - num / den).abs() < 1e-15);
    }

    #[test]
    fn prop_t_dis_uses_day_std_and_denominator_includes_bar0() {
        // constant returns r: w = cdfStudent(240, r/sqrt(0)) undefined? std of
        // identical values is 0 -> None. Use alternating returns instead and
        // verify against an independent recomputation.
        let t_end = 30;
        let bars = Box::new(std::array::from_fn(|t| {
            let c = if t % 2 == 0 { 10.0 } else { 10.5 };
            bar(t, c, c, c, c, 10.0, c * 10.0)
        }));
        let rets: Vec<f64> = (1..=t_end)
            .map(|t| (bars[t].close - bars[t - 1].close) / bars[t - 1].close)
            .collect();
        let sd = sample_std(&rets).unwrap();
        let mut num = 0.0;
        let mut den = 0.0;
        for t in 0..=t_end {
            den += bars[t].amount;
            if t >= 1 {
                let r = (bars[t].close - bars[t - 1].close) / bars[t - 1].close;
                num += bars[t].amount * student_t_cdf(T_DF, r / sd);
            }
        }
        let expected = num / den;
        assert!((prop_active(&bars, t_end, 0).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn gravity_centers_and_cross_section_residual() {
        // closes 10,11,10,12 -> rets +0.1, -1/11, +0.2 at t=1,2,3
        let bars = Box::new(std::array::from_fn(|t| match t {
            0 => bar(0, 10.0, 10.0, 10.0, 10.0, 1.0, 10.0),
            1 => bar(1, 11.0, 11.0, 11.0, 11.0, 1.0, 11.0),
            2 => bar(2, 10.0, 10.0, 10.0, 10.0, 1.0, 10.0),
            3 => bar(3, 12.0, 12.0, 12.0, 12.0, 1.0, 12.0),
            _ => bar(t, 12.0, 12.0, 12.0, 12.0, 1.0, 12.0),
        }));
        let (up, down) = gravity_centers(&bars, 3);
        let up_e = (1.0 * 0.1 + 3.0 * 0.2) / 0.3;
        let down_e = 2.0 * (-1.0 / 11.0) / (-1.0 / 11.0);
        assert!((up.unwrap() - up_e).abs() < 1e-12);
        assert!((down.unwrap() - down_e).abs() < 1e-12);
        // hand-checked OLS residuals: [(1,2),(2,4),(3,7)] -> [1/6, -1/3, 1/6]
        let r = cross_section_residual(&[(1.0, 2.0), (2.0, 4.0), (3.0, 7.0)]).unwrap();
        assert!((r[0] - 1.0 / 6.0).abs() < 1e-12);
        assert!((r[1] + 1.0 / 3.0).abs() < 1e-12);
        assert!((r[2] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn corr_absret_amount_lag_semantics() {
        // closes 10, 11, 10.5, 12 -> three distinct |log returns| at t=1,2,3
        let bars = Box::new(std::array::from_fn(|t| match t {
            0 => bar(0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0),
            1 => bar(1, 11.0, 11.0, 11.0, 11.0, 1.0, 1.0),
            2 => bar(2, 10.5, 10.5, 10.5, 10.5, 1.0, 2.0),
            3 => bar(3, 12.0, 12.0, 12.0, 12.0, 1.0, 3.0),
            _ => bar(t, 12.0, 12.0, 12.0, 12.0, 1.0, 5.0),
        }));
        let a1: Vec<(f64, f64)> = (1..=3)
            .map(|t| {
                (
                    ((bars[t].close / bars[t - 1].close).ln()).abs(),
                    bars[t].amount,
                )
            })
            .collect();
        assert_eq!(
            corr_absret_amount(&bars, 3, false, false).unwrap(),
            pearson(&a1).unwrap()
        );
        let a2: Vec<(f64, f64)> = (1..=3)
            .map(|t| {
                (
                    ((bars[t].close / bars[t - 1].close).ln()).abs(),
                    bars[t - 1].amount,
                )
            })
            .collect();
        assert_eq!(
            corr_absret_amount(&bars, 3, false, true).unwrap(),
            pearson(&a2).unwrap()
        );
        // lagged ret: bar 0/1 have no predecessor -> pairs start at t=2
        let a3: Vec<(f64, f64)> = (2..=3)
            .map(|t| {
                (
                    ((bars[t - 1].close / bars[t - 2].close).ln()).abs(),
                    bars[t].amount,
                )
            })
            .collect();
        assert_eq!(
            corr_absret_amount(&bars, 3, true, false).unwrap(),
            pearson(&a3).unwrap()
        );
    }

    #[test]
    fn fuzziness_matches_reference_loops() {
        let t_end = 40;
        let bars = Box::new(std::array::from_fn(|t| {
            // deterministic wiggle so fuzziness is non-degenerate
            let c = 10.0 * (1.0 + 0.01 * ((t * 7 % 13) as f64 - 6.0) * 0.1);
            bar(t, c, c, c, c, 100.0, c * 100.0)
        }));
        let ret = |t: usize| -> Option<f64> {
            (t >= 1).then(|| (bars[t].close - bars[t - 1].close) / bars[t - 1].close)
        };
        // independent reference implementation
        let mut inner = vec![None; t_end + 1];
        for j in 0..=t_end {
            let w: Vec<f64> = (j.saturating_sub(4)..=j).filter_map(|u| ret(u)).collect();
            if j >= 4 && w.len() == 5 {
                inner[j] = sample_std(&w);
            }
        }
        let mut expected = vec![None; t_end + 1];
        for k in 0..=t_end {
            let w: Vec<f64> = (k.saturating_sub(4)..=k).filter_map(|u| inner[u]).collect();
            if k >= 8 && w.len() == 5 {
                expected[k] = sample_std(&w);
            }
        }
        let got = fuzziness_series(&bars, t_end);
        for t in 0..=t_end {
            assert_eq!(got[t], expected[t], "t={t}");
        }
        // head nulls, tail populated
        assert!(got[0..9].iter().all(|x| x.is_none()));
        assert!(got[9..=t_end].iter().all(|x| x.is_some()));
    }

    #[test]
    fn support_area_hand_case() {
        // prices 10 (vol 6), 11 (vol 3), 12 (vol 1): support = 10 (max vol).
        // Sorted by |p-10|: 10 (6/10), 11 (3/10), 12 (1/10); cumsum hits 0.9
        // at the first row -> area = {10}.
        let bars = Box::new(std::array::from_fn(|t| match t % 3 {
            0 => bar(t, 10.0, 10.5, 9.5, 10.0, 2.0, 20.0),
            1 => bar(t, 11.0, 11.5, 10.5, 11.0, 1.0, 11.0),
            _ => bar(t, 12.0, 12.5, 11.5, 12.0, 0.5, 6.0),
        }));
        // pDisVol: min area price - day high = 10 - 12.5
        assert!((support_value(&bars, 30, true, false).unwrap() - (10.0 - 12.5)).abs() < 1e-12);
        // bDisVol: max area price - day low = 10 - 9.5
        assert!((support_value(&bars, 30, false, false).unwrap() - (10.0 - 9.5)).abs() < 1e-12);
        // vsaRatio: min area price - highest close = 10 - 12
        assert!((support_value(&bars, 30, true, true).unwrap() - (10.0 - 12.0)).abs() < 1e-12);
    }

    #[test]
    fn adj_amount_and_cross_day_pairing() {
        // two prior days with amounts 100/300 at every minute -> mean 200,
        // sample std of two values = |100|/sqrt(2)
        let mk = |a: f64| -> Box<[f64; SESSION_BARS]> { Box::new([a; SESSION_BARS]) };
        let mut hist = StockHist::default();
        hist.amounts.push_back(mk(100.0));
        hist.amounts.push_back(mk(300.0));
        let t_end = 5;
        let bars = Box::new(std::array::from_fn(|t| {
            bar(t, 10.0, 10.0, 10.0, 10.0, 1.0, 250.0)
        }));
        let adj = adj_amount_series(&hist.amounts, &bars, t_end);
        // mean(100,300)=200; sample std of two values = 100*sqrt(2)
        let expected = (250.0 - 200.0) / (100.0 * 2.0_f64.sqrt());
        assert!((adj[3].unwrap() - expected).abs() < 1e-12);
        // factor 3 pairing: bar 0's lagged amount comes from the prior day's
        // 14:57 bar, bar 0's return from the prior day's close
        let last = LastBar237 {
            close: 9.0,
            adj_amount: Some(1.5),
            ret_abs_log: Some(0.2),
        };
        let v = corr_ret_lag_adj(&bars, t_end, &adj, Some(&last), false, true);
        let mut pairs: Vec<(f64, f64)> = vec![(((10.0_f64 / 9.0).ln()).abs(), 1.5)];
        for t in 1..=t_end {
            pairs.push((0.0, adj[t - 1].unwrap()));
        }
        assert_eq!(v, pearson(&pairs));
        assert_eq!(pairs.len(), t_end + 1);
        // factor 13 bar 0 uses the prior day's |log1p(ret)| and current adj
        let v13 = corr_ret_lag_adj(&bars, t_end, &adj, Some(&last), true, false);
        let pairs13: Vec<(f64, f64)> = std::iter::once((0.2, adj[0].unwrap()))
            .chain((1..=t_end).map(|t| (0.0, adj[t].unwrap())))
            .collect();
        assert_eq!(v13, pearson(&pairs13));
    }

    #[test]
    fn patv_rank_and_moments() {
        let bars_a = synth(STD_END, |t| 10.0 + (t % 3) as f64 * 0.1, |_| 100.0);
        let ia = compute_stock_day(&bars_a, None);
        assert!(ia.values[I_ILLIQ].is_some());
        assert!(ia.vol_prop.iter().all(|x| x.is_some()));
        assert_eq!(mean(&[]), None);
        // kurtosis of two values -> excess = m4/m2^2 - 3 = -2
        assert!((kurtosis_excess(&[1.0, 2.0]).unwrap() + 2.0).abs() < 1e-12);
    }

    // ---- integration: block/thread invariance on a synthetic fixture ----

    struct Fixture {
        _dir: tempfile::TempDir,
        catalog: PathBuf,
        minute_root: PathBuf,
    }

    fn pseudo_random(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) % 10_000) as f64 / 10_000.0
        }
    }

    fn build_fixture() -> Fixture {
        use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray, UInt8Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        let dir = tempfile::tempdir().unwrap();
        let minute_root = dir.path().join("minute");
        let dates = [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ];
        let codes = ["000001.SZ", "600000.SH", "300750.SZ"];
        let catalog_path = dir.path().join("catalog.duckdb");
        {
            let conn = duckdb::Connection::open(&catalog_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE observed_calendar(trade_date DATE, is_observed_market_day BOOLEAN);
                 CREATE TABLE index_trading_universe(index_code VARCHAR, trade_date DATE, ts_code VARCHAR);",
            )
            .unwrap();
            for date in [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
                "2024-01-09",
            ] {
                let observed = dates.contains(&date);
                conn.execute(
                    "INSERT INTO observed_calendar VALUES (STRPTIME(?, '%Y-%m-%d'), ?)",
                    duckdb::params![date, observed],
                )
                .unwrap();
            }
            for date in dates {
                for code in codes {
                    conn.execute(
                        "INSERT INTO index_trading_universe VALUES ('000905.SH', STRPTIME(?, '%Y-%m-%d'), ?)",
                        duckdb::params![date, code],
                    )
                    .unwrap();
                }
            }
        }
        for (di, date) in dates.iter().enumerate() {
            let day_dir = minute_root
                .join(format!("year={}", &date[..4]))
                .join(format!("month={}", &date[5..7]))
                .join(format!("trade_date={date}"));
            std::fs::create_dir_all(&day_dir).unwrap();
            let mut rows: Vec<(String, u8, f64, f64, f64, f64, i64, f64)> = Vec::new();
            for (ci, code) in codes.iter().enumerate() {
                let mut drift = pseudo_random(500 + 31 * (di * 10 + ci) as u64);
                let mut price = 10.0 + ci as f64;
                for t in 0..SESSION_BARS {
                    let open = price;
                    price *= 1.0 + (drift() - 0.5) * 0.01;
                    let hi = open.max(price) * (1.0 + drift() * 0.002);
                    let lo = open.min(price) * (1.0 - drift() * 0.002);
                    let vol = 1_000 + (drift() * 900.0) as i64;
                    rows.push((
                        code.to_string(),
                        t as u8,
                        open,
                        hi,
                        lo,
                        price,
                        vol,
                        price * vol as f64,
                    ));
                }
            }
            let schema = Arc::new(Schema::new(vec![
                Field::new("ts_code", DataType::Utf8, false),
                Field::new("minute_index", DataType::UInt8, false),
                Field::new("open", DataType::Float64, true),
                Field::new("high", DataType::Float64, true),
                Field::new("low", DataType::Float64, true),
                Field::new("close", DataType::Float64, true),
                Field::new("volume_share", DataType::Int64, true),
                Field::new("amount_cny", DataType::Float64, true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(
                        rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                    )),
                    Arc::new(UInt8Array::from(
                        rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float64Array::from(
                        rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float64Array::from(
                        rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float64Array::from(
                        rows.iter().map(|r| r.4).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float64Array::from(
                        rows.iter().map(|r| r.5).collect::<Vec<_>>(),
                    )),
                    Arc::new(Int64Array::from(
                        rows.iter().map(|r| r.6).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float64Array::from(
                        rows.iter().map(|r| r.7).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap();
            let file = std::fs::File::create(day_dir.join("part.parquet")).unwrap();
            let mut w = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }
        Fixture {
            _dir: dir,
            catalog: catalog_path,
            minute_root,
        }
    }

    fn build_args(f: &Fixture, output: &Path) -> BuildArgs {
        BuildArgs {
            catalog: f.catalog.clone(),
            minute_root: f.minute_root.clone(),
            output: output.to_path_buf(),
            start: "2024-01-08".into(),
            end: "2024-01-09".into(),
            block_days: 2,
            jobs: 1,
            threads_per_job: 1,
            memory_limit_mb: 800,
            replace: false,
            factor_set: "dos_minute_v1".into(),
            index_codes: vec!["000905.SH".into()],
            raw_eligible_universe: false,
        }
    }

    fn read_day(path: &Path) -> Vec<(String, Vec<Option<f64>>)> {
        use arrow::array::Array;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let file = std::fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let mut out = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let codes = batch
                .column_by_name("ts_code")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                let mut vals = Vec::with_capacity(N);
                for name in NAMES {
                    let a = batch
                        .column_by_name(name)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                        .unwrap();
                    vals.push(if a.is_null(i) { None } else { Some(a.value(i)) });
                }
                out.push((codes.value(i).to_string(), vals));
            }
        }
        out
    }

    #[test]
    fn block_layout_cannot_change_output_and_warmup_not_written() {
        let f = build_fixture();
        let out1 = f._dir.path().join("o1");
        let out2 = f._dir.path().join("o2");
        let mut a1 = build_args(&f, &out1);
        a1.jobs = 1;
        a1.block_days = 2;
        let mut a2 = build_args(&f, &out2);
        a2.jobs = 2;
        a2.block_days = 1;
        run(a1).unwrap();
        run(a2).unwrap();
        for date in ["2024-01-08", "2024-01-09"] {
            let r1 = read_day(&day_file(&out1, date));
            let r2 = read_day(&day_file(&out2, date));
            assert_eq!(r1.len(), 3);
            assert_eq!(r1, r2, "date {date} differs between layouts");
        }
        // warmup days (before 2024-01-08) must not be written
        assert!(!day_file(&out1, "2024-01-02").exists());
        // sanity: the cross-sectional and state factors produced some values
        let rows = read_day(&day_file(&out1, "2024-01-09"));
        let any_patv = rows.iter().any(|(_, v)| v[I_PATV].is_some());
        let any_noon = rows.iter().any(|(_, v)| v[I_NOON].is_some());
        assert!(any_patv && any_noon);
    }
}
