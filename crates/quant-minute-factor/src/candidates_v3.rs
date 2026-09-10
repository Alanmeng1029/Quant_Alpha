//! V3 minute-OHLCV candidates.  Unlike the older candidate sets this module
//! deliberately keeps a whole block single-threaded: the only parallelism is
//! between independent, warm-up-overlapped blocks.
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::{
    loader,
    manifest::{self, DayStatus, Manifest},
    pipeline,
    schema::{FactorFormula, SESSION_BARS},
    writer::{self, DynamicWideRow},
    BuildArgs,
};

const N: usize = 20;
const W: usize = 20;
const MIN: usize = 10;
const EPS: f64 = 1e-12;
pub const NAMES: [&str; N] = [
    "mf_market_residual_return",
    "mf_tail30_market_residual_return",
    "mf_market_down_resilience",
    "mf_market_residual_rv",
    "mf_cs_rank_top30_persistence",
    "mf_cs_rank_tail30_minus_open30",
    "mf_signed_amount_imbalance",
    "mf_tail30_signed_amount_imbalance",
    "mf_abnormal_signed_amount_imbalance",
    "mf_signed_amount_positive_segment_share",
    "mf_abnormal_flow_followthrough_10m",
    "mf_traded_volume_above_close_share",
    "mf_traded_cost_dispersion",
    "mf_cost_q85_to_close",
    "mf_pm_am_vwap_log_ratio",
    "mf_overnight_gap_sum_w5",
    "mf_gap_open30_reversal",
    "mf_lunch_gap",
    "mf_market_residual_rv_z20",
    "mf_signed_amount_imbalance_z20",
];

#[derive(Clone)]
struct Raw {
    ret: [Option<f64>; 48],
    amount: [f64; 48],
    share: [Option<f64>; 48],
    signed: Option<f64>,
    tail_signed: Option<f64>,
    segment_positive: Option<f64>,
    above_close: Option<f64>,
    cost_dispersion: Option<f64>,
    q85_close: Option<f64>,
    pm_am: Option<f64>,
    lunch: Option<f64>,
    gap_open30: Option<f64>,
    gap: Option<f64>,
}

#[derive(Clone)]
struct Hist {
    day: u32,
    raw: Raw,
    residual_rv: Option<f64>,
    signed: Option<f64>,
    gap: Option<f64>,
}
#[derive(Default)]
struct State {
    stocks: HashMap<String, VecDeque<Hist>>,
}
#[derive(Clone, Copy)]
struct Qfq {
    open: f64,
    close: f64,
}
type QfqMap = HashMap<(String, String), Qfq>;

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
fn continuous_return(b: &[loader::Bar; SESSION_BARS], t: usize) -> Option<f64> {
    if b[t].amount <= 0. || b[t].volume_share <= 0. {
        return None;
    }
    let base = if t == 121 { b[t].open } else { b[t - 1].close };
    (base > 0. && b[t].close > 0.).then(|| (b[t].close / base).ln())
}
fn segment(t: usize) -> Option<usize> {
    match t {
        1..=120 => Some((t - 1) / 5),
        121..=240 => Some(24 + (t - 121) / 5),
        _ => None,
    }
}
fn raw(b: &[loader::Bar; SESSION_BARS], gap: Option<f64>) -> Raw {
    let mut ret = [None; 48];
    let mut amount = [0.; 48];
    let mut signed_n = 0.;
    let mut signed_d = 0.;
    let mut tail_n = 0.;
    let mut tail_d = 0.;
    let mut seg_sign = [0.; 8];
    let mut seg_amount = [0.; 8];
    for t in 1..SESSION_BARS {
        let r = continuous_return(b, t);
        if let Some(k) = segment(t) {
            amount[k] += b[t].amount;
        }
        if let Some(r) = r {
            let a = b[t].amount;
            signed_n += r.signum() * a;
            signed_d += a;
            let s = if t <= 120 {
                (t - 1) / 30
            } else {
                4 + (t - 121) / 30
            };
            seg_sign[s] += r.signum() * a;
            seg_amount[s] += a;
            if t >= 211 {
                tail_n += r.signum() * a;
                tail_d += a;
            }
        }
    }
    for k in 0..48 {
        let start = if k < 24 {
            1 + k * 5
        } else {
            121 + (k - 24) * 5
        };
        let end = start + 4;
        let base = if start == 121 {
            b[start].open
        } else {
            b[start - 1].close
        };
        // A five-minute return is observable when the segment traded at least
        // once; requiring all five one-minute bars to trade makes the 80%
        // cross-sectional coverage rule unusably strict for small A shares.
        if base > 0. && b[end].close > 0. && (start..=end).any(|t| b[t].amount > 0.) {
            ret[k] = finite((b[end].close / base).ln());
        }
    }
    let total: f64 = b.iter().map(|x| x.amount).sum();
    let share = if total > 0. {
        std::array::from_fn(|k| Some(amount[k] / total))
    } else {
        [None; 48]
    };
    let v: f64 = b.iter().map(|x| x.volume_share.max(0.)).sum();
    let close = b[240].close;
    let (above_close, cost_dispersion, q85_close, pm_am) = if v > 0. && close > 0. {
        let mut weighted = Vec::new();
        let mut sum_p = 0.;
        let mut above = 0.;
        let mut am_a = 0.;
        let mut am_v = 0.;
        let mut pm_a = 0.;
        let mut pm_v = 0.;
        for (t, x) in b.iter().enumerate() {
            if x.volume_share <= 0. || x.amount < 0. {
                continue;
            }
            let p = x.amount / x.volume_share;
            if !p.is_finite() || p <= 0. {
                continue;
            }
            sum_p += x.volume_share * p;
            if p > close {
                above += x.volume_share;
            }
            weighted.push((p, x.volume_share));
            if t <= 120 {
                am_a += x.amount;
                am_v += x.volume_share;
            } else {
                pm_a += x.amount;
                pm_v += x.volume_share;
            }
        }
        let vw = sum_p / v;
        let sd = (weighted
            .iter()
            .map(|(p, w)| w * (p - vw).powi(2))
            .sum::<f64>()
            / v)
            .sqrt();
        weighted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut c = 0.;
        let mut q = None;
        for (p, w) in weighted {
            c += w;
            if c >= 0.85 * v {
                q = Some(p);
                break;
            }
        }
        (
            finite(above / v),
            finite(sd / vw),
            q.and_then(|p| finite((p - close) / vw)),
            if am_v > 0. && pm_v > 0. {
                finite((pm_a / pm_v / (am_a / am_v)).ln())
            } else {
                None
            },
        )
    } else {
        (None, None, None, None)
    };
    let lunch = (b[121].open > 0. && b[120].close > 0.).then(|| (b[121].open / b[120].close).ln());
    let gap_open30 = match gap {
        Some(g) if b[0].open > 0. && b[30].close > 0. => {
            finite(-g.signum() * (b[30].close / b[0].open).ln())
        }
        _ => None,
    };
    Raw {
        ret,
        amount,
        share,
        signed: finite(signed_n / signed_d).filter(|_| signed_d > 0.),
        tail_signed: finite(tail_n / tail_d).filter(|_| tail_d > 0.),
        segment_positive: (seg_amount.iter().all(|x| *x > 0.))
            .then(|| seg_sign.iter().filter(|x| **x > 0.).count() as f64 / 8.),
        above_close,
        cost_dispersion,
        q85_close,
        pm_am,
        lunch,
        gap_open30,
        gap,
    }
}

fn history<'a>(h: &'a VecDeque<Hist>, day: u32) -> impl Iterator<Item = &'a Hist> {
    h.iter()
        .filter(move |x| x.day >= day.saturating_sub(W as u32) && x.day < day)
}
fn mean(v: impl Iterator<Item = f64>) -> Option<f64> {
    let x: Vec<_> = v.filter(|z| z.is_finite()).collect();
    (x.len() >= MIN).then(|| x.iter().sum::<f64>() / x.len() as f64)
}
fn z(v: impl Iterator<Item = f64>, x: Option<f64>) -> Option<f64> {
    let x = x?;
    let a: Vec<_> = v.filter(|z| z.is_finite()).collect();
    if a.len() < MIN {
        return None;
    };
    let m = a.iter().sum::<f64>() / a.len() as f64;
    let s = (a.iter().map(|z| (z - m).powi(2)).sum::<f64>() / (a.len() - 1) as f64).sqrt();
    (s > EPS).then(|| (x - m) / s)
}
fn beta(
    h: &VecDeque<Hist>,
    day: u32,
    market: &VecDeque<(u32, [f64; 48], [usize; 48])>,
) -> Option<f64> {
    let own: HashMap<u32, &Raw> = history(h, day).map(|x| (x.day, &x.raw)).collect();
    let mut pairs = Vec::new();
    for (d, sum, count) in market
        .iter()
        .filter(|(d, _, _)| *d >= day.saturating_sub(W as u32) && *d < day)
    {
        if let Some(r) = own.get(d) {
            for k in 0..48 {
                if let Some(a) = r.ret[k] {
                    if count[k] > 1 {
                        pairs.push((a, (sum[k] - a) / (count[k] - 1) as f64));
                    }
                }
            }
        }
    }
    if pairs.len() < MIN {
        return None;
    };
    let ma = pairs.iter().map(|x| x.0).sum::<f64>() / pairs.len() as f64;
    let mb = pairs.iter().map(|x| x.1).sum::<f64>() / pairs.len() as f64;
    let mut cv = 0.;
    let mut vv = 0.;
    for (a, b) in pairs {
        cv += (a - ma) * (b - mb);
        vv += (b - mb).powi(2)
    }
    (vv > EPS).then(|| cv / vv)
}

fn formulas() -> Vec<FactorFormula> {
    NAMES
        .iter()
        .map(|n| FactorFormula {
            name: n,
            formula: "V3 causal minute-OHLCV candidate; see V3 production plan for frozen formula"
                .into(),
        })
        .collect()
}

fn load_qfq(catalog: &Path, calendar: &[String]) -> Result<QfqMap> {
    let conn = duckdb::Connection::open_with_flags(
        catalog,
        duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?,
    )?;
    let start = calendar.first().context("empty calendar")?;
    let end = calendar.last().context("empty calendar")?;
    let mut out = QfqMap::new();
    let mut st=conn.prepare("SELECT trade_date::VARCHAR, ts_code, open, close FROM baostock_qfq_daily WHERE trade_date BETWEEN ?::DATE AND ?::DATE AND open>0 AND close>0")?;
    for row in st.query_map([start, end], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, f64>(2)?,
            r.get::<_, f64>(3)?,
        ))
    })? {
        let (d, c, o, cl) = row?;
        out.insert((d, c), Qfq { open: o, close: cl });
    }
    Ok(out)
}

pub fn run(args: BuildArgs) -> Result<PathBuf> {
    if args.start > args.end {
        bail!("--start must not exceed --end")
    };
    if args.threads_per_job != 1 {
        bail!("ohlcv_candidates_v3 requires --threads-per-job 1 because each block is serial")
    };
    if args.jobs == 0 || args.block_days == 0 {
        bail!("jobs and block-days must be positive")
    };
    std::fs::create_dir_all(args.output.join("_staging"))?;
    let catalog = args.catalog.canonicalize()?;
    let root = args.minute_root.canonicalize()?;
    let output = args.output;
    // Load membership before the earliest possible warm-up date, then only write targets.
    let ctx = Arc::new(pipeline::load_market_context(
        &catalog,
        "1900-01-01",
        &args.end,
        args.memory_limit_mb,
    )?);
    let blocks = Arc::new(pipeline::plan_blocks(
        &ctx.calendar,
        &args.start,
        &args.end,
        args.block_days,
        40,
    ));
    if blocks.is_empty() {
        bail!("no market dates")
    };
    let qfq = Arc::new(load_qfq(&catalog, &ctx.calendar)?);
    let mut src = BTreeMap::new();
    src.insert("catalog".into(), catalog.display().to_string());
    src.insert("minute_root".into(), root.display().to_string());
    src.insert("daily_qfq_source".into(), "baostock_qfq_daily".into());
    src.insert("universe".into(), "dynamic CSI300 union CSI500".into());
    let mp = output.join("manifest.json");
    let mut p = manifest::default_parameters(args.block_days, args.jobs, 1, args.memory_limit_mb);
    p.warmup_days = 40;
    let mut m = if mp.exists() {
        let mut x: Manifest = serde_json::from_str(&std::fs::read_to_string(&mp)?)?;
        if x.factor_set != "ohlcv_candidates_v3" {
            bail!("incompatible manifest")
        };
        x.formulas = formulas();
        x
    } else {
        Manifest::new_named("ohlcv_candidates_v3", p, src, formulas())
    };
    if args.replace {
        for b in blocks.iter() {
            for d in b.target_begin..=b.target_end {
                m.dates.remove(&ctx.calendar[d]);
            }
        }
    };
    let manifest = Arc::new(Mutex::new(m));
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..args.jobs {
            let ctx = ctx.clone();
            let qfq = qfq.clone();
            let blocks = blocks.clone();
            let manifest = manifest.clone();
            let next = next.clone();
            let root = root.clone();
            let output = output.clone();
            let failures = &failures;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= blocks.len() {
                    break;
                }
                if let Err(e) = run_block(&ctx, &qfq, &manifest, &root, &output, &blocks[i]) {
                    failures.lock().unwrap().push(format!("block {i}: {e:#}"));
                    break;
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
        bail!("V3 build failed: {}", f.join("; "))
    };
    write_quality(&output, &manifest.lock().unwrap())?;
    Ok(output)
}

fn run_block(
    ctx: &pipeline::MarketContext,
    qfq: &QfqMap,
    manifest: &Arc<Mutex<Manifest>>,
    root: &Path,
    output: &Path,
    block: &pipeline::Block,
) -> Result<()> {
    let mut state = State::default();
    let mut market_hist: VecDeque<(u32, [f64; 48], [usize; 48])> = VecDeque::new();
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
        let day = loader::load_day(root, date)?;
        if day.is_none() {
            if target && !done {
                record(manifest, output, date, "missing_partition", 0)?
            };
            continue;
        };
        let day = day.unwrap();
        let codes: Vec<_> = day.stocks.iter().map(|x| x.0.clone()).collect();
        let mut raws = HashMap::new();
        for (code, bars) in &day.stocks {
            let gap = previous_gap(qfq, &ctx.calendar, di, code);
            raws.insert(code.clone(), raw(bars, gap));
        }
        let members: Vec<_> = codes
            .iter()
            .filter(|c| universe.is_some_and(|u| u.contains(*c)))
            .cloned()
            .collect();
        let needed = ((members.len() * 4 + 4) / 5).max(3);
        let mut sums = [0.; 48];
        let mut counts = [0usize; 48];
        for c in &members {
            let r = &raws[c];
            for k in 0..48 {
                if let Some(x) = r.ret[k] {
                    sums[k] += x;
                    counts[k] += 1
                }
            }
        }
        let market: [Option<f64>; 48] = std::array::from_fn(|k| {
            if counts[k] >= needed {
                Some(sums[k] / counts[k] as f64)
            } else {
                None
            }
        });
        let mut residuals: HashMap<String, [Option<f64>; 48]> = HashMap::new();
        let mut rv = HashMap::new();
        for c in &codes {
            let beta = state
                .stocks
                .get(c)
                .and_then(|h| beta(h, di as u32, &market_hist));
            let r = &raws[c];
            let e = std::array::from_fn(|k| match (beta, r.ret[k], market[k]) {
                (Some(b), Some(x), Some(_m)) if counts[k] >= needed && counts[k] > 1 => {
                    finite(x - b * ((sums[k] - x) / (counts[k] - 1) as f64))
                }
                _ => None,
            });
            let x = e.iter().flatten().map(|x| x * x).sum::<f64>();
            let n = e.iter().flatten().count();
            rv.insert(c.clone(), (n == 48).then(|| x.sqrt()));
            residuals.insert(c.clone(), e);
        }
        let mut ranks: HashMap<String, [Option<f64>; 48]> = HashMap::new();
        for c in &members {
            ranks.insert(c.clone(), [None; 48]);
        }
        for k in 0..48 {
            let mut x: Vec<_> = members
                .iter()
                .filter_map(|c| residuals[c][k].map(|v| (c.clone(), v)))
                .collect();
            if x.len() < needed {
                continue;
            };
            x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let n = x.len();
            let mut a = 0;
            while a < n {
                let mut z = a + 1;
                while z < n && x[z].1 == x[a].1 {
                    z += 1
                }
                let rank = (a + z - 1) as f64 / 2.;
                let q = if n > 1 { rank / (n - 1) as f64 } else { 0. };
                for j in a..z {
                    ranks.get_mut(&x[j].0).unwrap()[k] = Some(q);
                }
                a = z
            }
        }
        let mut rows = Vec::new();
        for c in &codes {
            let r = &raws[c];
            let e = &residuals[c];
            let h = state.stocks.get(c);
            let mut v = [None; N];
            v[0] = (e.iter().flatten().count() == 48).then(|| e.iter().flatten().sum());
            v[1] =
                (e[42..48].iter().flatten().count() == 6).then(|| e[42..48].iter().flatten().sum());
            v[3] = rv[c];
            v[6] = r.signed;
            v[7] = r.tail_signed;
            v[9] = r.segment_positive;
            v[11] = r.above_close;
            v[12] = r.cost_dispersion;
            v[13] = r.q85_close;
            v[14] = r.pm_am;
            v[16] = r.gap_open30;
            v[17] = r.lunch;
            if let Some(h) = h {
                v[18] = z(history(h, di as u32).filter_map(|x| x.residual_rv), v[3]);
                v[19] = z(history(h, di as u32).filter_map(|x| x.signed), v[6]);
                v[15] = gap5(history(h, di as u32).filter_map(|x| x.gap), r.gap);
                let base: [Option<f64>; 48] = std::array::from_fn(|k| {
                    mean(history(h, di as u32).filter_map(|x| x.raw.share[k]))
                });
                v[8] = abnormal_signed(r, &base);
                v[10] = follow(r, &base, e);
                if beta(h, di as u32, &market_hist).is_some() {
                    v[2] = down_resilience(e, r, h, &market_hist, di as u32, &sums, &counts);
                }
            }
            if let Some(q) = ranks.get(c) {
                v[4] = rank_persist(q);
                v[5] = rank_diff(q);
            }
            if target && !done && universe.is_some_and(|u| u.contains(c)) {
                rows.push(DynamicWideRow {
                    ts_code: c.clone(),
                    values: v.to_vec(),
                })
            }
        }
        // History is advanced only after every factor used its strictly-prior state.
        for c in &codes {
            let entry = raws.remove(c).unwrap();
            let signed = entry.signed;
            let gap = entry.gap;
            let h = state.stocks.entry(c.clone()).or_default();
            h.push_back(Hist {
                day: di as u32,
                raw: entry,
                residual_rv: rv[c],
                signed,
                gap,
            });
            while h.len() > W {
                h.pop_front();
            }
        }
        market_hist.push_back((di as u32, sums, counts));
        while market_hist.len() > W {
            market_hist.pop_front();
        }
        if target && !done {
            if universe.is_some() {
                rows.sort_by(|a, b| a.ts_code.cmp(&b.ts_code));
                writer::write_day_dynamic(output, date, &NAMES, &rows)?;
                record(manifest, output, date, "ok", rows.len())?
            } else {
                record(manifest, output, date, "no_universe", 0)?
            }
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
fn previous_gap(q: &QfqMap, cal: &[String], di: usize, c: &str) -> Option<f64> {
    if di == 0 {
        return None;
    };
    let a = q.get(&(cal[di].clone(), c.into()))?;
    let p = q.get(&(cal[di - 1].clone(), c.into()))?;
    finite((a.open / p.close).ln())
}
fn gap5(h: impl Iterator<Item = f64>, now: Option<f64>) -> Option<f64> {
    let mut x: Vec<_> = h.collect();
    x.push(now?);
    (x.len() >= 5).then(|| x.iter().rev().take(5).sum())
}
fn abnormal_signed(r: &Raw, b: &[Option<f64>; 48]) -> Option<f64> {
    let total: f64 = r.amount.iter().sum();
    let (mut n, mut d) = (0., 0.);
    for k in 0..48 {
        let (x, base) = (r.ret[k]?, b[k]?);
        let excess = (r.amount[k] - total * base).max(0.);
        if r.share[k]? >= 2. * base && x != 0. {
            n += x.signum() * excess;
            d += excess
        }
    }
    (d > 0.).then(|| n / d)
}
fn follow(r: &Raw, b: &[Option<f64>; 48], e: &[Option<f64>; 48]) -> Option<f64> {
    let (mut vals, mut k) = (Vec::new(), 0);
    while k + 2 < 48 {
        let x = r.ret[k]?;
        let base = b[k]?;
        if r.share[k]? >= 2. * base && x != 0. {
            if let (Some(a), Some(b)) = (e[k + 1], e[k + 2]) {
                vals.push(x.signum() * (a + b));
                k += 3;
                continue;
            }
        }
        k += 1
    }
    (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
}
fn down_resilience(
    e: &[Option<f64>; 48],
    raw: &Raw,
    h: &VecDeque<Hist>,
    market: &VecDeque<(u32, [f64; 48], [usize; 48])>,
    day: u32,
    sums: &[f64; 48],
    counts: &[usize; 48],
) -> Option<f64> {
    let old: HashMap<u32, &Raw> = history(h, day).map(|x| (x.day, &x.raw)).collect();
    let mut xs = Vec::new();
    for k in 0..48 {
        let current = match (raw.ret[k], counts[k]) {
            (Some(x), n) if n > 1 => (sums[k] - x) / (n - 1) as f64,
            _ => continue,
        };
        let mut prior: Vec<f64> = market
            .iter()
            .filter(|(d, _, _)| *d >= day.saturating_sub(W as u32) && *d < day)
            .filter_map(|(d, sum, n)| {
                old.get(d)
                    .and_then(|r| r.ret[k])
                    .filter(|_| n[k] > 1)
                    .map(|x| (sum[k] - x) / (n[k] - 1) as f64)
            })
            .collect();
        if prior.len() < MIN || current >= 0. {
            continue;
        }
        prior.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = prior[((prior.len() - 1) as f64 * 0.20).floor() as usize];
        if current < threshold {
            if let Some(x) = e[k] {
                xs.push(x)
            }
        }
    }
    (xs.len() >= 3).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}
fn rank_persist(q: &[Option<f64>; 48]) -> Option<f64> {
    let mut a = 0;
    let mut n = 0;
    for k in 0..47 {
        if k == 23 {
            continue;
        }
        if let (Some(x), Some(y)) = (q[k], q[k + 1]) {
            n += 1;
            if x >= 0.7 && y >= 0.7 {
                a += 1
            }
        }
    }
    (n == 46).then(|| a as f64 / n as f64)
}
fn rank_diff(q: &[Option<f64>; 48]) -> Option<f64> {
    let a: Vec<_> = q[0..6].iter().flatten().collect();
    let b: Vec<_> = q[42..48].iter().flatten().collect();
    (a.len() == 6 && b.len() == 6)
        .then(|| b.iter().map(|x| **x).sum::<f64>() / 6. - a.iter().map(|x| **x).sum::<f64>() / 6.)
}
fn write_quality(root: &Path, m: &Manifest) -> Result<()> {
    use arrow::array::Array;
    let mut ok = 0;
    let mut rows = 0;
    for s in m.dates.values() {
        if s.status == "ok" {
            ok += 1;
            rows += s.rows
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
                files.push(p)
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
                    .context("missing V3 quality column")?
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .context("V3 quality type")?;
                for i in 0..a.len() {
                    if a.is_null(i) {
                        nulls[k] += 1
                    } else {
                        let x = a.value(i);
                        non_null[k] += 1;
                        lo[k] = lo[k].min(x);
                        hi[k] = hi[k].max(x)
                    }
                }
            }
        }
    }
    let factors:Vec<_>=(0..N).map(|k|serde_json::json!({"name":NAMES[k],"non_null":non_null[k],"null":nulls[k],"min":(non_null[k]>0).then_some(lo[k]),"max":(non_null[k]>0).then_some(hi[k]),"all_null":non_null[k]==0})).collect();
    std::fs::write(
        root.join("quality_report.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"factor_set":"ohlcv_candidates_v3","dates_ok":ok,"rows":rows,"factor_count":N,"factors":factors}),
        )?,
    )?;
    Ok(())
}
