use anyhow::{bail, Context, Result};
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const CSI500: &str = "000905.SH";
const CSI1000: &str = "000852.SH";

#[derive(Parser, Debug)]
#[command(about = "Independent CSI500/CSI1000 real-holdings backtest, blended 80/20")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    predictions: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 80)]
    csi500_holdings: usize,
    #[arg(long, default_value_t = 100)]
    csi1000_holdings: usize,
    #[arg(long, default_value_t = 0.03)]
    csi500_turnover: f64,
    #[arg(long, default_value_t = 0.02)]
    csi1000_turnover: f64,
    #[arg(long, default_value_t = 2.1)]
    buy_bps: f64,
    #[arg(long, default_value_t = 7.1)]
    sell_bps: f64,
    #[arg(long, default_value_t = 0.02)]
    cash_reserve: f64,
    #[arg(long, default_value_t = 0.8)]
    csi500_weight: f64,
    #[arg(long, default_value_t = 10_000_000.0)]
    initial_capital: f64,
    #[arg(long, default_value = "independent")]
    selection_mode: String,
    #[arg(long, default_value_t = 3)]
    shared_replacements: usize,
    #[arg(long, default_value_t = 1)]
    rebalance_every: usize,
}

#[derive(Clone)]
struct Prediction {
    code: String,
    h1: f64,
    h5: f64,
}
#[derive(Clone, Copy)]
struct Quote {
    raw_open: f64,
    ratio: f64,
    tradable: bool,
}
#[derive(Clone)]
struct DayTarget {
    signal: String,
    execution: String,
    next_execution: String,
    codes: BTreeSet<String>,
}
#[derive(Serialize)]
struct AccountSummary {
    total_return: f64,
    average_buy_turnover: f64,
    average_sell_turnover: f64,
    average_cash_weight: f64,
    average_holding_count: f64,
}
#[derive(Serialize)]
struct Metrics {
    days: usize,
    start: String,
    end: String,
    total_return: f64,
    csi500_total_return: f64,
    excess_curve_total_return: f64,
    excess_annual_return_243: f64,
    excess_sharpe_243: f64,
    excess_max_drawdown: f64,
    excess_calmar: f64,
    composite_score: f64,
    average_buy_turnover: f64,
    average_sell_turnover: f64,
    average_cash_weight: f64,
    average_holding_count: f64,
    csi500: AccountSummary,
    csi1000: AccountSummary,
}
#[derive(Clone)]
struct Daily {
    signal: String,
    execution: String,
    next_execution: String,
    gross: f64,
    net: f64,
    cost: f64,
    buy: f64,
    sell: f64,
    holdings: usize,
    cash_weight: f64,
    benchmark: f64,
}

fn sql_quote(path: &Path) -> String {
    path.to_string_lossy().replace(char::from(39), "''")
}
fn open(catalog: &Path) -> Result<Connection> {
    let cfg = Config::default().access_mode(AccessMode::ReadOnly)?;
    Ok(Connection::open_with_flags(catalog, cfg)?)
}
fn population_z(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let sd = var.sqrt();
    if sd <= 1e-12 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|x| (x - mean) / sd).collect()
    }
}
fn quota(day: usize, rate: f64, count: usize) -> usize {
    (((day + 1) as f64 * rate * count as f64).floor() - (day as f64 * rate * count as f64).floor())
        as usize
}
fn choose(
    ranked: &[String],
    members: &HashMap<String, String>,
    sleeve: &str,
    held: &mut BTreeSet<String>,
    target: usize,
    slots: usize,
) {
    held.retain(|c| members.get(c).map(String::as_str) == Some(sleeve));
    let sleeve_ranked: Vec<&String> = ranked
        .iter()
        .filter(|c| members.get(*c).map(String::as_str) == Some(sleeve))
        .collect();
    let mut remaining = slots;
    for code in &sleeve_ranked {
        if held.len() >= target || remaining == 0 {
            break;
        }
        if held.insert((*code).clone()) {
            remaining -= 1;
        }
    }
    let ranks: HashMap<&str, usize> = sleeve_ranked
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i + 1))
        .collect();
    let exit_rank = target * 6 / 5;
    while remaining > 0 {
        let worst = held
            .iter()
            .filter_map(|c| {
                let r = *ranks.get(c.as_str()).unwrap_or(&usize::MAX);
                (r > exit_rank).then_some((r, c.clone()))
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
        let Some((_, old)) = worst else { break };
        let replacement = sleeve_ranked
            .iter()
            .find(|c| !held.contains(c.as_str()))
            .map(|c| (*c).clone());
        let Some(new_code) = replacement else { break };
        held.remove(&old);
        held.insert(new_code);
        remaining -= 1;
    }
}

fn choose_shared(
    ranked: &[String],
    members: &HashMap<String, String>,
    held500: &mut BTreeSet<String>,
    held1000: &mut BTreeSet<String>,
    target500: usize,
    target1000: usize,
    slots: usize,
) {
    held500.retain(|c| members.get(c).map(String::as_str) == Some(CSI500));
    held1000.retain(|c| members.get(c).map(String::as_str) == Some(CSI1000));
    let r500: Vec<&String> = ranked
        .iter()
        .filter(|c| members.get(*c).map(String::as_str) == Some(CSI500))
        .collect();
    let r1000: Vec<&String> = ranked
        .iter()
        .filter(|c| members.get(*c).map(String::as_str) == Some(CSI1000))
        .collect();
    let ranks500: HashMap<&str, usize> = r500
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i + 1))
        .collect();
    let ranks1000: HashMap<&str, usize> = r1000
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i + 1))
        .collect();
    let mut remaining = slots;
    for (sleeve, target, candidates) in [
        (held500 as &mut BTreeSet<String>, target500, &r500),
        (held1000 as &mut BTreeSet<String>, target1000, &r1000),
    ] {
        for code in candidates.iter() {
            if sleeve.len() >= target || remaining == 0 {
                break;
            }
            if sleeve.insert((**code).clone()) {
                remaining -= 1;
            }
        }
    }
    #[derive(Clone)]
    struct Exit {
        deterioration: f64,
        sleeve: &'static str,
        code: String,
    }
    let mut exits = Vec::new();
    for code in held500.iter() {
        let rank = *ranks500.get(code.as_str()).unwrap_or(&usize::MAX);
        if rank > target500 * 6 / 5 {
            exits.push(Exit {
                deterioration: rank as f64 / target500 as f64,
                sleeve: CSI500,
                code: code.clone(),
            });
        }
    }
    for code in held1000.iter() {
        let rank = *ranks1000.get(code.as_str()).unwrap_or(&usize::MAX);
        if rank > target1000 * 6 / 5 {
            exits.push(Exit {
                deterioration: rank as f64 / target1000 as f64,
                sleeve: CSI1000,
                code: code.clone(),
            });
        }
    }
    exits.sort_by(|a, b| {
        b.deterioration
            .total_cmp(&a.deterioration)
            .then_with(|| a.sleeve.cmp(b.sleeve))
            .then_with(|| a.code.cmp(&b.code))
    });
    for exit in exits {
        if remaining == 0 {
            break;
        }
        let (held, candidates) = if exit.sleeve == CSI500 {
            (&mut *held500, &r500)
        } else {
            (&mut *held1000, &r1000)
        };
        let replacement = candidates
            .iter()
            .find(|c| !held.contains(c.as_str()))
            .map(|c| (**c).clone());
        if let Some(new_code) = replacement {
            held.remove(&exit.code);
            held.insert(new_code);
            remaining -= 1;
        }
    }
}

fn load_inputs(
    args: &Args,
) -> Result<(
    Vec<String>,
    BTreeMap<String, Vec<Prediction>>,
    HashMap<String, String>,
    HashMap<String, HashMap<String, String>>,
)> {
    let conn = open(&args.catalog)?;
    let pred_path = sql_quote(&args.predictions);
    let mut preds = BTreeMap::<String, Vec<Prediction>>::new();
    let query=format!("SELECT trade_date::VARCHAR,execution_date::VARCHAR,ts_code,pred_h1::DOUBLE,pred_h5::DOUBLE FROM read_parquet('{pred_path}') WHERE execution_date IS NOT NULL AND isfinite(pred_h1) AND isfinite(pred_h5) ORDER BY 1,3");
    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, f64>(3)?,
            r.get::<_, f64>(4)?,
        ))
    })?;
    let mut executions = HashMap::new();
    for row in rows {
        let (d, e, c, h1, h5) = row?;
        executions.insert(d.clone(), e);
        preds
            .entry(d)
            .or_default()
            .push(Prediction { code: c, h1, h5 });
    }
    let days: Vec<String> = preds.keys().cloned().collect();
    if days.len() < 3 {
        bail!("prediction dataset has fewer than three dates")
    }
    let lo = &days[0];
    let hi = &days[days.len() - 1];
    let membership_sql = format!(
        r#"SELECT cal.trade_date::VARCHAR,c.index_code,c.ts_code
      FROM observed_calendar cal JOIN index_monthly_constituents c ON c.index_code IN ('{CSI500}','{CSI1000}')
      AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date)
      WHERE cal.trade_date BETWEEN DATE '{lo}' AND DATE '{hi}'"#
    );
    let mut raw: HashMap<String, HashMap<String, BTreeSet<String>>> = HashMap::new();
    let mut stmt = conn.prepare(&membership_sql)?;
    for row in stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })? {
        let (d, i, c) = row?;
        raw.entry(d).or_default().entry(i).or_default().insert(c);
    }
    let mut membership = HashMap::new();
    for d in &days {
        let r = raw.get(d).context("missing point-in-time membership")?;
        let mut m = HashMap::new();
        if let Some(x) = r.get(CSI500) {
            for c in x {
                m.insert(c.clone(), CSI500.to_string());
            }
        }
        if let Some(x) = r.get(CSI1000) {
            for c in x {
                m.entry(c.clone()).or_insert_with(|| CSI1000.to_string());
            }
        }
        membership.insert(d.clone(), m);
    }
    Ok((days, preds, executions, membership))
}

fn build_targets(
    args: &Args,
    days: &[String],
    preds: &BTreeMap<String, Vec<Prediction>>,
    executions: &HashMap<String, String>,
    membership: &HashMap<String, HashMap<String, String>>,
) -> (Vec<DayTarget>, Vec<DayTarget>) {
    let mut h500 = BTreeSet::new();
    let mut h1000 = BTreeSet::new();
    let mut t500 = Vec::new();
    let mut t1000 = Vec::new();
    for (i, day) in days.iter().enumerate() {
        let p = &preds[day];
        let m = &membership[day];
        let h1: Vec<f64> = p.iter().map(|x| x.h1).collect();
        let h5: Vec<f64> = p.iter().map(|x| x.h5).collect();
        let z1 = population_z(&h1);
        let z5 = population_z(&h5);
        let mut ranked: Vec<(String, f64)> = p
            .iter()
            .enumerate()
            .filter(|(_, x)| m.contains_key(&x.code))
            .map(|(j, x)| (x.code.clone(), 0.5 * z1[j] + 0.5 * z5[j]))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let ranked: Vec<String> = ranked.into_iter().map(|x| x.0).collect();
        if i == 0 {
            choose(
                &ranked,
                m,
                CSI500,
                &mut h500,
                args.csi500_holdings,
                args.csi500_holdings,
            );
            choose(
                &ranked,
                m,
                CSI1000,
                &mut h1000,
                args.csi1000_holdings,
                args.csi1000_holdings,
            );
        } else if args.selection_mode == "shared" {
            let slots = if i % args.rebalance_every == 0 {
                args.shared_replacements
            } else {
                0
            };
            choose_shared(
                &ranked,
                m,
                &mut h500,
                &mut h1000,
                args.csi500_holdings,
                args.csi1000_holdings,
                slots,
            );
        } else {
            let slots500 = if args.rebalance_every == 1 {
                quota(i, args.csi500_turnover, args.csi500_holdings)
            } else if i % args.rebalance_every == 0 {
                (args.csi500_turnover * args.csi500_holdings as f64 * args.rebalance_every as f64)
                    .round() as usize
            } else {
                0
            };
            let slots1000 = if args.rebalance_every == 1 {
                quota(i, args.csi1000_turnover, args.csi1000_holdings)
            } else if i % args.rebalance_every == 0 {
                (args.csi1000_turnover * args.csi1000_holdings as f64 * args.rebalance_every as f64)
                    .round() as usize
            } else {
                0
            };
            choose(
                &ranked,
                m,
                CSI500,
                &mut h500,
                args.csi500_holdings,
                slots500,
            );
            choose(
                &ranked,
                m,
                CSI1000,
                &mut h1000,
                args.csi1000_holdings,
                slots1000,
            );
        }
        if i + 1 < days.len() {
            let base = (
                day.clone(),
                executions[day].clone(),
                executions[&days[i + 1]].clone(),
            );
            let available: BTreeSet<&str> = p.iter().map(|x| x.code.as_str()).collect();
            let executable500 = h500
                .iter()
                .filter(|c| available.contains(c.as_str()))
                .cloned()
                .collect();
            let executable1000 = h1000
                .iter()
                .filter(|c| available.contains(c.as_str()))
                .cloned()
                .collect();
            t500.push(DayTarget {
                signal: base.0.clone(),
                execution: base.1.clone(),
                next_execution: base.2.clone(),
                codes: executable500,
            });
            t1000.push(DayTarget {
                signal: base.0,
                execution: base.1,
                next_execution: base.2,
                codes: executable1000,
            });
        }
    }
    (t500, t1000)
}

fn load_market(
    args: &Args,
    start: &str,
    end: &str,
) -> Result<(HashMap<(String, String), Quote>, HashMap<String, f64>)> {
    let conn = open(&args.catalog)?;
    let pred = sql_quote(&args.predictions);
    let sql = format!(
        r#"SELECT d.trade_date::VARCHAR,d.ts_code,d.open::DOUBLE,d.qfq_open/d.open,
      d.observation_status='complete_trading' AND d.amount_cny>0 FROM daily_qfq d
      JOIN (SELECT DISTINCT ts_code FROM read_parquet('{pred}')) p USING(ts_code)
      WHERE d.trade_date BETWEEN DATE '{start}' AND DATE '{end}' AND d.open>0 AND d.qfq_open>0"#
    );
    let mut quotes = HashMap::new();
    let mut stmt = conn.prepare(&sql)?;
    for row in stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, f64>(2)?,
            r.get::<_, f64>(3)?,
            r.get::<_, bool>(4)?,
        ))
    })? {
        let (d, c, o, ratio, tradable) = row?;
        quotes.insert(
            (d, c),
            Quote {
                raw_open: o,
                ratio,
                tradable,
            },
        );
    }
    let sql=format!("SELECT trade_date::VARCHAR,open::DOUBLE FROM index_daily WHERE index_code='{CSI500}' AND trade_date BETWEEN DATE '{start}' AND DATE '{end}' AND open>0");
    let mut bench = HashMap::new();
    let mut stmt = conn.prepare(&sql)?;
    for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)))? {
        let (d, o) = row?;
        bench.insert(d, o);
    }
    Ok((quotes, bench))
}

fn account(
    targets: &[DayTarget],
    quotes: &HashMap<(String, String), Quote>,
    benchmark: &HashMap<String, f64>,
    initial: f64,
    n: usize,
    max_weight: f64,
    args: &Args,
) -> Vec<Daily> {
    let mut shares = HashMap::<String, f64>::new();
    let mut ratios = HashMap::<String, f64>::new();
    let mut last = HashMap::<String, f64>::new();
    let mut cash = initial;
    let mut previous = initial;
    let lot = 100.0;
    let mut out = Vec::new();
    for target in targets {
        for (c, q) in shares.clone() {
            if let Some(x) = quotes.get(&(target.execution.clone(), c.clone())) {
                if let Some(old) = ratios.get(&c) {
                    shares.insert(c.clone(), q * x.ratio / old);
                }
                ratios.insert(c, x.ratio);
            }
        }
        let equity = cash
            + shares
                .iter()
                .map(|(c, q)| {
                    quotes
                        .get(&(target.execution.clone(), c.clone()))
                        .map(|x| q * x.raw_open)
                        .unwrap_or(*last.get(c).unwrap_or(&0.0))
                })
                .sum::<f64>();
        let mut sold = 0.0;
        let mut bought = 0.0;
        let mut fees = 0.0;
        let exits: Vec<String> = shares
            .keys()
            .filter(|c| !target.codes.contains(*c))
            .cloned()
            .collect();
        for c in exits {
            if let Some(x) = quotes.get(&(target.execution.clone(), c.clone())) {
                if x.tradable {
                    let q = shares.remove(&c).unwrap();
                    let value = q * x.raw_open;
                    let fee = value * args.sell_bps / 10000.0;
                    cash += value - fee;
                    sold += value;
                    fees += fee;
                    ratios.remove(&c);
                    last.remove(&c);
                }
            }
        }
        let missing: Vec<String> = target
            .codes
            .iter()
            .filter(|c| !shares.contains_key(*c))
            .cloned()
            .collect();
        for (k, c) in missing.iter().enumerate() {
            if let Some(x) = quotes.get(&(target.execution.clone(), c.clone())) {
                if x.tradable {
                    let affordable = (cash - equity * args.cash_reserve).max(0.0)
                        / (1.0 + args.buy_bps / 10000.0);
                    let deploy = affordable / (missing.len() - k) as f64;
                    let reference = equity * (1.0 - args.cash_reserve) / n as f64;
                    // Match the cash-balanced research executor: a replacement
                    // may absorb accumulated sale proceeds, bounded by the
                    // sleeve-relative single-name cap.
                    let value = reference
                        .max(deploy.min(equity * max_weight))
                        .min(affordable);
                    let q = (value / x.raw_open / lot).floor() * lot;
                    let value = q * x.raw_open;
                    let fee = value * args.buy_bps / 10000.0;
                    if q > 0.0 && value + fee <= cash - equity * args.cash_reserve + 1e-8 {
                        cash -= value + fee;
                        bought += value;
                        fees += fee;
                        shares.insert(c.clone(), q);
                        ratios.insert(c.clone(), x.ratio);
                    }
                }
            }
        }
        let mut ending = cash;
        for (c, q) in &shares {
            let current = quotes.get(&(target.execution.clone(), c.clone()));
            let future = quotes.get(&(target.next_execution.clone(), c.clone()));
            if let (Some(a), Some(b)) = (current, future) {
                ending += q * b.ratio / a.ratio * b.raw_open;
            } else {
                ending += *last.get(c).unwrap_or(&0.0);
            }
            if let Some(a) = current {
                last.insert(c.clone(), q * a.raw_open);
            }
        }
        let gross = (ending - previous + fees) / previous;
        let net = ending / previous - 1.0;
        let b = benchmark
            .get(&target.execution)
            .zip(benchmark.get(&target.next_execution))
            .map(|(a, z)| z / a - 1.0)
            .unwrap_or(0.0);
        out.push(Daily {
            signal: target.signal.clone(),
            execution: target.execution.clone(),
            next_execution: target.next_execution.clone(),
            gross,
            net,
            cost: fees / previous,
            buy: bought / equity,
            sell: sold / equity,
            holdings: shares.len(),
            cash_weight: cash / equity,
            benchmark: b,
        });
        previous = ending;
    }
    out
}

fn summary(d: &[Daily]) -> AccountSummary {
    AccountSummary {
        total_return: d.iter().fold(1.0, |a, x| a * (1.0 + x.net)) - 1.0,
        average_buy_turnover: d.iter().map(|x| x.buy).sum::<f64>() / d.len() as f64,
        average_sell_turnover: d.iter().map(|x| x.sell).sum::<f64>() / d.len() as f64,
        average_cash_weight: d.iter().map(|x| x.cash_weight).sum::<f64>() / d.len() as f64,
        average_holding_count: d.iter().map(|x| x.holdings as f64).sum::<f64>() / d.len() as f64,
    }
}
fn metric(combined: &[Daily], s500: AccountSummary, s1000: AccountSummary) -> Metrics {
    let n = combined.len();
    let mut nav = 1.0;
    let mut bnav = 1.0;
    let mut curve = 1.0;
    let mut peak: f64 = 1.0;
    let mut mdd: f64 = 0.0;
    let mut eps = Vec::new();
    for d in combined {
        nav *= 1.0 + d.net;
        bnav *= 1.0 + d.benchmark;
        let e = d.net - d.benchmark;
        eps.push(e);
        curve *= 1.0 + e;
        peak = peak.max(curve);
        mdd = mdd.max(1.0 - curve / peak);
    }
    let mean = eps.iter().sum::<f64>() / n as f64;
    let sd = (eps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let sharpe = mean / sd * 243f64.sqrt();
    let annual = curve.powf(243.0 / n as f64) - 1.0;
    let calmar = annual / mdd;
    Metrics {
        days: n,
        start: combined[0].execution.clone(),
        end: combined[n - 1].next_execution.clone(),
        total_return: nav - 1.0,
        csi500_total_return: bnav - 1.0,
        excess_curve_total_return: curve - 1.0,
        excess_annual_return_243: annual,
        excess_sharpe_243: sharpe,
        excess_max_drawdown: mdd,
        excess_calmar: calmar,
        composite_score: 0.9 * sharpe + 0.1 * calmar,
        average_buy_turnover: combined.iter().map(|x| x.buy).sum::<f64>() / n as f64,
        average_sell_turnover: combined.iter().map(|x| x.sell).sum::<f64>() / n as f64,
        average_cash_weight: combined.iter().map(|x| x.cash_weight).sum::<f64>() / n as f64,
        average_holding_count: combined.iter().map(|x| x.holdings as f64).sum::<f64>() / n as f64,
        csi500: s500,
        csi1000: s1000,
    }
}
fn svg_points(values: &[f64], width: f64, height: f64) -> String {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-12);
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let x = i as f64 * width / (values.len().saturating_sub(1).max(1)) as f64;
            let y = height - (value - lo) * height / span;
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}
fn write_html(args: &Args, d: &[Daily], metrics: &Metrics) -> Result<()> {
    let mut nav = 1.0;
    let mut benchmark = 1.0;
    let mut excess = 1.0;
    let mut navs = Vec::with_capacity(d.len());
    let mut benchmarks = Vec::with_capacity(d.len());
    let mut excesses = Vec::with_capacity(d.len());
    for day in d {
        nav *= 1.0 + day.net;
        benchmark *= 1.0 + day.benchmark;
        excess *= 1.0 + day.net - day.benchmark;
        navs.push(nav);
        benchmarks.push(benchmark);
        excesses.push(excess);
    }
    let html = format!(
        r##"<!doctype html><meta charset="utf-8"><title>Rust dual-sleeve backtest</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;color:#18202a}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{border:1px solid #dfe5ec;border-radius:10px;padding:14px;background:#fff}}.v{{font-size:25px;font-weight:700;margin-top:5px}}svg{{width:100%;height:360px;border:1px solid #e3e7eb;background:#fafbfd}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.legend span{{margin-right:20px}}@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}</style>
<h1>Rust CSI500 {:.0}% + CSI1000 {:.0}% 组合回测</h1><p>{} 至 {}，{} 个收益日；CSI500 Top{} / {:.1}% 换手，CSI1000 Top{} / {:.1}% 换手。</p>
<div class="grid"><div class="card">组合总收益<div class="v">{:.2}%</div></div><div class="card">年化超额<div class="v">{:.2}%</div></div><div class="card">超额 Sharpe<div class="v">{:.3}</div></div><div class="card">综合评分<div class="v">{:.3}</div></div><div class="card">超额最大回撤<div class="v">{:.2}%</div></div><div class="card">超额 Calmar<div class="v">{:.3}</div></div><div class="card">日均买入换手<div class="v">{:.2}%</div></div><div class="card">平均现金<div class="v">{:.2}%</div></div></div>
<h2>净值与超额复利曲线</h2><div class="legend"><span style="color:#1976d2">● 组合净值</span><span style="color:#777">● CSI500</span><span style="color:#d84315">● 超额复利曲线</span></div><svg viewBox="0 0 1100 360" preserveAspectRatio="none"><polyline fill="none" stroke="#1976d2" stroke-width="2" points="{}"/><polyline fill="none" stroke="#777" stroke-width="2" points="{}"/><polyline fill="none" stroke="#d84315" stroke-width="2" points="{}"/></svg>
<h2>袖套执行</h2><table><tr><th>袖套</th><th>总收益</th><th>日均买入换手</th><th>平均现金</th><th>平均持股数</th></tr><tr><td>CSI500（{:.0}%）</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.1}</td></tr><tr><td>CSI1000（{:.0}%）</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.1}</td></tr></table><p>超额口径：逐日 <code>r_p-r_CSI500</code> 后复利；年化因子243；综合评分 = 0.9×超额Sharpe + 0.1×超额Calmar。</p>"##,
        100.0 * args.csi500_weight,
        100.0 * (1.0 - args.csi500_weight),
        metrics.start,
        metrics.end,
        metrics.days,
        args.csi500_holdings,
        100.0 * args.csi500_turnover,
        args.csi1000_holdings,
        100.0 * args.csi1000_turnover,
        100.0 * metrics.total_return,
        100.0 * metrics.excess_annual_return_243,
        metrics.excess_sharpe_243,
        metrics.composite_score,
        100.0 * metrics.excess_max_drawdown,
        metrics.excess_calmar,
        100.0 * metrics.average_buy_turnover,
        100.0 * metrics.average_cash_weight,
        svg_points(&navs, 1100.0, 360.0),
        svg_points(&benchmarks, 1100.0, 360.0),
        svg_points(&excesses, 1100.0, 360.0),
        100.0 * args.csi500_weight,
        100.0 * metrics.csi500.total_return,
        100.0 * metrics.csi500.average_buy_turnover,
        100.0 * metrics.csi500.average_cash_weight,
        metrics.csi500.average_holding_count,
        100.0 * (1.0 - args.csi500_weight),
        100.0 * metrics.csi1000.total_return,
        100.0 * metrics.csi1000.average_buy_turnover,
        100.0 * metrics.csi1000.average_cash_weight,
        metrics.csi1000.average_holding_count
    );
    fs::write(args.output.join("report.html"), html)?;
    Ok(())
}
fn write_output(args: &Args, d: &[Daily], metrics: &Metrics) -> Result<()> {
    fs::create_dir_all(&args.output)?;
    let tsv = args.output.join("portfolio_daily.tsv");
    let mut w = BufWriter::new(File::create(&tsv)?);
    writeln!(w,"signal_date\texecution_date\tnext_execution_date\tgross_return\tnet_return\ttransaction_cost\tbuy_turnover\tsell_turnover\tholding_count\tcash_weight\tcsi500_return\tactive_return")?;
    for x in d {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            x.signal,
            x.execution,
            x.next_execution,
            x.gross,
            x.net,
            x.cost,
            x.buy,
            x.sell,
            x.holdings,
            x.cash_weight,
            x.benchmark,
            x.net - x.benchmark
        )?;
    }
    drop(w);
    let conn = Connection::open_in_memory()?;
    let src = sql_quote(&tsv);
    let dst = sql_quote(&args.output.join("portfolio_daily.parquet"));
    conn.execute_batch(&format!("COPY (SELECT CAST(signal_date AS DATE) signal_date,CAST(execution_date AS DATE) execution_date,CAST(next_execution_date AS DATE) next_execution_date,* EXCLUDE(signal_date,execution_date,next_execution_date) FROM read_csv('{src}',delim='\\t',header=true)) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)"))?;
    fs::remove_file(tsv)?;
    fs::write(
        args.output.join("summary.json"),
        serde_json::to_string_pretty(metrics)? + "\n",
    )?;
    fs::write(
        args.output.join("manifest.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"engine":"quant-dual-sleeve-backtest-rust-v1","args":format!("{args:?}"),"method":"two independently traded sleeves; daily returns blended 80/20"}),
        )? + "\n",
    )?;
    write_html(args, d, metrics)?;
    Ok(())
}
fn write_targets(args: &Args, a: &[DayTarget], b: &[DayTarget]) -> Result<()> {
    fs::create_dir_all(&args.output)?;
    let tsv = args.output.join("targets.tsv");
    let mut w = BufWriter::new(File::create(&tsv)?);
    writeln!(w, "trade_date\texecution_date\tts_code\tsleeve")?;
    for (sleeve, days) in [(CSI500, a), (CSI1000, b)] {
        for day in days {
            for code in &day.codes {
                writeln!(w, "{}\t{}\t{}\t{}", day.signal, day.execution, code, sleeve)?;
            }
        }
    }
    drop(w);
    let conn = Connection::open_in_memory()?;
    let src = sql_quote(&tsv);
    let dst = sql_quote(&args.output.join("dual_sleeve_targets.parquet"));
    conn.execute_batch(&format!("COPY (SELECT CAST(trade_date AS DATE) trade_date,CAST(execution_date AS DATE) execution_date,ts_code,sleeve FROM read_csv('{src}',delim='\\t',header=true)) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)"))?;
    fs::remove_file(tsv)?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    if args.selection_mode != "independent" && args.selection_mode != "shared" {
        bail!("selection-mode must be independent or shared")
    }
    if args.rebalance_every == 0 {
        bail!("rebalance-every must be positive")
    }
    if !(0.0..=1.0).contains(&args.csi500_weight) || args.initial_capital <= 0.0 {
        bail!("csi500-weight must be in [0,1] and initial-capital must be positive")
    }
    if !(0.0..=1.0).contains(&args.csi500_turnover) || !(0.0..=1.0).contains(&args.csi1000_turnover)
    {
        bail!("turnover must be in [0,1]")
    }
    let (days, preds, executions, membership) = load_inputs(&args)?;
    let (t500, t1000) = build_targets(&args, &days, &preds, &executions, &membership);
    write_targets(&args, &t500, &t1000)?;
    let start = &t500[0].execution;
    let end = &t500.last().unwrap().next_execution;
    let (quotes, benchmark) = load_market(&args, start, end)?;
    let d500 = account(
        &t500,
        &quotes,
        &benchmark,
        args.initial_capital * args.csi500_weight.max(0.01),
        args.csi500_holdings,
        0.0375,
        &args,
    );
    let d1000 = account(
        &t1000,
        &quotes,
        &benchmark,
        args.initial_capital * (1.0 - args.csi500_weight).max(0.01),
        args.csi1000_holdings,
        0.15,
        &args,
    );
    let w500 = args.csi500_weight;
    let w1000 = 1.0 - w500;
    let combined: Vec<Daily> = d500
        .iter()
        .zip(&d1000)
        .map(|(a, b)| Daily {
            signal: a.signal.clone(),
            execution: a.execution.clone(),
            next_execution: a.next_execution.clone(),
            gross: w500 * a.gross + w1000 * b.gross,
            net: w500 * a.net + w1000 * b.net,
            cost: w500 * a.cost + w1000 * b.cost,
            buy: w500 * a.buy + w1000 * b.buy,
            sell: w500 * a.sell + w1000 * b.sell,
            holdings: (if w500 > 0.0 { a.holdings } else { 0 })
                + (if w1000 > 0.0 { b.holdings } else { 0 }),
            cash_weight: w500 * a.cash_weight + w1000 * b.cash_weight,
            benchmark: a.benchmark,
        })
        .collect();
    let metrics = metric(&combined, summary(&d500), summary(&d1000));
    write_output(&args, &combined, &metrics)?;
    println!("{}", serde_json::to_string(&metrics)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rate_quota_is_exact_over_time() {
        let total: usize = (0..100).map(|d| quota(d, 0.03, 80)).sum();
        assert_eq!(total, 240);
    }
    #[test]
    fn zscore_constant_is_zero() {
        assert_eq!(population_z(&[2.0, 2.0]), vec![0.0, 0.0]);
    }
}
