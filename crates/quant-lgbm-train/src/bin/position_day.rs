use anyhow::{Context, Result, bail};
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Create one production CSI500 target-position parquet from H1 predictions")]
struct Args {
    #[arg(long)]
    predictions: PathBuf,
    #[arg(long)]
    catalog: PathBuf,
    /// Local SSE exchange calendar. This may extend beyond the last ingested
    /// market observation, allowing the latest signal day to target tomorrow.
    #[arg(long, default_value = "A_stock_database/交易日历.csv")]
    exchange_calendar: PathBuf,
    /// Previous target or executed position parquet. Optional only for initial deployment.
    #[arg(long)]
    previous_positions: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0.98)]
    invested_weight: f64,
    #[arg(long, default_value_t = 0.01)]
    max_weight: f64,
    #[arg(long, default_value_t = 2.0)]
    buy_bps: f64,
    #[arg(long, default_value_t = 2.0)]
    sell_bps: f64,
}

fn next_execution_date(calendar: &Path, signal_date: &str) -> Result<String> {
    let text = fs::read_to_string(calendar)
        .with_context(|| format!("read exchange calendar {}", calendar.display()))?;
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let mut fields = line.trim_start_matches('\u{feff}').split(',');
            let exchange = fields.next()?;
            let date = fields.next()?;
            let status = fields.next()?;
            (exchange == "SSE" && status == "交易" && date > signal_date).then_some(date.to_owned())
        })
        .min()
        .with_context(|| format!("exchange calendar has no trading day after {signal_date}"))
}

#[derive(Clone)]
struct Candidate {
    code: String,
    mu: f64,
    old: f64,
}

#[derive(Serialize)]
struct PositionManifest {
    engine: &'static str,
    signal_date: String,
    execution_date: String,
    prediction_file: String,
    previous_positions: Option<String>,
    rows: usize,
    invested_weight: f64,
    max_weight: f64,
    buy_bps: f64,
    sell_bps: f64,
    target_buy_turnover: f64,
    target_sell_turnover: f64,
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(args.max_weight > 0.0
        && args.max_weight <= args.invested_weight
        && args.invested_weight <= 1.0)
    {
        bail!("require 0 < max_weight <= invested_weight <= 1");
    }
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(&args.catalog, config)?;
    let prediction = args.predictions.canonicalize()?;
    let dates: (String, i64) = conn.query_row(
        &format!(
            "SELECT min(trade_date)::VARCHAR,count(distinct trade_date) FROM read_parquet('{}')",
            quote(&prediction)
        ),
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if dates.1 != 1 {
        bail!("prediction parquet must contain exactly one trade_date");
    }
    let signal_date = dates.0;
    let execution_date = next_execution_date(&args.exchange_calendar, &signal_date)?;

    let mut previous = HashMap::<String, f64>::new();
    if let Some(path) = &args.previous_positions {
        let path = path.canonicalize()?;
        let columns = conn
            .prepare(&format!(
                "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{}'))",
                quote(&path)
            ))?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let weight_column = if columns.iter().any(|x| x == "target_weight") {
            "target_weight"
        } else if columns.iter().any(|x| x == "weight") {
            "weight"
        } else {
            bail!("previous position parquet has neither target_weight nor weight")
        };
        let query = format!(
            "SELECT ts_code,{weight_column}::DOUBLE FROM read_parquet('{}')",
            quote(&path),
        );
        for row in conn
            .prepare(&query)?
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)))?
        {
            let (code, weight) = row?;
            if weight.is_finite() && weight > 0.0 {
                previous.insert(code, weight);
            }
        }
    }
    let query = format!(
        r#"SELECT p.ts_code,p.raw_h1
      FROM read_parquet('{}') p
      JOIN index_monthly_constituents c ON c.ts_code=p.ts_code AND c.index_code='000905.SH'
       AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2
         WHERE c2.index_code=c.index_code AND c2.as_of_date<=p.trade_date)
      WHERE isfinite(p.raw_h1) AND p.ts_code<>'000937.SZ' ORDER BY p.ts_code"#,
        quote(&prediction)
    );
    let mut candidates = conn
        .prepare(&query)?
        .query_map([], |r| {
            Ok(Candidate {
                code: r.get(0)?,
                mu: r.get(1)?,
                old: 0.0,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if candidates.len() as f64 * args.max_weight + 1e-12 < args.invested_weight {
        bail!("eligible universe cannot satisfy invested weight");
    }
    for row in &mut candidates {
        row.old = previous
            .get(&row.code)
            .copied()
            .unwrap_or(0.0)
            .min(args.max_weight);
    }
    let buy_cost = args.buy_bps / 10_000.0;
    let sell_cost = args.sell_bps / 10_000.0;
    let mut segments = Vec::<(f64, String, usize, f64)>::new();
    for (i, row) in candidates.iter().enumerate() {
        if row.old > 0.0 {
            segments.push((row.mu + sell_cost, row.code.clone(), i, row.old));
        }
        let room = args.max_weight - row.old;
        if room > 1e-15 {
            segments.push((row.mu - buy_cost, row.code.clone(), i, room));
        }
    }
    segments.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut weights = vec![0.0; candidates.len()];
    let mut remaining = args.invested_weight;
    for (_, _, i, capacity) in segments {
        let allocation = capacity.min(remaining);
        weights[i] += allocation;
        remaining -= allocation;
        if remaining <= 1e-12 {
            break;
        }
    }
    if remaining > 1e-9 {
        bail!("optimizer failed to allocate {remaining}");
    }
    let eligible_old: f64 = candidates.iter().map(|x| x.old).sum();
    let forced_sales = (previous.values().sum::<f64>() - eligible_old).max(0.0);
    let buys: f64 = weights
        .iter()
        .zip(&candidates)
        .map(|(w, c)| (w - c.old).max(0.0))
        .sum();
    let sells: f64 = weights
        .iter()
        .zip(&candidates)
        .map(|(w, c)| (c.old - w).max(0.0))
        .sum::<f64>()
        + forced_sales;
    let values = candidates
        .iter()
        .zip(&weights)
        .filter(|(_, w)| **w > 1e-12)
        .map(|(row, w)| {
            format!(
                "('{}','{}','{}',{},{})",
                signal_date,
                execution_date,
                row.code.replace('\'', "''"),
                w,
                row.mu
            )
        })
        .collect::<Vec<_>>();
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = args.output.with_extension("parquet.tmp");
    conn.execute_batch(&format!("COPY (SELECT signal_date::DATE signal_date,execution_date::DATE execution_date,ts_code,target_weight,raw_h1 FROM (VALUES {}) v(signal_date,execution_date,ts_code,target_weight,raw_h1) ORDER BY ts_code) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD)", values.join(","), quote(&temporary)))?;
    if args.output.exists() {
        fs::remove_file(&args.output)?;
    }
    fs::rename(&temporary, &args.output)?;
    let manifest = PositionManifest {
        engine: "quant-position-day-rust-v1",
        signal_date,
        execution_date,
        prediction_file: prediction.display().to_string(),
        previous_positions: args.previous_positions.map(|p| p.display().to_string()),
        rows: values.len(),
        invested_weight: args.invested_weight,
        max_weight: args.max_weight,
        buy_bps: args.buy_bps,
        sell_bps: args.sell_bps,
        target_buy_turnover: buys,
        target_sell_turnover: sells,
    };
    fs::write(
        args.output.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("{}", serde_json::to_string(&manifest)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_date_uses_future_exchange_calendar() {
        let dir = tempfile::tempdir().unwrap();
        let calendar = dir.path().join("calendar.csv");
        fs::write(
            &calendar,
            "交易所,日期,是否交易,上一个交易日\nSSE,2026-08-30,休市,2026-08-28\nSSE,2026-08-31,交易,2026-08-28\nSSE,2026-08-28,交易,2026-08-27\n",
        )
        .unwrap();
        assert_eq!(
            next_execution_date(&calendar, "2026-08-28").unwrap(),
            "2026-08-31"
        );
    }
}
