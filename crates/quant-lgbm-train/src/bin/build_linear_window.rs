use anyhow::{Context, Result, bail};
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde_json::json;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Stream one factor-major temporary window for linear solvers")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    daily_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    train_start: String,
    #[arg(long)]
    train_end: String,
    #[arg(long)]
    test_start: String,
    #[arg(long)]
    test_end: String,
    #[arg(long, default_value = "000300.SH,000905.SH")]
    index_code: String,
    #[arg(long, default_value_t = 2048)]
    memory_limit_mb: usize,
}

#[derive(Debug)]
struct BaseRow {
    date: String,
    code: String,
    execution: Option<String>,
    h1: Option<f64>,
    h5: Option<f64>,
    h10: Option<f64>,
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn factors(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        let file = path.join("v1/factor.parquet");
        if file.exists() {
            let stem = path
                .file_name()
                .context("factor directory without name")?
                .to_string_lossy();
            out.push((format!("{stem}_v1"), file));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    if out.is_empty() {
        bail!("no factor artifacts under {}", root.display());
    }
    Ok(out)
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

fn standardize_day(column: &mut [f32], indices: &[usize], values: &[f64]) {
    if values.len() < 2 {
        return;
    }
    let mut lo_values = values.to_vec();
    let mut hi_values = values.to_vec();
    let lo = quantile(&mut lo_values, 0.01);
    let hi = quantile(&mut hi_values, 0.99);
    let clipped = values.iter().map(|v| v.clamp(lo, hi)).collect::<Vec<_>>();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let variance =
        clipped.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (clipped.len() - 1) as f64;
    let sd = variance.sqrt();
    if sd <= 1e-12 {
        return;
    }
    for (&index, value) in indices.iter().zip(clipped) {
        column[index] = ((value - mean) / sd) as f32;
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let factors = factors(&args.daily_root)?;
    fs::create_dir_all(&args.output)?;
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(&args.catalog, config)?;
    conn.execute_batch(&format!(
        "SET threads=1; SET memory_limit='{}MB'; SET preserve_insertion_order=false;",
        args.memory_limit_mb
    ))?;
    let codes = args
        .index_code
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if codes.is_empty()
        || codes
            .iter()
            .any(|v| v.is_empty() || !v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.'))
    {
        bail!("invalid index-code list");
    }
    let code_values = codes
        .iter()
        .map(|v| format!("'{v}'"))
        .collect::<Vec<_>>()
        .join(",");
    let ranges = format!(
        "((cal.trade_date BETWEEN DATE '{}' AND DATE '{}') OR (cal.trade_date BETWEEN DATE '{}' AND DATE '{}'))",
        args.train_start, args.train_end, args.test_start, args.test_end
    );
    let base_query = format!(
        r#"
      WITH calendar AS (SELECT trade_date,row_number() OVER(ORDER BY trade_date) n FROM observed_calendar WHERE is_observed_market_day),
      universe AS (
        SELECT DISTINCT cal.trade_date,c.ts_code FROM calendar cal
        JOIN index_monthly_constituents c ON c.index_code IN ({code_values})
          AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date)
        JOIN daily_aggregated d ON d.trade_date=cal.trade_date AND d.ts_code=c.ts_code
        WHERE {ranges} AND c.ts_code<>'000937.SZ' AND d.open>0 AND d.high>0 AND d.low>0 AND d.close>0
          AND d.amount_cny>0 AND d.volume_share>0 AND d.observation_status='complete_trading'
      ) SELECT u.trade_date::VARCHAR,u.ts_code,ce.trade_date::VARCHAR,
        CASE WHEN d1.qfq_open>0 AND d2.qfq_open>0 AND i1.open>0 AND i2.open>0 AND d1.amount_cny>0 AND d2.amount_cny>0 AND d1.observation_status='complete_trading' AND d2.observation_status='complete_trading' THEN d2.qfq_open/d1.qfq_open-i2.open/i1.open END,
        CASE WHEN d1.qfq_open>0 AND d6.qfq_open>0 AND i1.open>0 AND i6.open>0 AND d1.amount_cny>0 AND d6.amount_cny>0 AND d1.observation_status='complete_trading' AND d6.observation_status='complete_trading' THEN d6.qfq_open/d1.qfq_open-i6.open/i1.open END,
        CASE WHEN d1.qfq_open>0 AND d11.qfq_open>0 AND i1.open>0 AND i11.open>0 AND d1.amount_cny>0 AND d11.amount_cny>0 AND d1.observation_status='complete_trading' AND d11.observation_status='complete_trading' THEN d11.qfq_open/d1.qfq_open-i11.open/i1.open END
      FROM universe u JOIN calendar c ON c.trade_date=u.trade_date
      LEFT JOIN calendar ce ON ce.n=c.n+1 LEFT JOIN calendar c2 ON c2.n=c.n+2 LEFT JOIN calendar c6 ON c6.n=c.n+6 LEFT JOIN calendar c11 ON c11.n=c.n+11
      LEFT JOIN daily_qfq d1 ON d1.ts_code=u.ts_code AND d1.trade_date=ce.trade_date LEFT JOIN daily_qfq d2 ON d2.ts_code=u.ts_code AND d2.trade_date=c2.trade_date
      LEFT JOIN daily_qfq d6 ON d6.ts_code=u.ts_code AND d6.trade_date=c6.trade_date LEFT JOIN daily_qfq d11 ON d11.ts_code=u.ts_code AND d11.trade_date=c11.trade_date
      LEFT JOIN index_daily i1 ON i1.index_code='000905.SH' AND i1.trade_date=ce.trade_date LEFT JOIN index_daily i2 ON i2.index_code='000905.SH' AND i2.trade_date=c2.trade_date
      LEFT JOIN index_daily i6 ON i6.index_code='000905.SH' AND i6.trade_date=c6.trade_date LEFT JOIN index_daily i11 ON i11.index_code='000905.SH' AND i11.trade_date=c11.trade_date
      ORDER BY u.trade_date,u.ts_code
    "#
    );
    let mut stmt = conn.prepare(&base_query)?;
    let mapped = stmt.query_map([], |row| {
        Ok(BaseRow {
            date: row.get(0)?,
            code: row.get(1)?,
            execution: row.get(2)?,
            h1: row.get(3)?,
            h5: row.get(4)?,
            h10: row.get(5)?,
        })
    })?;
    let rows = mapped.collect::<std::result::Result<Vec<_>, _>>()?;
    let train_rows = rows.partition_point(|row| row.date.as_str() <= args.train_end.as_str());
    let mut positions = HashMap::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        positions.insert((row.date.clone(), row.code.clone()), i);
    }
    let mut metadata = BufWriter::new(File::create(args.output.join("rows.tsv"))?);
    writeln!(metadata, "trade_date\tts_code\texecution_date\th1\th5\th10")?;
    for row in &rows {
        writeln!(
            metadata,
            "{}\t{}\t{}\t{}\t{}\t{}",
            row.date,
            row.code,
            row.execution.as_deref().unwrap_or(""),
            row.h1.map_or(String::new(), |v| v.to_string()),
            row.h5.map_or(String::new(), |v| v.to_string()),
            row.h10.map_or(String::new(), |v| v.to_string())
        )?;
    }
    metadata.flush()?;
    let mut matrix = BufWriter::new(File::create(args.output.join("x_col_major.f32"))?);
    for (factor_number, (_, path)) in factors.iter().enumerate() {
        let query = format!(
            "SELECT trade_date::VARCHAR,ts_code,factor_value FROM read_parquet('{}') WHERE (trade_date BETWEEN DATE '{}' AND DATE '{}') OR (trade_date BETWEEN DATE '{}' AND DATE '{}') ORDER BY trade_date,ts_code",
            quote(path),
            args.train_start,
            args.train_end,
            args.test_start,
            args.test_end
        );
        let mut factor_stmt = conn.prepare(&query)?;
        let values = factor_stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<f64>>(2)?,
            ))
        })?;
        let mut column = vec![0f32; rows.len()];
        let mut current = String::new();
        let mut day_indices = Vec::new();
        let mut day_values = Vec::new();
        for item in values {
            let (date, code, value) = item?;
            if !current.is_empty() && date != current {
                standardize_day(&mut column, &day_indices, &day_values);
                day_indices.clear();
                day_values.clear();
            }
            current = date.clone();
            if let (Some(&index), Some(value)) = (
                positions.get(&(date, code)),
                value.filter(|v| v.is_finite()),
            ) {
                day_indices.push(index);
                day_values.push(value);
            }
        }
        standardize_day(&mut column, &day_indices, &day_values);
        for value in column {
            matrix.write_all(&value.to_le_bytes())?;
        }
        if (factor_number + 1) % 25 == 0 || factor_number + 1 == factors.len() {
            eprintln!("factors {}/{}", factor_number + 1, factors.len());
        }
    }
    matrix.flush()?;
    fs::write(
        args.output.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
          "builder":"quant-lgbm-train/build-linear-window-column-stream-v2","factor_ids":factors.iter().map(|x|&x.0).collect::<Vec<_>>(),
          "feature_count":factors.len(),"rows":rows.len(),"train_rows":train_rows,"test_rows":rows.len()-train_rows,
          "layout":"column_major_f32","train_start":args.train_start,"train_end":args.train_end,"test_start":args.test_start,"test_end":args.test_end,
          "missing_policy":"cross_sectional_mean_after_standardization","price_basis":"raw_factors_qfq_open_returns"
        }))?,
    )?;
    println!(
        "{}",
        json!({"output":args.output,"factors":factors.len(),"train_rows":train_rows,"test_rows":rows.len()-train_rows})
    );
    Ok(())
}
