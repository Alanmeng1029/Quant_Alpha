use anyhow::{Result, bail};
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Merge independently trained CSI500 and CSI1000 prediction parquet files")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    csi500: PathBuf,
    #[arg(long)]
    csi1000: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct Summary {
    csi500_rows: i64,
    csi1000_rows: i64,
    merged_rows: i64,
    duplicate_keys: i64,
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(&args.catalog, config)?;
    let p500 = quote(&args.csi500);
    let p1000 = quote(&args.csi1000);
    let universe = "SELECT trade_date,index_code,ts_code FROM index_trading_universe WHERE index_code IN ('000905.SH','000852.SH')";
    let union = format!(
        "WITH universe AS ({universe}), m500 AS (SELECT DISTINCT trade_date,ts_code FROM universe WHERE index_code='000905.SH'), m1000 AS (SELECT DISTINCT trade_date,ts_code FROM universe WHERE index_code='000852.SH' EXCEPT SELECT trade_date,ts_code FROM m500) SELECT p.*,'000905.SH' sleeve_model FROM read_parquet('{p500}') p JOIN m500 USING(trade_date,ts_code) UNION ALL SELECT p.*,'000852.SH' sleeve_model FROM read_parquet('{p1000}') p JOIN m1000 USING(trade_date,ts_code)"
    );
    conn.execute_batch(&format!("CREATE TEMP TABLE merged AS {union}"))?;
    let csi500_rows: i64 = conn.query_row(
        "SELECT count(*) FROM merged WHERE sleeve_model='000905.SH'",
        [],
        |row| row.get(0),
    )?;
    let csi1000_rows: i64 = conn.query_row(
        "SELECT count(*) FROM merged WHERE sleeve_model='000852.SH'",
        [],
        |row| row.get(0),
    )?;
    let duplicate_keys: i64 = conn.query_row(
        "SELECT count(*) FROM (SELECT trade_date,ts_code,count(*) n FROM merged GROUP BY 1,2 HAVING n>1)",
        [],
        |row| row.get(0),
    )?;
    if duplicate_keys != 0 {
        bail!("merged sleeve predictions contain {duplicate_keys} duplicate date/code keys")
    }
    let output = quote(&args.output);
    conn.execute_batch(&format!(
        "COPY (SELECT trade_date,ts_code,raw_h1,raw_h5,pred_h1,pred_h5,execution_date,sleeve_model FROM merged ORDER BY trade_date,ts_code) TO '{output}' (FORMAT PARQUET,COMPRESSION ZSTD,OVERWRITE_OR_IGNORE TRUE)"
    ))?;
    let summary = Summary {
        csi500_rows,
        csi1000_rows,
        merged_rows: csi500_rows + csi1000_rows,
        duplicate_keys,
    };
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
