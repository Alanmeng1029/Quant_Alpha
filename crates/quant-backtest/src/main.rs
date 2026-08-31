use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use duckdb::{AccessMode, Config as DuckDbConfig, Connection, Row};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "quant-backtest",
    about = "Deterministic A-share daily factor evaluation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    FactorEval(FactorEvalArgs),
    BatchFactorEval(BatchFactorEvalArgs),
    #[command(name = "batch-eval", hide = true)]
    BatchEval(BatchEvalLegacyArgs),
}

#[derive(Clone, Parser)]
struct FactorEvalArgs {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    factor: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Parser)]
struct BatchFactorEvalArgs {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    factor_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    batch_id: Option<String>,
    #[arg(long, value_delimiter = ',', default_values_t = [String::from("csi500"), String::from("csi300")])]
    universes: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    tasks: Option<Vec<String>>,
    #[arg(long, default_value = "configs/factor_eval.yaml")]
    csi300_config: PathBuf,
    #[arg(long, default_value = "configs/factor_eval_csi500.yaml")]
    csi500_config: PathBuf,
    #[arg(long, default_value = "configs/factor_eval_all.yaml")]
    all_config: PathBuf,
    #[arg(long)]
    rebuild_cache: bool,
}

/// Compatibility spelling for the pre-cache process-per-factor runner.
#[derive(Parser)]
struct BatchEvalLegacyArgs {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    factor_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    batch_id: Option<String>,
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    #[arg(long, value_delimiter = ',', default_values_t = [String::from("csi500"), String::from("csi300")])]
    universes: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    tasks: Option<Vec<String>>,
    #[arg(long, default_value = "configs/factor_eval.yaml")]
    csi300_config: PathBuf,
    #[arg(long, default_value = "configs/factor_eval_csi500.yaml")]
    csi500_config: PathBuf,
    #[arg(long, default_value = "configs/factor_eval_all.yaml")]
    all_config: PathBuf,
    #[arg(long)]
    rebuild_cache: bool,
}

#[derive(Debug, Deserialize)]
struct Config {
    universe: Option<String>,
    start: Option<String>,
    end: Option<String>,
    horizons: Option<Vec<i32>>,
    transaction_cost_bps: Option<f64>,
    portfolio_direction: Option<String>,
    engine_threads: Option<usize>,
    memory_limit_mb: Option<usize>,
}

#[derive(Debug, Serialize)]
struct Summary {
    factor_id: String,
    factor_path: String,
    catalog: String,
    universe: String,
    start: String,
    end: String,
    horizons: Vec<i32>,
    transaction_cost_bps: f64,
    portfolio_direction: String,
    rows: i64,
    daily_ic_rows: i64,
    portfolio_rows: i64,
    note: String,
}

const LABEL_CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SourceFileFingerprint {
    path: String,
    bytes: u64,
    modified_unix_nanos: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DataStatusFingerprint {
    dataset: String,
    first_date: Option<String>,
    last_date: Option<String>,
    rows: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CacheManifest {
    label_schema_version: u32,
    universe: String,
    start: String,
    end: String,
    horizons: Vec<i32>,
    catalog: String,
    source_files: Vec<SourceFileFingerprint>,
    data_status: Vec<DataStatusFingerprint>,
    index_daily_rows: i64,
    index_daily_first_date: Option<String>,
    index_daily_last_date: Option<String>,
}

#[derive(Clone)]
struct LabelSpec {
    label: &'static str,
    horizon: i32,
    raw_column: String,
    benchmark_column: String,
}

fn label_specs(horizons: &[i32]) -> Vec<LabelSpec> {
    let mut specs = Vec::new();
    for horizon in horizons {
        for label in ["close_to_close", "vwap_to_vwap", "twap_to_twap"] {
            let prefix = format!("{label}_h{horizon}");
            specs.push(LabelSpec {
                label,
                horizon: *horizon,
                raw_column: format!("{prefix}_raw"),
                benchmark_column: format!("{prefix}_csi500"),
            });
        }
    }
    specs
}

#[derive(Clone)]
struct Signal {
    code: String,
    execution_date: String,
    open: Option<f64>,
    vwap: Option<f64>,
    tradable: bool,
}

#[derive(Serialize)]
struct PortfolioRow {
    execution_mode: String,
    execution_date: String,
    gross_return: f64,
    turnover: f64,
    transaction_cost: f64,
    net_return: f64,
    nav: f64,
    holding_count: usize,
    frozen_holding_count: usize,
    stale_price_count: usize,
}

#[derive(Serialize)]
struct HoldingRow {
    execution_mode: String,
    execution_date: String,
    ts_code: String,
    weight: f64,
}

fn quote_sql(value: &str) -> String {
    value.replace('\'', "''")
}

fn copy_query(conn: &Connection, query: &str, path: &Path) -> Result<()> {
    let escaped = quote_sql(&path.to_string_lossy());
    conn.execute_batch(&format!(
        "COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD);"
    ))?;
    Ok(())
}

fn scalar_i64(conn: &Connection, query: &str) -> Result<i64> {
    Ok(conn.query_row(query, [], |row| row.get(0))?)
}

fn universe_sql(name: &str) -> Result<&'static str> {
    match name {
        "all" => Ok("SELECT trade_date, ts_code FROM trading_universe WHERE is_eligible"),
        "csi300" => Ok(
            "SELECT trade_date, ts_code FROM index_trading_universe WHERE index_code = '000300.SH'",
        ),
        "csi500" => Ok(
            "SELECT trade_date, ts_code FROM index_trading_universe WHERE index_code = '000905.SH'",
        ),
        "csi300_csi500" => Ok(
            "SELECT trade_date, ts_code FROM index_trading_universe WHERE index_code IN ('000300.SH','000905.SH') GROUP BY trade_date, ts_code",
        ),
        _ => bail!("Unsupported universe '{name}'. Use all, csi300, csi500, or csi300_csi500"),
    }
}

fn factor_id(path: &Path) -> String {
    let manifest = path
        .parent()
        .unwrap_or(Path::new("."))
        .join("manifest.json");
    fs::read_to_string(manifest)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| value["factor_id"].as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "unknown_factor".to_string())
}

fn config_dates(config: &Config) -> (&str, &str) {
    (
        config.start.as_deref().unwrap_or("1900-01-01"),
        config.end.as_deref().unwrap_or("2999-12-31"),
    )
}

fn config_horizons(config: &Config) -> Vec<i32> {
    config
        .horizons
        .clone()
        .unwrap_or_else(|| vec![1, 5, 10, 20])
}

fn set_duckdb_options(conn: &Connection, config: &Config, temp_directory: &Path) -> Result<()> {
    fs::create_dir_all(temp_directory)?;
    conn.execute_batch(&format!(
        "SET threads TO {}; SET memory_limit='{}MB'; SET preserve_insertion_order=false; SET temp_directory='{}';",
        config.engine_threads.unwrap_or(1),
        config.memory_limit_mb.unwrap_or(3000),
        quote_sql(&temp_directory.to_string_lossy()),
    ))?;
    Ok(())
}

fn collect_file_metadata(
    path: &Path,
    base: &Path,
    output: &mut Vec<SourceFileFingerprint>,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        let mut entries = fs::read_dir(path)?.collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            collect_file_metadata(&entry.path(), base, output)?;
        }
        return Ok(());
    }
    let metadata = fs::metadata(path)?;
    let modified_unix_nanos = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    output.push(SourceFileFingerprint {
        path: path
            .strip_prefix(base)
            .unwrap_or(path)
            .display()
            .to_string(),
        bytes: metadata.len(),
        modified_unix_nanos,
    });
    Ok(())
}

fn cache_manifest(
    conn: &Connection,
    catalog: &Path,
    universe: &str,
    config: &Config,
) -> Result<CacheManifest> {
    let catalog = catalog.canonicalize()?;
    let lake_root = catalog
        .parent()
        .and_then(Path::parent)
        .unwrap_or(Path::new("."));
    let mut source_files = Vec::new();
    for source in [
        catalog.clone(),
        lake_root.join("canonical/daily_aggregated"),
        lake_root.join("canonical/adjustment"),
        lake_root.join("canonical/reference/trading_universe"),
        lake_root.join("canonical/reference/index_constituents"),
        lake_root.join("canonical/reference/observed_calendar.parquet"),
        lake_root.join("canonical/index_daily"),
    ] {
        collect_file_metadata(&source, lake_root, &mut source_files)?;
    }
    source_files.sort_by(|left, right| left.path.cmp(&right.path));
    let mut statement = conn.prepare("SELECT dataset, first_date::VARCHAR, last_date::VARCHAR, rows FROM data_status ORDER BY dataset")?;
    let data_status = statement
        .query_map([], |row| {
            Ok(DataStatusFingerprint {
                dataset: row.get(0)?,
                first_date: row.get(1)?,
                last_date: row.get(2)?,
                rows: row.get(3)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let (index_daily_rows, index_daily_first_date, index_daily_last_date): (i64, Option<String>, Option<String>) = conn.query_row(
        "SELECT count(*)::BIGINT, min(trade_date)::VARCHAR, max(trade_date)::VARCHAR FROM index_daily WHERE index_code = '000905.SH'",
        [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let (start, end) = config_dates(config);
    Ok(CacheManifest {
        label_schema_version: LABEL_CACHE_SCHEMA_VERSION,
        universe: universe.to_string(),
        start: start.to_string(),
        end: end.to_string(),
        horizons: config_horizons(config),
        catalog: catalog.display().to_string(),
        source_files,
        data_status,
        index_daily_rows,
        index_daily_first_date,
        index_daily_last_date,
    })
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_string_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn prepare_views(conn: &Connection, args: &FactorEvalArgs, config: &Config) -> Result<()> {
    let universe = universe_sql(config.universe.as_deref().unwrap_or("all"))?;
    let factor = quote_sql(&args.factor.canonicalize()?.to_string_lossy());
    let start = config.start.as_deref().unwrap_or("1900-01-01");
    let end = config.end.as_deref().unwrap_or("2999-12-31");
    conn.execute_batch(&format!(
        r#"
        CREATE OR REPLACE TEMP VIEW market_calendar AS
          SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS day_index
          FROM observed_calendar WHERE is_observed_market_day;
        CREATE OR REPLACE TEMP VIEW selected_universe AS {universe};
        CREATE OR REPLACE TEMP VIEW cohort AS
          SELECT f.trade_date, f.ts_code, f.factor_value, d.qfq_close
          FROM read_parquet('{factor}') f
          JOIN selected_universe u USING (trade_date, ts_code)
          JOIN daily_qfq d USING (trade_date, ts_code)
          WHERE f.trade_date BETWEEN DATE '{start}' AND DATE '{end}'
            AND isfinite(f.factor_value) AND d.qfq_close > 0;
    "#
    ))?;
    if scalar_i64(conn, "SELECT count(*) FROM cohort")? == 0 {
        bail!("No factor rows remain after date and universe filtering");
    }
    Ok(())
}

fn return_view_sql(horizon: i32, label: &str) -> String {
    if label == "close_to_close" {
        return format!(
            r#"
        SELECT c.trade_date, c.ts_code, c.factor_value,
          target.qfq_close / c.qfq_close - 1 AS raw_return,
          target.qfq_close / c.qfq_close - 1 - (i500_end.close / i500_start.close - 1) AS excess_csi500
        FROM cohort c
        JOIN market_calendar current_day ON current_day.trade_date = c.trade_date
        JOIN market_calendar target_day ON target_day.day_index = current_day.day_index + {horizon}
        JOIN daily_qfq target ON target.ts_code = c.ts_code AND target.trade_date = target_day.trade_date AND target.qfq_close > 0
        LEFT JOIN index_daily i500_start ON i500_start.index_code = '000905.SH' AND i500_start.trade_date = c.trade_date
        LEFT JOIN index_daily i500_end ON i500_end.index_code = '000905.SH' AND i500_end.trade_date = target_day.trade_date
        "#
        );
    }
    let price = match label {
        "vwap_to_vwap" => "qfq_vwap",
        "twap_to_twap" => "qfq_twap",
        _ => unreachable!("validated return label"),
    };
    format!(
        r#"
        SELECT c.trade_date, c.ts_code, c.factor_value,
          exit_day.{price} / entry_day.{price} - 1 AS raw_return,
          exit_day.{price} / entry_day.{price} - 1 - (i500_end.close / i500_start.close - 1) AS excess_csi500
        FROM cohort c
        JOIN market_calendar signal_day ON signal_day.trade_date = c.trade_date
        JOIN market_calendar entry_calendar ON entry_calendar.day_index = signal_day.day_index + 1
        JOIN market_calendar exit_calendar ON exit_calendar.day_index = signal_day.day_index + 1 + {horizon}
        JOIN daily_qfq entry_day ON entry_day.ts_code = c.ts_code AND entry_day.trade_date = entry_calendar.trade_date
        JOIN daily_qfq exit_day ON exit_day.ts_code = c.ts_code AND exit_day.trade_date = exit_calendar.trade_date
        LEFT JOIN index_daily i500_start ON i500_start.index_code = '000905.SH' AND i500_start.trade_date = entry_calendar.trade_date
        LEFT JOIN index_daily i500_end ON i500_end.index_code = '000905.SH' AND i500_end.trade_date = exit_calendar.trade_date
        WHERE entry_day.{price} > 0 AND exit_day.{price} > 0
          AND entry_day.observation_status = 'complete_trading' AND exit_day.observation_status = 'complete_trading'
          AND entry_day.amount_cny > 0 AND exit_day.amount_cny > 0
    "#
    )
}

fn build_ic_and_groups(conn: &Connection, horizons: &[i32], output: &Path) -> Result<()> {
    let mut ic_parts = Vec::new();
    let mut group_parts = Vec::new();
    for horizon in horizons {
        for label in ["close_to_close", "vwap_to_vwap", "twap_to_twap"] {
            let returns = return_view_sql(*horizon, label);
            ic_parts.push(format!(r#"
          WITH returns AS ({returns}), long_returns AS (
            SELECT trade_date, ts_code, factor_value, kind, value FROM returns,
            LATERAL (VALUES ('{label}_raw', raw_return), ('{label}_excess_csi500', excess_csi500)) x(kind, value)
            WHERE value IS NOT NULL AND isfinite(value)
          ), ranked AS (
            SELECT *, rank() OVER (PARTITION BY trade_date, kind ORDER BY factor_value, ts_code) factor_rank,
              rank() OVER (PARTITION BY trade_date, kind ORDER BY value, ts_code) return_rank
            FROM long_returns
          )
          SELECT {horizon}::INTEGER AS horizon, kind AS return_kind, trade_date, count(*) AS sample_count,
            NULL::DOUBLE AS pearson_ic,
            corr(return_rank, factor_rank) AS rank_ic,
            NULL::DOUBLE AS winsorized_pearson_ic
          FROM ranked GROUP BY horizon, return_kind, trade_date
        "#));
            group_parts.push(format!(r#"
          WITH returns AS ({returns}), long_returns AS (
            SELECT trade_date, ts_code, factor_value, kind, value FROM returns,
            LATERAL (VALUES ('{label}_raw', raw_return), ('{label}_excess_csi500', excess_csi500)) x(kind, value)
            WHERE value IS NOT NULL AND isfinite(value)
          ), grouped AS (
            SELECT *, ntile(10) OVER (PARTITION BY trade_date, kind ORDER BY factor_value, ts_code) AS group_number FROM long_returns
          )
          SELECT {horizon}::INTEGER AS horizon, kind AS return_kind, trade_date, group_number,
            avg(value) AS mean_return, count(*) AS sample_count
          FROM grouped GROUP BY horizon, return_kind, trade_date, group_number
        "#));
        }
    }
    copy_query(
        conn,
        &ic_parts
            .iter()
            .map(|part| format!("({part})"))
            .collect::<Vec<_>>()
            .join(" UNION ALL "),
        &output.join("daily_ic.parquet"),
    )?;
    copy_query(
        conn,
        &group_parts
            .iter()
            .map(|part| format!("({part})"))
            .collect::<Vec<_>>()
            .join(" UNION ALL "),
        &output.join("group_returns.parquet"),
    )?;
    copy_query(conn, r#"
        SELECT horizon, return_kind, year(trade_date) AS calendar_year, count(*) AS days,
          avg(rank_ic) AS mean_rank_ic, stddev_samp(rank_ic) AS std_rank_ic,
          avg(rank_ic) / nullif(stddev_samp(rank_ic), 0) * sqrt(252) AS rank_icir,
          avg(CASE WHEN rank_ic > 0 THEN 1.0 ELSE 0.0 END) AS positive_rank_ic_ratio
        FROM read_parquet('__OUTPUT__/daily_ic.parquet') GROUP BY horizon, return_kind, calendar_year
    "#.replace("__OUTPUT__", &quote_sql(&output.to_string_lossy())).as_str(), &output.join("annual_metrics.parquet"))?;
    Ok(())
}

fn create_market_calendar_and_universe(conn: &Connection, universe_name: &str) -> Result<()> {
    let universe = universe_sql(universe_name)?;
    conn.execute_batch(&format!(
        r#"
        CREATE OR REPLACE TEMP VIEW market_calendar AS
          SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS day_index
          FROM observed_calendar WHERE is_observed_market_day;
        CREATE OR REPLACE TEMP VIEW selected_universe AS {universe};
    "#
    ))?;
    Ok(())
}

fn build_label_cache(
    conn: &Connection,
    cache_root: &Path,
    manifest: &CacheManifest,
    config: &Config,
) -> Result<()> {
    create_market_calendar_and_universe(conn, &manifest.universe)?;
    let (start, end) = config_dates(config);
    let horizons = config_horizons(config);
    let temporary = cache_root.parent().unwrap_or(cache_root).join(format!(
        ".{}-building-{}",
        manifest.universe,
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::create_dir_all(&temporary)?;

    let mut joins = vec![
        "JOIN market_calendar signal_calendar ON signal_calendar.trade_date = base.trade_date".to_string(),
        "LEFT JOIN market_calendar entry_calendar ON entry_calendar.day_index = signal_calendar.day_index + 1".to_string(),
        "LEFT JOIN daily_qfq entry_day ON entry_day.ts_code = base.ts_code AND entry_day.trade_date = entry_calendar.trade_date".to_string(),
    ];
    let mut columns = Vec::new();
    for horizon in &horizons {
        joins.push(format!("LEFT JOIN market_calendar close_calendar_{horizon} ON close_calendar_{horizon}.day_index = signal_calendar.day_index + {horizon}"));
        joins.push(format!("LEFT JOIN daily_qfq close_target_{horizon} ON close_target_{horizon}.ts_code = base.ts_code AND close_target_{horizon}.trade_date = close_calendar_{horizon}.trade_date"));
        joins.push(format!("LEFT JOIN market_calendar exit_calendar_{horizon} ON exit_calendar_{horizon}.day_index = signal_calendar.day_index + 1 + {horizon}"));
        joins.push(format!("LEFT JOIN daily_qfq exit_day_{horizon} ON exit_day_{horizon}.ts_code = base.ts_code AND exit_day_{horizon}.trade_date = exit_calendar_{horizon}.trade_date"));
        columns.push(format!("CASE WHEN close_target_{horizon}.qfq_close > 0 THEN close_target_{horizon}.qfq_close / base.qfq_close - 1 END AS close_to_close_h{horizon}_raw"));
        for (label, price) in [("vwap_to_vwap", "qfq_vwap"), ("twap_to_twap", "qfq_twap")] {
            columns.push(format!("CASE WHEN entry_day.{price} > 0 AND exit_day_{horizon}.{price} > 0 AND entry_day.observation_status = 'complete_trading' AND exit_day_{horizon}.observation_status = 'complete_trading' AND entry_day.amount_cny > 0 AND exit_day_{horizon}.amount_cny > 0 THEN exit_day_{horizon}.{price} / entry_day.{price} - 1 END AS {label}_h{horizon}_raw"));
        }
    }
    let panel_query = format!(
        r#"
        WITH base AS (
          SELECT u.trade_date, u.ts_code, d.qfq_close
          FROM selected_universe u JOIN daily_qfq d USING (trade_date, ts_code)
          WHERE u.trade_date BETWEEN DATE '{start}' AND DATE '{end}' AND d.qfq_close > 0
        )
        SELECT base.trade_date, base.ts_code, {columns}
        FROM base {joins}
    "#,
        columns = columns.join(",\n"),
        joins = joins.join("\n")
    );
    let panel_path = quote_sql(&temporary.join("panel").to_string_lossy());
    conn.execute_batch(&format!("COPY (SELECT *, year(trade_date) AS year FROM ({panel_query})) TO '{panel_path}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year));"))?;

    let mut benchmark_joins = vec![
        "LEFT JOIN market_calendar entry_calendar ON entry_calendar.day_index = signal_calendar.day_index + 1".to_string(),
        "LEFT JOIN index_daily entry_index ON entry_index.index_code = '000905.SH' AND entry_index.trade_date = entry_calendar.trade_date".to_string(),
    ];
    let mut benchmark_columns = Vec::new();
    for horizon in &horizons {
        benchmark_joins.push(format!("LEFT JOIN market_calendar close_calendar_{horizon} ON close_calendar_{horizon}.day_index = signal_calendar.day_index + {horizon}"));
        benchmark_joins.push(format!("LEFT JOIN index_daily close_end_{horizon} ON close_end_{horizon}.index_code = '000905.SH' AND close_end_{horizon}.trade_date = close_calendar_{horizon}.trade_date"));
        benchmark_joins.push(format!("LEFT JOIN market_calendar exit_calendar_{horizon} ON exit_calendar_{horizon}.day_index = signal_calendar.day_index + 1 + {horizon}"));
        benchmark_joins.push(format!("LEFT JOIN index_daily exit_index_{horizon} ON exit_index_{horizon}.index_code = '000905.SH' AND exit_index_{horizon}.trade_date = exit_calendar_{horizon}.trade_date"));
        benchmark_columns.push(format!("CASE WHEN start_index.close > 0 AND close_end_{horizon}.close > 0 THEN close_end_{horizon}.close / start_index.close - 1 END AS close_to_close_h{horizon}_csi500"));
        for label in ["vwap_to_vwap", "twap_to_twap"] {
            benchmark_columns.push(format!("CASE WHEN entry_index.close > 0 AND exit_index_{horizon}.close > 0 THEN exit_index_{horizon}.close / entry_index.close - 1 END AS {label}_h{horizon}_csi500"));
        }
    }
    let benchmark_query = format!(
        r#"
        SELECT signal_calendar.trade_date, {columns}
        FROM market_calendar signal_calendar
        LEFT JOIN index_daily start_index ON start_index.index_code = '000905.SH' AND start_index.trade_date = signal_calendar.trade_date
        {joins}
        WHERE signal_calendar.trade_date BETWEEN DATE '{start}' AND DATE '{end}'
    "#,
        columns = benchmark_columns.join(",\n"),
        joins = benchmark_joins.join("\n")
    );
    copy_query(
        conn,
        &benchmark_query,
        &temporary.join("csi500_benchmark.parquet"),
    )?;
    copy_query(
        conn,
        &format!(
            "SELECT trade_date, count(*)::BIGINT AS universe_count FROM selected_universe WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}' GROUP BY trade_date"
        ),
        &temporary.join("universe_counts.parquet"),
    )?;
    write_json_atomic(&temporary.join("cache_manifest.json"), manifest)?;
    fs::rename(temporary, cache_root)?;
    Ok(())
}

fn ensure_label_cache(
    conn: &Connection,
    batch_root: &Path,
    catalog: &Path,
    universe: &str,
    config: &Config,
    rebuild_cache: bool,
) -> Result<(PathBuf, CacheManifest, bool)> {
    let cache_root = batch_root.join("_market_labels").join(universe);
    let expected = cache_manifest(conn, catalog, universe, config)?;
    let manifest_path = cache_root.join("cache_manifest.json");
    if cache_root.exists() {
        let cached: CacheManifest = fs::read_to_string(&manifest_path)
            .context("read cache manifest")
            .and_then(|text| serde_json::from_str(&text).context("parse cache manifest"))?;
        let complete = cache_root.join("panel").exists()
            && cache_root.join("csi500_benchmark.parquet").exists()
            && cache_root.join("universe_counts.parquet").exists();
        if cached == expected && complete && !rebuild_cache {
            return Ok((cache_root, expected, true));
        }
        if !rebuild_cache {
            bail!(
                "Market-label cache at {} does not match current inputs; rerun with --rebuild-cache",
                cache_root.display()
            );
        }
        fs::remove_dir_all(&cache_root)
            .with_context(|| format!("remove requested cache rebuild {}", cache_root.display()))?;
    }
    fs::create_dir_all(cache_root.parent().unwrap())?;
    build_label_cache(conn, &cache_root, &expected, config)?;
    Ok((cache_root, expected, false))
}

fn install_cache_views(conn: &Connection, cache_root: &Path) -> Result<()> {
    let panel = quote_sql(&cache_root.join("panel/**/*.parquet").to_string_lossy());
    let benchmark = quote_sql(
        &cache_root
            .join("csi500_benchmark.parquet")
            .to_string_lossy(),
    );
    let counts = quote_sql(&cache_root.join("universe_counts.parquet").to_string_lossy());
    conn.execute_batch(&format!(r#"
        CREATE OR REPLACE TEMP VIEW cache_panel AS SELECT * FROM read_parquet('{panel}', hive_partitioning = true);
        CREATE OR REPLACE TEMP VIEW cache_benchmark AS SELECT * FROM read_parquet('{benchmark}');
        CREATE OR REPLACE TEMP VIEW cache_universe_counts AS SELECT * FROM read_parquet('{counts}');
    "#))?;
    Ok(())
}

fn build_cached_ic_and_groups(conn: &Connection, specs: &[LabelSpec], output: &Path) -> Result<()> {
    conn.execute_batch(
        r#"
        DROP TABLE IF EXISTS cached_factor_metrics;
        CREATE TEMP TABLE cached_factor_metrics (
          metric_type VARCHAR, horizon INTEGER, return_kind VARCHAR, trade_date DATE,
          sample_count BIGINT, rank_ic DOUBLE, group_number BIGINT, mean_return DOUBLE
        );
    "#,
    )?;
    for spec in specs {
        let raw_kind = format!("{}_raw", spec.label);
        let excess_kind = format!("{}_excess_csi500", spec.label);
        let returns = format!(
            r#"
            SELECT c.trade_date, c.ts_code, c.factor_value, kind, value
            FROM factor_cohort c
            JOIN cache_panel p USING (trade_date, ts_code)
            LEFT JOIN cache_benchmark b USING (trade_date),
            LATERAL (VALUES
              ('{raw_kind}', p.{raw}),
              ('{excess_kind}', p.{raw} - b.{benchmark})
            ) x(kind, value)
            WHERE value IS NOT NULL AND isfinite(value)
        "#,
            raw = spec.raw_column,
            benchmark = spec.benchmark_column
        );
        conn.execute_batch(&format!(r#"
            INSERT INTO cached_factor_metrics
            WITH returns AS ({returns}), ranked AS (
              SELECT *,
                rank() OVER (PARTITION BY trade_date, kind ORDER BY factor_value, ts_code) AS factor_rank,
                rank() OVER (PARTITION BY trade_date, kind ORDER BY value, ts_code) AS return_rank,
                ntile(10) OVER (PARTITION BY trade_date, kind ORDER BY factor_value, ts_code) AS group_number
              FROM returns
            )
            SELECT CASE WHEN group_number IS NULL THEN 'ic' ELSE 'group' END AS metric_type,
              {horizon}::INTEGER AS horizon, kind AS return_kind, trade_date, count(*)::BIGINT AS sample_count,
              CASE WHEN group_number IS NULL THEN corr(return_rank, factor_rank) END AS rank_ic,
              group_number, CASE WHEN group_number IS NOT NULL THEN avg(value) END AS mean_return
            FROM ranked
            GROUP BY GROUPING SETS ((trade_date, kind), (trade_date, kind, group_number))
        "#, horizon = spec.horizon))?;
    }
    copy_query(
        conn,
        "SELECT horizon, return_kind, trade_date, sample_count, NULL::DOUBLE AS pearson_ic, rank_ic, NULL::DOUBLE AS winsorized_pearson_ic FROM cached_factor_metrics WHERE metric_type = 'ic'",
        &output.join("daily_ic.parquet"),
    )?;
    copy_query(
        conn,
        "SELECT horizon, return_kind, trade_date, group_number, mean_return, sample_count FROM cached_factor_metrics WHERE metric_type = 'group'",
        &output.join("group_returns.parquet"),
    )?;
    let ic_path = quote_sql(&output.join("daily_ic.parquet").to_string_lossy());
    copy_query(
        conn,
        &format!(
            r#"
        SELECT horizon, return_kind, year(trade_date) AS calendar_year, count(*) AS days,
          avg(rank_ic) AS mean_rank_ic, stddev_samp(rank_ic) AS std_rank_ic,
          avg(rank_ic) / nullif(stddev_samp(rank_ic), 0) * sqrt(252) AS rank_icir,
          avg(CASE WHEN rank_ic > 0 THEN 1.0 ELSE 0.0 END) AS positive_rank_ic_ratio
        FROM read_parquet('{ic_path}') WHERE isfinite(rank_ic)
        GROUP BY horizon, return_kind, calendar_year
    "#
        ),
        &output.join("annual_metrics.parquet"),
    )?;
    conn.execute_batch("DROP TABLE IF EXISTS cached_factor_metrics;")?;
    Ok(())
}

fn evaluate_cached_factor(
    conn: &Connection,
    factor: &Path,
    catalog: &Path,
    universe: &str,
    config: &Config,
    cache_manifest: &CacheManifest,
    output: &Path,
) -> Result<()> {
    fs::create_dir_all(output)?;
    let factor_path = quote_sql(&factor.canonicalize()?.to_string_lossy());
    conn.execute_batch("DROP TABLE IF EXISTS factor_cohort;")?;
    conn.execute_batch(&format!(
        r#"
        CREATE TEMP TABLE factor_cohort AS
        SELECT f.trade_date, f.ts_code, f.factor_value
        FROM read_parquet('{factor_path}') f
        JOIN cache_panel p USING (trade_date, ts_code)
        WHERE isfinite(f.factor_value);
    "#
    ))?;
    let rows = scalar_i64(conn, "SELECT count(*) FROM factor_cohort")?;
    if rows == 0 {
        bail!("No factor rows remain after cache universe/date filtering");
    }
    let horizons = config_horizons(config);
    build_cached_ic_and_groups(conn, &label_specs(&horizons), output)?;
    let daily_ic_rows = scalar_i64(
        conn,
        &format!(
            "SELECT count(*) FROM read_parquet('{}')",
            quote_sql(&output.join("daily_ic.parquet").to_string_lossy())
        ),
    )?;
    let summary = Summary {
        factor_id: factor_id(factor), factor_path: factor.canonicalize()?.display().to_string(), catalog: catalog.canonicalize()?.display().to_string(),
        universe: universe.to_string(), start: config.start.clone().unwrap_or_else(|| "dataset minimum".to_string()), end: config.end.clone().unwrap_or_else(|| "dataset maximum".to_string()),
        horizons, transaction_cost_bps: config.transaction_cost_bps.unwrap_or(0.0),
        portfolio_direction: "not_applicable_batch_factor_prediction_report".to_string(), rows, daily_ic_rows, portfolio_rows: 0,
        note: "Batch-factor predictive report only: market labels were cached once for this universe. Close-to-close is a research label. VWAP-to-VWAP and TWAP-to-TWAP enter on T+1 and exit after the stated market-session horizon. All excess-return diagnostics use matching CSI500 close-to-close benchmark periods.".to_string(),
    };
    fs::write(
        output.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    fs::write(
        output.join("manifest.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "engine": env!("CARGO_PKG_VERSION"), "evaluation_mode": "batch-factor-eval-v2", "summary": summary,
            "cache_schema_version": cache_manifest.label_schema_version, "cache_universe": cache_manifest.universe,
        }))?,
    )?;
    conn.execute_batch("DROP TABLE IF EXISTS factor_cohort;")?;
    Ok(())
}

fn build_factor_diagnostics(conn: &Connection, output: &Path) -> Result<()> {
    copy_query(
        conn,
        r#"
        WITH factor_bounds AS (
          SELECT trade_date,
            quantile_cont(factor_value, 0.01) AS factor_p01,
            quantile_cont(factor_value, 0.99) AS factor_p99
          FROM cohort GROUP BY trade_date
        ), clipped AS (
          SELECT c.trade_date, c.ts_code,
            least(greatest(c.factor_value, b.factor_p01), b.factor_p99) AS factor_value
          FROM cohort c JOIN factor_bounds b USING (trade_date)
        ), previous_calendar AS (
          SELECT trade_date, lag(trade_date) OVER (ORDER BY trade_date) AS previous_trade_date
          FROM market_calendar
        )
        SELECT u.trade_date,
          count(*) AS universe_count,
          count(clipped.ts_code) AS factor_count,
          count(clipped.ts_code)::DOUBLE / nullif(count(*), 0) AS coverage_ratio,
          max(bounds.factor_p01) AS factor_p01,
          max(bounds.factor_p99) AS factor_p99,
          avg(clipped.factor_value) AS factor_mean,
          NULL::DOUBLE AS factor_std,
          NULL::DOUBLE AS factor_autocorrelation
        FROM selected_universe u
        LEFT JOIN clipped USING (trade_date, ts_code)
        LEFT JOIN factor_bounds bounds USING (trade_date)
        LEFT JOIN previous_calendar calendar USING (trade_date)
        LEFT JOIN clipped previous ON previous.ts_code = clipped.ts_code AND previous.trade_date = calendar.previous_trade_date
        GROUP BY u.trade_date
    "#,
        &output.join("factor_diagnostics.parquet"),
    )
}

fn load_signals(conn: &Connection) -> Result<BTreeMap<String, Vec<Signal>>> {
    let mut statement = conn.prepare(
        r#"
        SELECT c.trade_date::VARCHAR, execution_day.trade_date::VARCHAR, c.ts_code, c.factor_value,
          d.qfq_open, d.qfq_vwap,
          coalesce(d.observation_status = 'complete_trading' AND d.amount_cny > 0, false)
        FROM cohort c
        JOIN market_calendar day ON day.trade_date = c.trade_date
        JOIN market_calendar execution_day ON execution_day.day_index = day.day_index + 1
        LEFT JOIN daily_qfq d ON d.ts_code = c.ts_code AND d.trade_date = execution_day.trade_date
        ORDER BY c.trade_date, c.factor_value DESC, c.ts_code
    "#,
    )?;
    let rows = statement.query_map([], |row: &Row| {
        Ok((
            row.get::<_, String>(0)?,
            Signal {
                execution_date: row.get(1)?,
                code: row.get(2)?,
                open: row.get(4)?,
                vwap: row.get(5)?,
                tradable: row.get(6)?,
            },
        ))
    })?;
    let mut by_day: BTreeMap<String, Vec<Signal>> = BTreeMap::new();
    for row in rows {
        let (day, signal) = row?;
        by_day.entry(day).or_default().push(signal);
    }
    Ok(by_day)
}

fn price(signal: &Signal, mode: &str) -> Option<f64> {
    match mode {
        "open" => signal.open,
        "vwap" => signal.vwap,
        _ => None,
    }
    .filter(|v| v.is_finite() && *v > 0.0)
}

fn simulate_mode(
    signals: &BTreeMap<String, Vec<Signal>>,
    mode: &str,
    cost_rate: f64,
    select_high: bool,
) -> (Vec<PortfolioRow>, Vec<HoldingRow>) {
    let dates: Vec<String> = signals.keys().cloned().collect();
    let mut weights: HashMap<String, f64> = HashMap::new();
    let mut nav = 1.0;
    let mut rows = Vec::new();
    let mut holdings = Vec::new();
    for index in 0..dates.len().saturating_sub(1) {
        let date = &dates[index];
        let current = &signals[date];
        let next = &signals[&dates[index + 1]];
        let current_map: HashMap<&str, &Signal> =
            current.iter().map(|s| (s.code.as_str(), s)).collect();
        let next_map: HashMap<&str, &Signal> = next.iter().map(|s| (s.code.as_str(), s)).collect();
        let mut frozen = HashMap::new();
        for (code, weight) in &weights {
            if !current_map
                .get(code.as_str())
                .map(|s| s.tradable && price(s, mode).is_some())
                .unwrap_or(false)
            {
                frozen.insert(code.clone(), *weight);
            }
        }
        let frozen_count = frozen.len();
        let frozen_weight: f64 = frozen.values().sum();
        let n = ((current.len() as f64) * 0.10).ceil().max(1.0) as usize;
        let selected: Vec<&Signal> = if select_high {
            current
                .iter()
                .take(n)
                .filter(|s| s.tradable && price(s, mode).is_some())
                .collect()
        } else {
            current
                .iter()
                .rev()
                .take(n)
                .filter(|s| s.tradable && price(s, mode).is_some())
                .collect()
        };
        let allocation = if selected.is_empty() {
            0.0
        } else {
            (1.0 - frozen_weight).max(0.0) / selected.len() as f64
        };
        let mut next_weights = frozen;
        for signal in selected {
            next_weights.insert(signal.code.clone(), allocation);
        }
        let turnover: f64 = weights
            .iter()
            .map(|(code, old)| (next_weights.get(code).copied().unwrap_or(0.0) - old).abs())
            .sum::<f64>()
            + next_weights
                .iter()
                .filter(|(code, _)| !weights.contains_key(*code))
                .map(|(_, new)| new.abs())
                .sum::<f64>();
        let cost = turnover * cost_rate;
        let mut gross_return = 0.0;
        let mut stale = 0_usize;
        for (code, weight) in &next_weights {
            let current_price = current_map.get(code.as_str()).and_then(|s| price(s, mode));
            let next_price = next_map.get(code.as_str()).and_then(|s| price(s, mode));
            if let (Some(p0), Some(p1)) = (current_price, next_price) {
                gross_return += weight * (p1 / p0 - 1.0);
            } else {
                stale += 1;
            }
        }
        let net_return = gross_return - cost;
        nav *= 1.0 + net_return;
        let denominator = (1.0 + net_return).max(1e-12);
        for (code, weight) in &mut next_weights {
            let ret = match (
                current_map.get(code.as_str()).and_then(|s| price(s, mode)),
                next_map.get(code.as_str()).and_then(|s| price(s, mode)),
            ) {
                (Some(p0), Some(p1)) => p1 / p0 - 1.0,
                _ => 0.0,
            };
            *weight = *weight * (1.0 + ret) / denominator;
        }
        let execution_date = current
            .first()
            .map(|signal| signal.execution_date.clone())
            .unwrap_or_else(|| date.clone());
        for (code, weight) in &next_weights {
            holdings.push(HoldingRow {
                execution_mode: mode.to_string(),
                execution_date: execution_date.clone(),
                ts_code: code.clone(),
                weight: *weight,
            });
        }
        rows.push(PortfolioRow {
            execution_mode: mode.to_string(),
            execution_date,
            gross_return,
            turnover,
            transaction_cost: cost,
            net_return,
            nav,
            holding_count: next_weights.len(),
            frozen_holding_count: frozen_count,
            stale_price_count: stale,
        });
        weights = next_weights;
    }
    (rows, holdings)
}

fn write_tsv<T: Serialize>(rows: &[T], path: &Path) -> Result<()> {
    let mut output = String::new();
    if let Some(first) = rows.first() {
        let value = serde_json::to_value(first)?;
        let object = value
            .as_object()
            .context("serialized row is not an object")?;
        let columns: Vec<&String> = object.keys().collect();
        output.push_str(
            &columns
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\t"),
        );
        output.push('\n');
        for row in rows {
            let value = serde_json::to_value(row)?;
            let object = value.as_object().unwrap();
            output.push_str(
                &columns
                    .iter()
                    .map(|key| object[*key].to_string().trim_matches('"').to_string())
                    .collect::<Vec<_>>()
                    .join("\t"),
            );
            output.push('\n');
        }
    }
    fs::write(path, output)?;
    Ok(())
}

fn portfolio_direction(
    conn: &Connection,
    output: &Path,
    requested: Option<&str>,
) -> Result<(bool, String)> {
    match requested.unwrap_or("auto_full_sample_rank_ic") {
        "high" => Ok((true, "high_factor_g10".to_string())),
        "low" => Ok((false, "low_factor_g1".to_string())),
        "auto_full_sample_rank_ic" => {
            let path = quote_sql(&output.join("daily_ic.parquet").to_string_lossy());
            let mean_rank_ic: Option<f64> = conn.query_row(
                &format!("SELECT avg(rank_ic) FROM read_parquet('{path}') WHERE horizon = 1 AND return_kind = 'raw'"),
                [], |row| row.get(0),
            )?;
            if mean_rank_ic.unwrap_or(0.0) >= 0.0 {
                Ok((
                    true,
                    "auto_high_factor_g10_from_full_sample_rank_ic".to_string(),
                ))
            } else {
                Ok((
                    false,
                    "auto_low_factor_g1_from_full_sample_rank_ic".to_string(),
                ))
            }
        }
        other => bail!(
            "Unsupported portfolio_direction '{other}'. Use high, low, or auto_full_sample_rank_ic"
        ),
    }
}

fn build_portfolio(
    conn: &Connection,
    output: &Path,
    cost_bps: f64,
    select_high: bool,
) -> Result<i64> {
    let signals = load_signals(conn)?;
    let (mut portfolio, mut holdings) =
        simulate_mode(&signals, "open", cost_bps / 10_000.0, select_high);
    let (portfolio_vwap, holdings_vwap) =
        simulate_mode(&signals, "vwap", cost_bps / 10_000.0, select_high);
    portfolio.extend(portfolio_vwap);
    holdings.extend(holdings_vwap);
    let temp = output.join(".portfolio.tsv");
    let holdings_temp = output.join(".holdings.tsv");
    write_tsv(&portfolio, &temp)?;
    write_tsv(&holdings, &holdings_temp)?;
    let p = quote_sql(&temp.to_string_lossy());
    let h = quote_sql(&holdings_temp.to_string_lossy());
    copy_query(
        conn,
        &format!("SELECT * FROM read_csv('{p}', delim='\\t', header=true)"),
        &output.join("portfolio_daily.parquet"),
    )?;
    copy_query(
        conn,
        &format!("SELECT * FROM read_csv('{h}', delim='\\t', header=true)"),
        &output.join("holdings.parquet"),
    )?;
    let _ = fs::remove_file(temp);
    let _ = fs::remove_file(holdings_temp);
    Ok(portfolio.len() as i64)
}

fn run_factor_eval(args: FactorEvalArgs) -> Result<PathBuf> {
    let config: Config =
        serde_yaml::from_str(&fs::read_to_string(&args.config).context("read config")?)?;
    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let output = args.output.join(&run_id);
    fs::create_dir_all(&output)?;
    let conn = Connection::open_with_flags(
        &args.catalog,
        DuckDbConfig::default().access_mode(AccessMode::ReadOnly)?,
    )
    .context("open DuckDB catalog read-only")?;
    let temp_directory = output.join(".duckdb_tmp");
    fs::create_dir_all(&temp_directory)?;
    conn.execute_batch(&format!("SET threads TO {}; SET memory_limit='{}MB'; SET preserve_insertion_order=false; SET temp_directory='{}';", config.engine_threads.unwrap_or(1), config.memory_limit_mb.unwrap_or(3000), quote_sql(&temp_directory.to_string_lossy())))?;
    prepare_views(&conn, &args, &config)?;
    let horizons = config
        .horizons
        .clone()
        .unwrap_or_else(|| vec![1, 5, 10, 20]);
    build_ic_and_groups(&conn, &horizons, &output)?;
    build_factor_diagnostics(&conn, &output)?;
    let rows = scalar_i64(&conn, "SELECT count(*) FROM cohort")?;
    let daily_ic_rows = scalar_i64(
        &conn,
        &format!(
            "SELECT count(*) FROM read_parquet('{}')",
            quote_sql(&output.join("daily_ic.parquet").to_string_lossy())
        ),
    )?;
    copy_query(
        &conn,
        r#"
                SELECT 'factor_rows' AS check_name, count(*)::BIGINT AS value, 'cohort after universe/date filters' AS detail FROM cohort
                UNION ALL SELECT 'nonfinite_factor_rows', count(*)::BIGINT, 'must be zero' FROM cohort WHERE NOT isfinite(factor_value)
            "#,
        &output.join("diagnostics.parquet"),
    )?;
    let summary = Summary { factor_id: factor_id(&args.factor), factor_path: args.factor.canonicalize()?.display().to_string(), catalog: args.catalog.canonicalize()?.display().to_string(), universe: config.universe.unwrap_or_else(|| "all".to_string()), start: config.start.unwrap_or_else(|| "dataset minimum".to_string()), end: config.end.unwrap_or_else(|| "dataset maximum".to_string()), horizons, transaction_cost_bps: config.transaction_cost_bps.unwrap_or(0.0), portfolio_direction: "not_applicable_single_factor_prediction_report".to_string(), rows, daily_ic_rows, portfolio_rows: 0, note: "Single-factor predictive report only: no optimizer, holdings, portfolio NAV, transaction-cost deduction, or automatic direction selection. Close-to-close is a research label. VWAP-to-VWAP and TWAP-to-TWAP labels enter on T+1 and exit after the stated market-session horizon. All excess-return diagnostics use CSI500 close-to-close over the matching label period.".to_string() };
    fs::write(
        output.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    fs::write(
        output.join("manifest.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"run_id": run_id, "engine": env!("CARGO_PKG_VERSION"), "summary": summary}),
        )?,
    )?;
    Ok(output)
}

#[derive(Serialize)]
struct BatchTaskStatus {
    factor_id: String,
    universe: String,
    status: String,
    output: String,
    elapsed_seconds: f64,
    error: Option<String>,
}

#[derive(Serialize)]
struct CacheTiming {
    universe: String,
    cache_reused: bool,
    elapsed_seconds: f64,
    error: Option<String>,
}

fn discover_factors(root: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            discover_factors(&path, found)?;
        } else if path
            .file_name()
            .is_some_and(|name| name == "factor.parquet")
            && path
                .parent()
                .is_some_and(|parent| parent.join("manifest.json").exists())
        {
            found.push(path);
        }
    }
    Ok(())
}

fn batch_result_is_complete(output: &Path) -> bool {
    if !output.join("summary.json").exists() {
        return false;
    }
    fs::read_to_string(output.join("manifest.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|manifest| {
            manifest["evaluation_mode"]
                .as_str()
                .map(|value| value == "batch-factor-eval-v2")
        })
        .unwrap_or(false)
}

fn stage_path(batch_root: &Path, factor_id: &str, universe: &str) -> PathBuf {
    batch_root.join("_staging").join(factor_id).join(format!(
        "{universe}-{}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ))
}

fn run_batch_factor_eval(args: BatchFactorEvalArgs) -> Result<PathBuf> {
    let batch_started = Instant::now();
    let batch_id = args
        .batch_id
        .clone()
        .unwrap_or_else(|| chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let batch_root = args.output.join(&batch_id);
    fs::create_dir_all(&batch_root)?;
    let mut factors = Vec::new();
    discover_factors(&args.factor_root, &mut factors)?;
    factors.sort();
    if factors.is_empty() {
        bail!(
            "No factor.parquet files with manifests under {}",
            args.factor_root.display()
        );
    }
    let all_configs = vec![
        ("csi300", args.csi300_config.clone()),
        ("csi500", args.csi500_config.clone()),
        ("all", args.all_config.clone()),
    ];
    let configs = all_configs
        .into_iter()
        .filter(|(universe, _)| args.universes.iter().any(|requested| requested == universe))
        .collect::<Vec<_>>();
    if configs.len() != args.universes.len() {
        bail!("Unsupported universe in --universes. Use csi300, csi500, or all");
    }
    let selected_factors = factors
        .into_iter()
        .map(|factor| (factor_id(&factor), factor))
        .collect::<Vec<_>>();
    let requested_tasks = args.tasks.clone().unwrap_or_default();
    let task_count = selected_factors
        .iter()
        .flat_map(|(id, _)| {
            configs
                .iter()
                .map(move |(universe, _)| format!("{id}/{universe}"))
        })
        .filter(|task| requested_tasks.is_empty() || requested_tasks.contains(task))
        .count();
    if !requested_tasks.is_empty() && task_count != requested_tasks.len() {
        bail!("At least one --tasks entry did not match a discovered factor and selected universe");
    }
    write_json_atomic(
        &batch_root.join("batch_manifest.json"),
        &serde_json::json!({
            "batch_id": batch_id, "catalog": args.catalog, "factor_root": args.factor_root, "tasks": task_count,
            "execution": "batch-factor-eval-v2: one market-label cache and one DuckDB connection per universe; factors evaluated sequentially",
            "universes": args.universes,
        }),
    )?;

    let mut statuses = Vec::<BatchTaskStatus>::new();
    let mut cache_timings = Vec::<CacheTiming>::new();
    let mut failed = false;
    for (universe, config_path) in configs {
        let config: Config = serde_yaml::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("read config {}", config_path.display()))?,
        )?;
        if config.universe.as_deref().unwrap_or("all") != universe {
            bail!(
                "Config {} declares a different universe than requested {universe}",
                config_path.display()
            );
        }
        let started_cache = Instant::now();
        let conn = Connection::open_with_flags(
            &args.catalog,
            DuckDbConfig::default().access_mode(AccessMode::ReadOnly)?,
        )
        .context("open DuckDB catalog read-only")?;
        set_duckdb_options(
            &conn,
            &config,
            &batch_root.join("_duckdb_tmp").join(universe),
        )?;
        let cache = ensure_label_cache(
            &conn,
            &batch_root,
            &args.catalog,
            universe,
            &config,
            args.rebuild_cache,
        );
        let (cache_root, cache_manifest, cache_reused) = match cache {
            Ok(cache) => cache,
            Err(error) => {
                failed = true;
                cache_timings.push(CacheTiming {
                    universe: universe.to_string(),
                    cache_reused: false,
                    elapsed_seconds: started_cache.elapsed().as_secs_f64(),
                    error: Some(format!("{error:#}")),
                });
                for (id, _) in &selected_factors {
                    let task = format!("{id}/{universe}");
                    if !requested_tasks.is_empty() && !requested_tasks.contains(&task) {
                        continue;
                    }
                    statuses.push(BatchTaskStatus {
                        factor_id: id.clone(),
                        universe: universe.to_string(),
                        status: "failed".to_string(),
                        output: batch_root.join(id).join(universe).display().to_string(),
                        elapsed_seconds: started_cache.elapsed().as_secs_f64(),
                        error: Some(format!("market-label cache: {error:#}")),
                    });
                }
                write_json_atomic(&batch_root.join("task_status.json"), &statuses)?;
                continue;
            }
        };
        cache_timings.push(CacheTiming {
            universe: universe.to_string(),
            cache_reused,
            elapsed_seconds: started_cache.elapsed().as_secs_f64(),
            error: None,
        });
        install_cache_views(&conn, &cache_root)?;
        for (id, factor) in &selected_factors {
            let task = format!("{id}/{universe}");
            if !requested_tasks.is_empty() && !requested_tasks.contains(&task) {
                continue;
            }
            let output = batch_root.join(id).join(universe);
            if cache_reused && !args.rebuild_cache && batch_result_is_complete(&output) {
                statuses.push(BatchTaskStatus {
                    factor_id: id.clone(),
                    universe: universe.to_string(),
                    status: "skipped".to_string(),
                    output: output.display().to_string(),
                    elapsed_seconds: 0.0,
                    error: None,
                });
                write_json_atomic(&batch_root.join("task_status.json"), &statuses)?;
                continue;
            }
            let started = Instant::now();
            let stage = stage_path(&batch_root, id, universe);
            let result = evaluate_cached_factor(
                &conn,
                factor,
                &args.catalog,
                universe,
                &config,
                &cache_manifest,
                &stage,
            );
            match result {
                Ok(()) => {
                    if output.exists() {
                        fs::remove_dir_all(&output).with_context(|| {
                            format!(
                                "replace incomplete or stale task output {}",
                                output.display()
                            )
                        })?;
                    }
                    fs::create_dir_all(output.parent().unwrap())?;
                    fs::rename(&stage, &output)?;
                    statuses.push(BatchTaskStatus {
                        factor_id: id.clone(),
                        universe: universe.to_string(),
                        status: "ok".to_string(),
                        output: output.display().to_string(),
                        elapsed_seconds: started.elapsed().as_secs_f64(),
                        error: None,
                    });
                }
                Err(error) => {
                    failed = true;
                    statuses.push(BatchTaskStatus {
                        factor_id: id.clone(),
                        universe: universe.to_string(),
                        status: "failed".to_string(),
                        output: output.display().to_string(),
                        elapsed_seconds: started.elapsed().as_secs_f64(),
                        error: Some(format!("{error:#}")),
                    });
                }
            }
            write_json_atomic(&batch_root.join("task_status.json"), &statuses)?;
        }
    }
    write_json_atomic(
        &batch_root.join("batch_manifest.json"),
        &serde_json::json!({
            "batch_id": batch_id, "catalog": args.catalog, "factor_root": args.factor_root, "tasks": task_count,
            "execution": "batch-factor-eval-v2: one market-label cache and one DuckDB connection per universe; factors evaluated sequentially",
            "universes": args.universes, "cache_timings": cache_timings,
            "total_elapsed_seconds": batch_started.elapsed().as_secs_f64(), "has_failures": failed,
        }),
    )?;
    if failed {
        bail!(
            "Batch completed with failures; inspect {}",
            batch_root.join("task_status.json").display()
        );
    }
    Ok(batch_root)
}

fn legacy_batch_args(args: BatchEvalLegacyArgs) -> Result<BatchFactorEvalArgs> {
    if args.jobs > 1 {
        bail!(
            "batch-eval no longer supports factor-level --jobs > 1; use batch-factor-eval and configure DuckDB engine_threads instead"
        );
    }
    Ok(BatchFactorEvalArgs {
        catalog: args.catalog,
        factor_root: args.factor_root,
        output: args.output,
        batch_id: args.batch_id,
        universes: args.universes,
        tasks: args.tasks,
        csi300_config: args.csi300_config,
        csi500_config: args.csi500_config,
        all_config: args.all_config,
        rebuild_cache: args.rebuild_cache,
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let output = match cli.command {
        Command::FactorEval(args) => run_factor_eval(args)?,
        Command::BatchFactorEval(args) => run_batch_factor_eval(args)?,
        Command::BatchEval(args) => run_batch_factor_eval(legacy_batch_args(args)?)?,
    };
    println!("{}", output.display());
    Ok(())
}
