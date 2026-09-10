//! quant-minute-factor: minute-bar daily wide-factor production.
//!
//! Reads per-trade-date minute parquet partitions, computes the 24 core factors
//! per stock-day, crops to the dynamic CSI300 ∪ CSI500 universe, and writes one
//! wide parquet per trade date under `<output>/year=YYYY/`.  Blocks of
//! `--block-days` market days (plus 20 warm-up days) run in parallel; each block
//! owns an independent rolling state, so block layout and thread count cannot
//! change any output value.
mod candidates;
mod candidates_v3;
mod daily;
mod loader;
mod manifest;
mod pipeline;
mod schema;
mod state;
mod writer;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use manifest::{BuildParameters, DayStatus, Manifest};
use pipeline::{plan_blocks, process_day, Block, MarketContext};

#[derive(Parser)]
#[command(
    name = "quant-minute-factor",
    about = "Minute-bar daily wide-factor production"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Produce the core24 wide dataset over a market-date range.
    Build(BuildArgs),
}

#[derive(Parser)]
pub(crate) struct BuildArgs {
    /// DuckDB catalog with observed_calendar and index_trading_universe.
    #[arg(long)]
    pub(crate) catalog: PathBuf,
    /// Per-trade-date minute parquet root (year=/month=/trade_date= layout).
    #[arg(long, default_value = "A_stock_database/lake/canonical/minute")]
    pub(crate) minute_root: PathBuf,
    /// Output root for the wide dataset and its manifest.
    #[arg(long)]
    pub(crate) output: PathBuf,
    #[arg(long)]
    pub(crate) start: String,
    #[arg(long)]
    pub(crate) end: String,
    #[arg(long, default_value_t = 60)]
    pub(crate) block_days: usize,
    #[arg(long, default_value_t = 4)]
    pub(crate) jobs: usize,
    #[arg(long, default_value_t = 2)]
    pub(crate) threads_per_job: usize,
    #[arg(long, default_value_t = 12_000)]
    pub(crate) memory_limit_mb: usize,
    /// Recompute the requested range even when the manifest marks it complete.
    #[arg(long)]
    pub(crate) replace: bool,
    /// Factor registry to produce: core24 or ohlcv_candidates_v1.
    #[arg(long, default_value = "core24")]
    pub(crate) factor_set: String,
}

fn day_file(output: &Path, trade_date: &str) -> PathBuf {
    output
        .join(format!("year={}", &trade_date[..4]))
        .join(format!("{trade_date}.parquet"))
}

fn load_manifest_or_new(
    output: &Path,
    parameters: BuildParameters,
    sources: BTreeMap<String, String>,
) -> Result<Manifest> {
    let path = output.join("manifest.json");
    if !path.exists() {
        return Ok(Manifest::new(parameters, sources));
    }
    let existing: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&path).context("read existing manifest")?)
            .context("parse existing manifest")?;
    if existing.version != manifest::MANIFEST_VERSION || existing.factor_set != "core24" {
        bail!(
            "Existing manifest at {} is not compatible (version {}, factor_set {}); remove it or choose another --output",
            path.display(),
            existing.version,
            existing.factor_set
        );
    }
    // Re-derive the descriptive sections from the current run; the per-date
    // status map is what enables resume.
    let mut manifest = Manifest::new(parameters, sources);
    manifest.dates = existing.dates;
    Ok(manifest)
}

fn run_build(args: BuildArgs) -> Result<PathBuf> {
    if args.factor_set == "ohlcv_candidates_v1" {
        return candidates::run(args);
    }
    if args.factor_set == "ohlcv_candidates_v3" {
        return candidates_v3::run(args);
    }
    if args.factor_set != "core24" {
        bail!(
            "unknown --factor-set {}; expected core24, ohlcv_candidates_v1, or ohlcv_candidates_v3",
            args.factor_set
        );
    }
    if args.block_days == 0 || args.jobs == 0 || args.threads_per_job == 0 {
        bail!("block_days, jobs, and threads_per_job must be positive");
    }
    if args.start > args.end {
        bail!("--start must not exceed --end");
    }
    std::fs::create_dir_all(args.output.join("_staging"))?;
    let catalog = args
        .catalog
        .canonicalize()
        .with_context(|| format!("canonicalize catalog {}", args.catalog.display()))?;
    let minute_root = args
        .minute_root
        .canonicalize()
        .with_context(|| format!("canonicalize minute root {}", args.minute_root.display()))?;
    let output = std::fs::canonicalize(&args.output).unwrap_or_else(|_| args.output.clone());

    let parameters = manifest::default_parameters(
        args.block_days,
        args.jobs,
        args.threads_per_job,
        args.memory_limit_mb,
    );
    let mut sources = BTreeMap::new();
    sources.insert("catalog".to_string(), catalog.display().to_string());
    sources.insert("minute_root".to_string(), minute_root.display().to_string());
    sources.insert("start".to_string(), args.start.clone());
    sources.insert("end".to_string(), args.end.clone());

    let started = std::time::Instant::now();
    // One planning load; the connection closes here and blocks never touch
    // DuckDB again, so blocks share only immutable inputs.
    let context = Arc::new(pipeline::load_market_context(
        &catalog,
        &args.start,
        &args.end,
        args.memory_limit_mb,
    )?);
    let blocks = plan_blocks(
        &context.calendar,
        &args.start,
        &args.end,
        args.block_days,
        parameters.warmup_days,
    );
    if blocks.is_empty() {
        bail!(
            "No requested dates fall on observed market days between {} and {}",
            args.start,
            args.end
        );
    }

    let manifest_path = output.join("manifest.json");
    let manifest = Arc::new(Mutex::new(load_manifest_or_new(
        &output, parameters, sources,
    )?));
    if args.replace {
        let mut guard = manifest.lock().expect("manifest mutex poisoned");
        for block in &blocks {
            for day in block.target_begin..=block.target_end {
                guard.dates.remove(&context.calendar[day]);
            }
        }
        guard.save(&manifest_path)?;
    }

    let next_block = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let blocks = Arc::new(blocks);
    let failures = Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for _ in 0..args.jobs {
            let context = Arc::clone(&context);
            let manifest = Arc::clone(&manifest);
            let next_block = Arc::clone(&next_block);
            let blocks = Arc::clone(&blocks);
            let output = output.clone();
            let minute_root = minute_root.clone();
            let failures = &failures;
            scope.spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(args.threads_per_job)
                    .build()
                    .expect("build per-job rayon pool");
                loop {
                    let block_index = next_block.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if block_index >= blocks.len() {
                        return;
                    }
                    let block = &blocks[block_index];
                    if let Err(error) =
                        run_block(&pool, &context, &manifest, &minute_root, &output, block)
                    {
                        let guard = manifest.lock().expect("manifest mutex poisoned");
                        guard.save(&output.join("manifest.json")).ok();
                        drop(guard);
                        failures
                            .lock()
                            .expect("failures mutex poisoned")
                            .push(format!(
                                "block {} ({}..{}): {error:#}",
                                block_index,
                                context.calendar[block.target_begin],
                                context.calendar[block.target_end],
                            ));
                    }
                }
            });
        }
    });

    let failures = failures.into_inner().expect("failures mutex poisoned");
    let (completed, failed, missing, warmups, no_universe, rows_total) = {
        let guard = manifest.lock().expect("manifest mutex poisoned");
        let mut counts = (0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0_usize);
        for status in guard.dates.values() {
            match status.status.as_str() {
                "ok" => {
                    counts.0 += 1;
                    counts.5 += status.rows;
                }
                "warmup" => counts.3 += 1,
                "no_universe" => counts.4 += 1,
                "missing_partition" => counts.2 += 1,
                _ => counts.1 += 1,
            }
        }
        counts
    };
    {
        let guard = manifest.lock().expect("manifest mutex poisoned");
        guard.save(&manifest_path)?;
    }
    let summary = serde_json::json!({
        "output": output.display().to_string(),
        "blocks": blocks.len(),
        "dates_ok": completed,
        "dates_warmup": warmups,
        "dates_no_universe": no_universe,
        "dates_failed": failed,
        "dates_missing_partition": missing,
        "rows_written": rows_total,
        "total_seconds": started.elapsed().as_secs_f64(),
        "failures": failures,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !failures.is_empty() {
        bail!(
            "build finished with failures; inspect {}",
            manifest_path.display()
        );
    }
    Ok(output)
}

fn run_block(
    pool: &rayon::ThreadPool,
    context: &Arc<MarketContext>,
    manifest: &Arc<Mutex<Manifest>>,
    minute_root: &Path,
    output: &Path,
    block: &Block,
) -> Result<()> {
    let mut state = state::RollingState::new();
    for day in block.warmup_begin..=block.target_end {
        let trade_date = context.calendar[day].clone();
        let write_output = day >= block.target_begin;
        // The point-in-time universe begins at the first month-end constituent
        // snapshot, so early requested dates may predate it: they advance the
        // rolling state but produce no output and are recorded as no_universe.
        let universe_ok = !write_output || context.universe.contains_key(&trade_date);
        // A completed day is replayed (state only) rather than skipped: its
        // history must still feed the rolling baselines of the days after it,
        // so an interrupted-then-resumed block matches a fresh run exactly.
        let already_done = {
            let guard = manifest.lock().expect("manifest mutex poisoned");
            write_output
                && universe_ok
                && guard.date_complete(&trade_date) == Some(true)
                && day_file(output, &trade_date).is_file()
        };
        let mode = if write_output && universe_ok && !already_done {
            pipeline::DayMode::Produce
        } else {
            pipeline::DayMode::Replay
        };
        let outcome = process_day(
            minute_root,
            output,
            context,
            &mut state,
            pool,
            &trade_date,
            context.day_index(&trade_date),
            mode,
        )?;
        if already_done {
            continue;
        }
        let status = if !write_output {
            "warmup".to_string()
        } else if !universe_ok {
            "no_universe".to_string()
        } else {
            outcome.status
        };
        {
            let mut guard = manifest.lock().expect("manifest mutex poisoned");
            // Non-producing passes must never clobber a completed day: with
            // parallel blocks, one block's warm-up replay can race another
            // block's production of the same calendar day, and the ok record
            // plus its file are the authoritative state.
            if mode == pipeline::DayMode::Replay
                && guard.date_complete(&trade_date) == Some(true)
                && (!write_output || day_file(output, &trade_date).is_file())
            {
                continue;
            }
            guard.record(
                &trade_date,
                DayStatus {
                    status,
                    rows: outcome.rows,
                    excluded: outcome.excluded,
                    elapsed_seconds: outcome.elapsed_seconds,
                },
            );
            guard.save(&output.join("manifest.json"))?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build(args) => {
            let output = run_build(args)?;
            println!("{}", output.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FACTOR_NAMES, N_FACTORS, SESSION_BARS};
    use arrow::array::{Array as _, Date32Array, Float64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    /// Synthetic market: 3 stocks, warm-up + 4 target days, deterministic
    /// pseudo-random price paths so factor values are non-trivial.
    struct Fixture {
        _dir: tempfile::TempDir,
        catalog: PathBuf,
        minute_root: PathBuf,
    }

    fn pseudo_random(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed | 1;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) % 10_000) as f64 / 10_000.0
        }
    }

    fn build_fixture() -> Fixture {
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

        // calendar: the 6 trading days plus the New Year holiday marked off
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

        for (date_index, date) in dates.iter().enumerate() {
            let day_dir = minute_root
                .join(format!("year={}", &date[..4]))
                .join(format!("month={}", &date[5..7]))
                .join(format!("trade_date={date}"));
            std::fs::create_dir_all(&day_dir).unwrap();
            let mut rows = Vec::new();
            for (code_index, code) in codes.iter().enumerate() {
                let mut drift = pseudo_random(1_000 + 97 * (date_index * 10 + code_index) as u64);
                let mut price = 10.0 + code_index as f64;
                for minute in 0..SESSION_BARS {
                    let active = (date_index + code_index) % 3 != 0 || minute < 100;
                    let step = if active { (drift() - 0.5) * 0.01 } else { 0.0 };
                    let open = price;
                    price *= 1.0 + step;
                    let volume = if active {
                        1_000 + (drift() * 900.0) as i64
                    } else {
                        0
                    };
                    rows.push((
                        code.to_string(),
                        minute as u8,
                        open,
                        price,
                        volume,
                        volume as f64 * price,
                    ));
                }
            }
            write_minute_day(&day_dir, &rows);
        }
        Fixture {
            _dir: dir,
            catalog: catalog_path,
            minute_root,
        }
    }

    fn write_minute_day(day_dir: &Path, rows: &[(String, u8, f64, f64, i64, f64)]) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("ts_code", DataType::Utf8, false),
            Field::new("minute_index", DataType::UInt8, false),
            Field::new("open", DataType::Float64, true),
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
                Arc::new(arrow::array::UInt8Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                )),
                Arc::new(arrow::array::Int64Array::from(
                    rows.iter().map(|r| r.4).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.5).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        std::fs::create_dir_all(day_dir).unwrap();
        let file = std::fs::File::create(day_dir.join("part.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn build_args(fixture: &Fixture, output: &Path, jobs: usize, block_days: usize) -> BuildArgs {
        BuildArgs {
            catalog: fixture.catalog.clone(),
            minute_root: fixture.minute_root.clone(),
            output: output.to_path_buf(),
            start: "2024-01-04".into(),
            end: "2024-01-09".into(),
            block_days,
            jobs,
            threads_per_job: 2,
            memory_limit_mb: 2_000,
            replace: false,
            factor_set: "core24".into(),
        }
    }

    fn read_day(output: &Path, date: &str) -> Vec<(String, Vec<Option<f64>>)> {
        let file = std::fs::File::open(day_file(output, date)).unwrap();
        let batches: Vec<RecordBatch> =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap()
                .map(std::result::Result::unwrap)
                .collect();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        let codes = batch
            .column_by_name("ts_code")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values: Vec<Vec<Option<f64>>> = FACTOR_NAMES
            .iter()
            .map(|name| {
                let array = batch
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                (0..array.len())
                    .map(|i| {
                        if array.is_null(i) {
                            None
                        } else {
                            Some(array.value(i))
                        }
                    })
                    .collect()
            })
            .collect();
        (0..codes.len())
            .map(|row| {
                (
                    codes.value(row).to_string(),
                    (0..N_FACTORS).map(|slot| values[slot][row]).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn thread_count_and_block_layout_cannot_change_output() {
        let fixture = build_fixture();
        let serial = tempfile::tempdir().unwrap();
        let parallel = tempfile::tempdir().unwrap();
        run_build(build_args(&fixture, serial.path(), 1, 10)).unwrap();
        run_build(build_args(&fixture, parallel.path(), 2, 2)).unwrap();
        for date in ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"] {
            assert_eq!(
                read_day(serial.path(), date),
                read_day(parallel.path(), date),
                "{date}"
            );
        }
    }

    #[test]
    fn modifying_future_days_cannot_change_earlier_outputs() {
        let fixture = build_fixture();
        let baseline = tempfile::tempdir().unwrap();
        run_build(build_args(&fixture, baseline.path(), 1, 10)).unwrap();

        // corrupt the last two days' minute data, then rebuild
        let last = fixture
            .minute_root
            .join("year=2024")
            .join("month=01")
            .join("trade_date=2024-01-09");
        std::fs::remove_dir_all(&last).unwrap();
        let mut rows = Vec::new();
        for code in ["000001.SZ", "600000.SH", "300750.SZ"] {
            for minute in 0..SESSION_BARS {
                rows.push((code.to_string(), minute as u8, 1.0, 2.0, 1, 2.0));
            }
        }
        write_minute_day(&last, &rows);
        let rebuilt = tempfile::tempdir().unwrap();
        let mut args = build_args(&fixture, rebuilt.path(), 1, 10);
        args.replace = true;
        run_build(args).unwrap();

        for date in ["2024-01-04", "2024-01-05", "2024-01-08"] {
            assert_eq!(
                read_day(baseline.path(), date),
                read_day(rebuilt.path(), date),
                "{date}"
            );
        }
    }

    #[test]
    fn warmup_days_are_not_written_and_resume_skips_complete_days() {
        let fixture = build_fixture();
        let output = tempfile::tempdir().unwrap();
        run_build(build_args(&fixture, output.path(), 1, 10)).unwrap();
        for date in ["2024-01-02", "2024-01-03"] {
            assert!(
                !day_file(output.path(), date).exists(),
                "{date} is warm-up only"
            );
        }
        for date in ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"] {
            assert!(day_file(output.path(), date).is_file(), "{date} must exist");
        }

        // Simulate an interrupted run: drop a middle day's output and rerun.
        // The recomputed day and every later day must match a fresh build
        // value-for-value, because replayed history still feeds baselines.
        std::fs::remove_file(day_file(output.path(), "2024-01-05")).unwrap();
        run_build(build_args(&fixture, output.path(), 1, 10)).unwrap();
        assert!(day_file(output.path(), "2024-01-05").is_file());
        let fresh = tempfile::tempdir().unwrap();
        run_build(build_args(&fixture, fresh.path(), 1, 10)).unwrap();
        for date in ["2024-01-05", "2024-01-08", "2024-01-09"] {
            assert_eq!(
                read_day(output.path(), date),
                read_day(fresh.path(), date),
                "resumed values must equal fresh values on {date}"
            );
        }
    }

    #[test]
    fn overlapping_warmup_rerun_never_clobbers_completed_days() {
        let fixture = build_fixture();
        let output = tempfile::tempdir().unwrap();
        run_build(build_args(&fixture, output.path(), 1, 10)).unwrap();
        let early = day_file(output.path(), "2024-01-04");
        let before = std::fs::read(&early).unwrap();

        // A later partial build whose warm-up range covers completed target
        // days: those days must stay ok with their file untouched.
        let mut args = build_args(&fixture, output.path(), 1, 10);
        args.start = "2024-01-08".into();
        run_build(args).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(output.path().join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["dates"]["2024-01-04"]["status"], "ok");
        assert_eq!(manifest["dates"]["2024-01-05"]["status"], "ok");
        assert_eq!(std::fs::read(&early).unwrap(), before);
    }

    #[test]
    fn date32_encoding_matches_unix_epoch_days() {
        let expected = (chrono::NaiveDate::from_ymd_opt(2024, 1, 4).unwrap()
            - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
        .num_days() as i32;
        assert_eq!(expected, 19_726);
    }
}
