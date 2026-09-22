use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use duckdb::{AccessMode, Config, Connection};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_CATALOG: &str = "A_stock_database/lake/catalog/a_share.duckdb";
const DEFAULT_MINUTE: &str = "A_stock_database/lake/canonical/minute";
const DEFAULT_FEATURE_MANIFEST: &str = "A_stock_database/lake/derived/predict_features_o2o_raw_daily60_minute45_dos20_csi300_csi500_v1/manifest.json";
const DEFAULT_MODEL_ROOT: &str =
    "results/predict/research-oos-raw-daily60-minute45-dos20-csi300-csi500-h1-h5-h10-v1/models";
const DEFAULT_OUTPUT: &str = "results/production/raw_daily60_minute45_dos20_h1h5_blend_80_20_v1";
const DEFAULT_PYTHON: &str = "/Users/alanmxy/anaconda3/envs/ml311/bin/python";
const DEFAULT_STRATEGY: &str = "configs/production_strategy_csi500_h1h5_blend_80_20_2bps_v5.json";
const DEFAULT_CALENDAR: &str = "A_stock_database/交易日历.csv";

#[derive(Parser)]
#[command(
    name = "quant-production",
    about = "One-command Rust daily factor, prediction and position production"
)]
struct Cli {
    #[command(subcommand)]
    command: Subcommands,
}

#[derive(Subcommand)]
enum Subcommands {
    /// Produce factors.parquet, prediction.parquet, and positions.parquet for one signal date.
    Run(RunArgs),
}

#[derive(Args, Debug)]
struct RunArgs {
    #[arg(long)]
    date: String,
    #[arg(long, default_value = DEFAULT_CATALOG)]
    catalog: PathBuf,
    #[arg(long, default_value = DEFAULT_MINUTE)]
    minute_root: PathBuf,
    #[arg(long, default_value = DEFAULT_FEATURE_MANIFEST)]
    feature_manifest: PathBuf,
    #[arg(long, default_value = DEFAULT_MODEL_ROOT)]
    model_root: PathBuf,
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output_root: PathBuf,
    #[arg(long, default_value = DEFAULT_PYTHON)]
    python_bin: PathBuf,
    #[arg(long, default_value = DEFAULT_STRATEGY)]
    strategy_config: PathBuf,
    #[arg(long, default_value = DEFAULT_CALENDAR)]
    exchange_calendar: PathBuf,
    /// Directory containing quant-daily-factor, quant-minute-factor,
    /// quant-lgbm-predict-day, and quant-position-day. Defaults to this executable's directory.
    #[arg(long)]
    bin_dir: Option<PathBuf>,
    /// Force rebuilding already materialized minute-factor day files.
    #[arg(long)]
    replace_minute: bool,
    /// Override automatic lookup of the latest earlier production day directory.
    #[arg(long)]
    previous_production_dir: Option<PathBuf>,
    /// Start deployment from cash instead of carrying any earlier position artifact.
    #[arg(long, conflicts_with = "previous_production_dir")]
    start_flat: bool,
}

#[derive(Clone)]
struct MinuteSource {
    root: PathBuf,
    factor_ids: Vec<String>,
    factor_set: &'static str,
    index_codes: &'static str,
}

#[derive(Serialize)]
struct RunManifest {
    engine: &'static str,
    status: &'static str,
    signal_date: String,
    factor_count: usize,
    factor_rows: i64,
    prediction_rows: i64,
    position_rows: i64,
    factors: String,
    prediction: String,
    positions: String,
    selected_h1_model: String,
    selected_h5_model: String,
    strategy_config: String,
    previous_production_dir: Option<String>,
    start_flat: bool,
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn ident(value: &str) -> Result<String> {
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        bail!("unsafe factor identifier: {value}")
    }
    Ok(format!("\"{value}\""))
}

fn day_file(root: &Path, date: &str) -> PathBuf {
    root.join(format!("year={}", &date[..4]))
        .join(format!("{date}.parquet"))
}

fn historical_file(root: &Path, date: &str) -> Option<PathBuf> {
    let daily = day_file(root, date);
    if daily.is_file() {
        return Some(daily);
    }
    let annual = root
        .join(format!("year={}", &date[..4]))
        .join("factors.parquet");
    annual.is_file().then_some(annual)
}

fn parquet_has_date(conn: &Connection, path: &Path, date: &str) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    conn.query_row(
        &format!(
            "SELECT count(*)>0 FROM read_parquet('{}') WHERE trade_date=?::DATE",
            quote(path)
        ),
        [date],
        |r| r.get(0),
    )
    .map_err(Into::into)
}

fn executable(bin_dir: &Path, name: &str) -> Result<PathBuf> {
    let path = bin_dir.join(name);
    if !path.is_file() {
        bail!(
            "required Rust executable is missing: {}; build the release workspace first",
            path.display()
        )
    }
    Ok(path)
}

fn run_command(program: &Path, arguments: &[String]) -> Result<()> {
    eprintln!("running {} {}", program.display(), arguments.join(" "));
    let status = Command::new(program)
        .args(arguments)
        .status()
        .with_context(|| format!("launch {}", program.display()))?;
    if !status.success() {
        bail!("{} failed with {status}", program.display())
    }
    Ok(())
}

fn validate_date(conn: &Connection, date: &str) -> Result<()> {
    let observed: bool = conn
        .query_row(
            "SELECT count(*)>0 FROM (SELECT trade_date FROM observed_calendar WHERE is_observed_market_day UNION SELECT DISTINCT trade_date FROM market_daily_aggregated) WHERE trade_date=?::DATE",
            [date],
            |r| r.get(0),
        )
        .with_context(|| format!("validate signal date {date}"))?;
    if !observed {
        bail!("{date} is not an observed market date")
    }
    Ok(())
}

fn source_kind(root: &str) -> Result<(&'static str, &'static str)> {
    if root.contains("ohlcv_candidates_v1") {
        Ok(("ohlcv_candidates_v1", "000300.SH,000905.SH,000852.SH"))
    } else if root.contains("core24") {
        Ok(("core24", "000300.SH,000905.SH,000852.SH"))
    } else if root.contains("ohlcv_candidates_v3") {
        Ok(("ohlcv_candidates_v3", "000300.SH,000905.SH,000852.SH"))
    } else if root.contains("dos_minute_v1") {
        Ok(("dos_minute_v1", "000300.SH,000905.SH"))
    } else {
        bail!("cannot infer Rust minute factor set from dataset {root}")
    }
}

fn load_contract(path: &Path) -> Result<(Vec<String>, Vec<String>, Vec<MinuteSource>, String)> {
    let document: Value = serde_json::from_slice(&fs::read(path)?)?;
    let strings = |key: &str| -> Result<Vec<String>> {
        document[key]
            .as_array()
            .with_context(|| format!("manifest.{key} missing"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .context("factor id must be a string")
                    .map(str::to_owned)
            })
            .collect()
    };
    let all = strings("factor_ids")?;
    let daily = strings("daily_factor_ids")?;
    if all.len() != 125 || daily.len() != 60 {
        bail!(
            "production contract requires 125 total and 60 daily factors; got {} and {}",
            all.len(),
            daily.len()
        )
    }
    let mut minute = Vec::new();
    for item in document["minute_factor_sources"]
        .as_array()
        .context("manifest.minute_factor_sources missing")?
    {
        let root = item["dataset"].as_str().context("minute dataset missing")?;
        let factors = item["factor_ids"]
            .as_array()
            .context("minute factor_ids missing")?
            .iter()
            .map(|v| {
                v.as_str()
                    .context("minute factor id must be a string")
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>>>()?;
        let (factor_set, index_codes) = source_kind(root)?;
        minute.push(MinuteSource {
            root: PathBuf::from(root),
            factor_ids: factors,
            factor_set,
            index_codes,
        });
    }
    let minute_count: usize = minute.iter().map(|x| x.factor_ids.len()).sum();
    if minute_count != 65 || daily.len() + minute_count != all.len() {
        bail!("production contract requires Minute45 + DOS20 = 65 factors; got {minute_count}")
    }
    let indexes = document["index_code"]
        .as_str()
        .unwrap_or("000300.SH,000905.SH")
        .to_owned();
    Ok((all, daily, minute, indexes))
}

fn build_features(
    conn: &Connection,
    date: &str,
    daily_file: &Path,
    sources: &[MinuteSource],
    minute_files: &[PathBuf],
    factors: &[String],
    daily_ids: &[String],
    indexes: &str,
    output: &Path,
) -> Result<i64> {
    let index_values = indexes
        .split(',')
        .map(|x| format!("'{}'", x.trim().replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(",");
    let mut selects = daily_ids
        .iter()
        .map(|x| ident(x).map(|id| format!("d.{id} AS {id}")))
        .collect::<Result<Vec<_>>>()?;
    let mut joins = vec![format!(
        "LEFT JOIN read_parquet('{}') d USING(trade_date,ts_code)",
        quote(daily_file)
    )];
    for (i, (source, file)) in sources.iter().zip(minute_files).enumerate() {
        if !file.is_file() {
            bail!(
                "minute factor day is missing after build: {}",
                file.display()
            )
        }
        joins.push(format!(
            "LEFT JOIN read_parquet('{}') m{i} USING(trade_date,ts_code)",
            quote(&file)
        ));
        selects.extend(
            source
                .factor_ids
                .iter()
                .map(|x| ident(x).map(|id| format!("m{i}.{id} AS {id}")))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    if selects.len() != factors.len() {
        bail!(
            "assembled column count {} differs from contract {}",
            selects.len(),
            factors.len()
        )
    }
    let temp = output.with_extension("parquet.tmp");
    if temp.exists() {
        fs::remove_file(&temp)?;
    }
    let query = format!(
        "WITH universe AS (SELECT DISTINCT c.ts_code, ?::DATE trade_date FROM index_monthly_constituents c JOIN market_daily_aggregated b ON b.ts_code=c.ts_code AND b.trade_date=?::DATE WHERE c.index_code IN ({index_values}) AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=?::DATE) AND b.open>0 AND b.high>0 AND b.low>0 AND b.close>0 AND b.volume_share>0 AND b.amount_cny>0 AND b.observation_status='complete_trading' AND c.ts_code<>'000937.SZ') SELECT u.trade_date,u.ts_code,{} FROM universe u {} ORDER BY u.ts_code",
        selects.join(","),
        joins.join(" ")
    );
    let rendered = query
        .replacen('?', &format!("DATE '{date}'"), 1)
        .replacen("?::DATE", &format!("DATE '{date}'"), 1)
        .replacen("?::DATE", &format!("DATE '{date}'"), 1);
    conn.execute_batch(&format!(
        "COPY ({rendered}) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD)",
        quote(&temp)
    ))?;
    if output.exists() {
        fs::remove_file(output)?;
    }
    fs::rename(&temp, output)?;
    let (rows, dates, duplicates, columns): (i64, i64, i64, i64) = conn.query_row(
        &format!("SELECT count(*),count(distinct trade_date),count(*)-count(distinct (trade_date,ts_code)),(SELECT count(*) FROM (DESCRIBE SELECT * FROM read_parquet('{}')) WHERE column_name NOT IN ('trade_date','ts_code','date','year','month')) FROM read_parquet('{}')", quote(output), quote(output)),
        [], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?;
    if rows == 0 || dates != 1 || duplicates != 0 || columns as usize != factors.len() {
        bail!(
            "invalid factor artifact: rows={rows}, dates={dates}, duplicates={duplicates}, factors={columns}"
        )
    }
    Ok(rows)
}

fn latest_previous(root: &Path, date: &str) -> Result<Option<PathBuf>> {
    if !root.is_dir() {
        return Ok(None);
    }
    let mut found = fs::read_dir(root)?
        .filter_map(|x| x.ok())
        .filter_map(|x| {
            let name = x.file_name().to_string_lossy().to_string();
            let d = name.strip_prefix("date=")?;
            let directory = x.path();
            let core = directory.join("sleeves/core/target_weights.parquet");
            let alpha = directory.join("sleeves/alpha/target_weights.parquet");
            (d < date && core.is_file() && alpha.is_file()).then_some((d.to_owned(), directory))
        })
        .collect::<Vec<_>>();
    found.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(found.pop().map(|x| x.1))
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

fn parquet_rows(conn: &Connection, path: &Path) -> Result<i64> {
    conn.query_row(
        &format!("SELECT count(*) FROM read_parquet('{}')", quote(path)),
        [],
        |r| r.get(0),
    )
    .map_err(Into::into)
}

fn run(args: RunArgs) -> Result<()> {
    if args.date.len() != 10 {
        bail!("--date must be YYYY-MM-DD")
    }
    let catalog = args
        .catalog
        .canonicalize()
        .with_context(|| format!("catalog {}", args.catalog.display()))?;
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(&catalog, config)?;
    validate_date(&conn, &args.date)?;
    let (factors, daily_ids, sources, indexes) = load_contract(&args.feature_manifest)?;
    let bin_dir = args.bin_dir.unwrap_or(
        std::env::current_exe()?
            .parent()
            .context("executable has no parent")?
            .to_owned(),
    );
    let daily_bin = executable(&bin_dir, "quant-daily-factor")?;
    let minute_bin = executable(&bin_dir, "quant-minute-factor")?;
    let predict_bin = executable(&bin_dir, "quant-lgbm-predict-day")?;
    if !args.python_bin.is_file() {
        bail!("Python runtime is missing: {}", args.python_bin.display())
    }
    let strategy: Value = serde_json::from_slice(&fs::read(&args.strategy_config)?)?;
    if strategy["status"] != "active"
        || strategy["model"]["production_signals"] != serde_json::json!(["H1", "H5"])
        || strategy["portfolio"]["strategy"] != "netted_target_weight_blend_v1"
    {
        bail!("strategy config is not the active H1/H5 netted 80/20 production baseline")
    }
    let day_dir = args.output_root.join(format!("date={}", args.date));
    fs::create_dir_all(&day_dir)?;
    let daily_file = day_dir.join("daily60.parquet");
    if !parquet_has_date(&conn, &daily_file, &args.date)? {
        run_command(
            &daily_bin,
            &[
                "build".into(),
                "--catalog".into(),
                catalog.display().to_string(),
                "--output".into(),
                daily_file.display().to_string(),
                "--date".into(),
                args.date.clone(),
            ],
        )?;
    }
    let mut minute_files = Vec::with_capacity(sources.len());
    for source in &sources {
        if !args.replace_minute
            && let Some(file) = historical_file(&source.root, &args.date)
            && parquet_has_date(&conn, &file, &args.date)?
        {
            minute_files.push(file);
            continue;
        }
        // Historical datasets have a different annual-file manifest contract.
        // New daily increments are isolated here so production never mutates
        // or invalidates those immutable research artifacts.
        let live_root = args
            .output_root
            .join("_minute_cache")
            .join(source.factor_set);
        let live_file = day_file(&live_root, &args.date);
        if !args.replace_minute && parquet_has_date(&conn, &live_file, &args.date)? {
            minute_files.push(live_file);
            continue;
        }
        let threads = if matches!(source.factor_set, "ohlcv_candidates_v3" | "dos_minute_v1") {
            "1"
        } else {
            "4"
        };
        let mut command = vec![
            "build".into(),
            "--catalog".into(),
            catalog.display().to_string(),
            "--minute-root".into(),
            args.minute_root.display().to_string(),
            "--output".into(),
            live_root.display().to_string(),
            "--start".into(),
            args.date.clone(),
            "--end".into(),
            args.date.clone(),
            "--block-days".into(),
            "1".into(),
            "--jobs".into(),
            "1".into(),
            "--threads-per-job".into(),
            threads.into(),
            "--factor-set".into(),
            source.factor_set.into(),
            "--index-codes".into(),
            source.index_codes.into(),
            "--raw-eligible-universe".into(),
        ];
        if args.replace_minute {
            command.push("--replace".into());
        }
        run_command(&minute_bin, &command)?;
        if !parquet_has_date(&conn, &live_file, &args.date)? {
            bail!("Rust minute build did not publish {}", live_file.display())
        }
        minute_files.push(live_file);
    }
    let factor_file = day_dir.join("factors.parquet");
    let factor_rows = build_features(
        &conn,
        &args.date,
        &daily_file,
        &sources,
        &minute_files,
        &factors,
        &daily_ids,
        &indexes,
        &factor_file,
    )?;
    let h1_file = day_dir.join("prediction_h1.parquet");
    let h5_file = day_dir.join("prediction_h5.parquet");
    for (horizon, model_name, output) in [
        (1_u8, "raw105_000300_SH,000905_SH_h1.txt", &h1_file),
        (5_u8, "raw105_000300_SH,000905_SH_h5.txt", &h5_file),
    ] {
        run_command(
            &predict_bin,
            &[
                "--features".into(),
                factor_file.display().to_string(),
                "--manifest".into(),
                args.feature_manifest.display().to_string(),
                "--model-root".into(),
                args.model_root.display().to_string(),
                "--model-name".into(),
                model_name.into(),
                "--horizon".into(),
                horizon.to_string(),
                "--output".into(),
                output.display().to_string(),
            ],
        )?;
    }
    let execution_date = next_execution_date(&args.exchange_calendar, &args.date)?;
    let prediction_file = day_dir.join("prediction.parquet");
    let prediction_temp = prediction_file.with_extension("parquet.tmp");
    conn.execute_batch(&format!(
        "COPY (SELECT h1.trade_date,DATE '{execution_date}' execution_date,h1.ts_code,h1.raw_h1,h1.pred_h1,h5.raw_h5,h5.pred_h5 FROM read_parquet('{}') h1 JOIN read_parquet('{}') h5 USING(trade_date,ts_code) ORDER BY h1.ts_code) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD)",
        quote(&h1_file), quote(&h5_file), quote(&prediction_temp)
    ))?;
    if prediction_file.exists() {
        fs::remove_file(&prediction_file)?;
    }
    fs::rename(&prediction_temp, &prediction_file)?;
    let prediction_rows = parquet_rows(&conn, &prediction_file)?;
    if prediction_rows != factor_rows {
        bail!("H1/H5 prediction join has {prediction_rows} rows, expected {factor_rows}")
    }

    let core_prediction = day_dir.join("prediction_csi500.parquet");
    let core_temp = core_prediction.with_extension("parquet.tmp");
    conn.execute_batch(&format!(
        "COPY (SELECT p.* FROM read_parquet('{}') p JOIN index_monthly_constituents c ON c.ts_code=p.ts_code AND c.index_code='000905.SH' AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=p.trade_date) ORDER BY p.ts_code) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD)",
        quote(&prediction_file), quote(&core_temp)
    ))?;
    if core_prediction.exists() {
        fs::remove_file(&core_prediction)?;
    }
    fs::rename(&core_temp, &core_prediction)?;

    let previous = match (args.start_flat, args.previous_production_dir) {
        (true, _) => None,
        (false, Some(x)) => Some(x),
        (false, None) => latest_previous(&args.output_root, &args.date)?,
    };
    let sleeves = day_dir.join("sleeves");
    let core_output = sleeves.join("core");
    let alpha_output = sleeves.join("alpha");
    for (name, predictions, output, cap) in [
        ("core", &core_prediction, &core_output, "0.01"),
        ("alpha", &prediction_file, &alpha_output, "0.05"),
    ] {
        let mut optimizer_args = vec![
            "scripts/optimize_multiperiod_mu_turnover.py".into(),
            "--predictions".into(),
            predictions.display().to_string(),
            "--output".into(),
            output.display().to_string(),
            "--max-weight".into(),
            cap.into(),
            "--term-structure".into(),
            "h1h5".into(),
            "--buy-bps".into(),
            "2".into(),
            "--sell-bps".into(),
            "2".into(),
        ];
        if let Some(root) = &previous {
            let prior = root.join(format!("sleeves/{name}/target_weights.parquet"));
            if !prior.is_file() {
                bail!("previous {name} sleeve is missing: {}", prior.display())
            }
            optimizer_args.extend(["--previous-positions".into(), prior.display().to_string()]);
        }
        run_command(&args.python_bin, &optimizer_args)?;
    }
    let blend_output = sleeves.join("blend");
    run_command(
        &args.python_bin,
        &[
            "scripts/blend_target_weights.py".into(),
            "--target-weights".into(),
            core_output
                .join("target_weights.parquet")
                .display()
                .to_string(),
            "--allocation".into(),
            "0.8".into(),
            "--target-weights".into(),
            alpha_output
                .join("target_weights.parquet")
                .display()
                .to_string(),
            "--allocation".into(),
            "0.2".into(),
            "--output".into(),
            blend_output.display().to_string(),
        ],
    )?;
    let position_file = day_dir.join("positions.parquet");
    fs::copy(blend_output.join("target_weights.parquet"), &position_file)?;
    let position_rows = parquet_rows(&conn, &position_file)?;
    let h1_manifest: Value =
        serde_json::from_slice(&fs::read(h1_file.with_extension("manifest.json"))?)?;
    let h5_manifest: Value =
        serde_json::from_slice(&fs::read(h5_file.with_extension("manifest.json"))?)?;
    let selected_h1_model = h1_manifest["model"]
        .as_str()
        .context("H1 prediction manifest model missing")?
        .to_owned();
    let selected_h5_model = h5_manifest["model"]
        .as_str()
        .context("H5 prediction manifest model missing")?
        .to_owned();
    let blend_summary: Value =
        serde_json::from_slice(&fs::read(blend_output.join("optimizer_summary.json"))?)?;
    fs::write(
        day_dir.join("positions.manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "engine": "h1h5-five-period-80-20-production-v1",
            "signal_date": args.date,
            "execution_date": execution_date,
            "strategy_config": args.strategy_config,
            "start_flat": args.start_flat,
            "previous_production_dir": previous,
            "core_sleeve": {"allocation": 0.8, "max_weight": 0.01},
            "alpha_sleeve": {"allocation": 0.2, "max_weight": 0.05},
            "blend": blend_summary,
        }))?,
    )?;
    let manifest = RunManifest {
        engine: "quant-production-rust-v1",
        status: "complete",
        signal_date: args.date,
        factor_count: factors.len(),
        factor_rows,
        prediction_rows,
        position_rows,
        factors: factor_file.display().to_string(),
        prediction: prediction_file.display().to_string(),
        positions: position_file.display().to_string(),
        selected_h1_model,
        selected_h5_model,
        strategy_config: args.strategy_config.display().to_string(),
        previous_production_dir: previous.map(|x| x.display().to_string()),
        start_flat: args.start_flat,
    };
    fs::write(
        day_dir.join("run_manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Subcommands::Run(args) => run(args),
    }
}
