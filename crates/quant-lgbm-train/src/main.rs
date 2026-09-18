use anyhow::{Context, Result, bail};
use chrono::{Datelike, NaiveDate};
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde::Serialize;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::ptr;

const FLOAT32: c_int = 0;
const PREDICT_NORMAL: c_int = 0;
const SEED: u64 = 20_260_908;
const TRAIN_DAYS: usize = 756;
const LABEL_LAG: usize = 6;

type DatasetHandle = *mut c_void;
type BoosterHandle = *mut c_void;

#[link(name = "_lightgbm")]
unsafe extern "C" {
    fn LGBM_GetLastError() -> *const c_char;
    fn LGBM_DatasetCreateFromMat(
        data: *const c_void,
        data_type: c_int,
        nrow: i32,
        ncol: i32,
        is_row_major: c_int,
        parameters: *const c_char,
        reference: DatasetHandle,
        out: *mut DatasetHandle,
    ) -> c_int;
    fn LGBM_DatasetSetField(
        handle: DatasetHandle,
        field_name: *const c_char,
        field_data: *const c_void,
        num_element: c_int,
        data_type: c_int,
    ) -> c_int;
    fn LGBM_DatasetFree(handle: DatasetHandle) -> c_int;
    fn LGBM_BoosterCreate(
        train_data: DatasetHandle,
        parameters: *const c_char,
        out: *mut BoosterHandle,
    ) -> c_int;
    fn LGBM_BoosterAddValidData(handle: BoosterHandle, valid_data: DatasetHandle) -> c_int;
    fn LGBM_BoosterUpdateOneIter(handle: BoosterHandle, is_finished: *mut c_int) -> c_int;
    fn LGBM_BoosterGetEval(
        handle: BoosterHandle,
        data_idx: c_int,
        out_len: *mut c_int,
        out_results: *mut f64,
    ) -> c_int;
    fn LGBM_BoosterPredictForMat(
        handle: BoosterHandle,
        data: *const c_void,
        data_type: c_int,
        nrow: i32,
        ncol: i32,
        is_row_major: c_int,
        predict_type: c_int,
        start_iteration: c_int,
        num_iteration: c_int,
        parameters: *const c_char,
        out_len: *mut i64,
        out_result: *mut f64,
    ) -> c_int;
    fn LGBM_BoosterSaveModel(
        handle: BoosterHandle,
        start_iteration: c_int,
        num_iteration: c_int,
        feature_importance_type: c_int,
        filename: *const c_char,
    ) -> c_int;
    fn LGBM_BoosterFree(handle: BoosterHandle) -> c_int;
}

#[derive(Parser, Debug)]
#[command(about = "Rust rolling LightGBM trainer for one point-in-time CSI sleeve")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    feature_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    index_code: String,
    /// Reproduce the legacy QFQ research universe from index_trading_universe.
    /// The default uses the raw OHLCV completeness checks required by the new baseline.
    #[arg(long, default_value_t = false)]
    legacy_qfq_universe: bool,
    #[arg(long, default_value = "2021-04-01")]
    oos_start: String,
    #[arg(long, default_value = "2026-08-28")]
    oos_end: String,
    #[arg(long, default_value_t = 2048)]
    memory_limit_mb: usize,
    #[arg(long, default_value_t = 8)]
    threads: usize,
}

#[derive(Default)]
struct Panel {
    feature_count: usize,
    dates: Vec<String>,
    ranges: Vec<(usize, usize)>,
    codes: Vec<String>,
    executions: Vec<Option<String>>,
    x: Vec<f32>,
    raw_h1: Vec<f32>,
    raw_h5: Vec<f32>,
    win_h1: Vec<f32>,
    win_h5: Vec<f32>,
}

struct DayRow {
    code: String,
    execution: Option<String>,
    x: Vec<f64>,
    h1: f64,
    h5: f64,
}

struct Matrix {
    x: Vec<f32>,
    y: Vec<f32>,
    rows: usize,
}

struct Dataset(DatasetHandle);
impl Drop for Dataset {
    fn drop(&mut self) {
        unsafe {
            let _ = LGBM_DatasetFree(self.0);
        }
    }
}
struct Booster(BoosterHandle);
impl Drop for Booster {
    fn drop(&mut self) {
        unsafe {
            let _ = LGBM_BoosterFree(self.0);
        }
    }
}

#[derive(Serialize)]
struct WindowLog {
    signal: String,
    test_end: String,
    training_rows_h1: usize,
    training_rows_h5: usize,
    rounds_h1: usize,
    rounds_h5: usize,
}

fn lgb_check(status: c_int) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned();
    bail!("LightGBM C API: {message}")
}

fn sql_quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn ident(value: &str) -> Result<String> {
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        bail!("unsafe feature identifier: {value}")
    }
    Ok(format!("\"{value}\""))
}

fn open_catalog(path: &Path, memory_limit_mb: usize) -> Result<Connection> {
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(path, config)?;
    conn.execute_batch(&format!(
        "SET threads=1; SET memory_limit='{}MB'; SET preserve_insertion_order=false;",
        memory_limit_mb.max(512)
    ))?;
    Ok(conn)
}

fn nearest_quantile(sorted: &[f64], q: f64) -> f64 {
    let index = (q * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[index]
}

fn clipped(values: &[f64]) -> Vec<f32> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return vec![f32::NAN; values.len()];
    }
    finite.sort_by(f64::total_cmp);
    let lo = nearest_quantile(&finite, 0.01);
    let hi = nearest_quantile(&finite, 0.99);
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                value.clamp(lo, hi) as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}

fn append_day(panel: &mut Panel, date: String, rows: Vec<DayRow>) {
    if rows.is_empty() {
        return;
    }
    let begin = panel.codes.len();
    let n = rows.len();
    let p = panel.feature_count;
    let mut standardized = vec![f32::NAN; n * p];
    for feature in 0..p {
        let values = rows.iter().map(|row| row.x[feature]).collect::<Vec<_>>();
        let clipped_values = clipped(&values);
        let finite = clipped_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .map(f64::from)
            .collect::<Vec<_>>();
        if finite.len() < 2 {
            continue;
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let variance = finite
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (finite.len() - 1) as f64;
        let sd = variance.sqrt();
        if sd <= 1e-12 {
            continue;
        }
        for (row, value) in clipped_values.into_iter().enumerate() {
            if value.is_finite() {
                standardized[row * p + feature] = ((f64::from(value) - mean) / sd) as f32;
            }
        }
    }
    let h1 = rows.iter().map(|row| row.h1).collect::<Vec<_>>();
    let h5 = rows.iter().map(|row| row.h5).collect::<Vec<_>>();
    let win_h1 = clipped(&h1);
    let win_h5 = clipped(&h5);
    for (index, row) in rows.into_iter().enumerate() {
        panel.codes.push(row.code);
        panel.executions.push(row.execution);
        panel.raw_h1.push(row.h1 as f32);
        panel.raw_h5.push(row.h5 as f32);
        panel.win_h1.push(win_h1[index]);
        panel.win_h5.push(win_h5[index]);
    }
    panel.x.extend(standardized);
    panel.dates.push(date);
    panel.ranges.push((begin, begin + n));
}

fn load_panel(args: &Args, factors: &[String]) -> Result<Panel> {
    let conn = open_catalog(&args.catalog, args.memory_limit_mb)?;
    let feature_glob = sql_quote(&args.feature_root.join("year=*/features.parquet"));
    let columns = factors
        .iter()
        .map(|factor| ident(factor).map(|name| format!("f.{name}::DOUBLE")))
        .collect::<Result<Vec<_>>>()?
        .join(",");
    let exclude_overlap = if args.index_code == "000852.SH" {
        "AND NOT EXISTS (SELECT 1 FROM index_monthly_constituents c5 WHERE c5.index_code='000905.SH' AND c5.ts_code=c.ts_code AND c5.as_of_date=(SELECT max(c6.as_of_date) FROM index_monthly_constituents c6 WHERE c6.index_code='000905.SH' AND c6.as_of_date<=cal.trade_date))"
    } else {
        ""
    };
    let index_codes = args
        .index_code
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if index_codes.is_empty()
        || index_codes.iter().any(|code| {
            code.is_empty()
                || !code
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'.' || byte.is_ascii_uppercase())
        })
    {
        bail!("invalid --index-code list: {}", args.index_code);
    }
    let index_predicate = format!(
        "c.index_code IN ({})",
        index_codes
            .iter()
            .map(|code| format!("'{code}'"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let universe_sql = if args.legacy_qfq_universe {
        format!(
            r#"SELECT DISTINCT trade_date,ts_code
            FROM index_trading_universe
            WHERE index_code IN ({}) AND ts_code <> '000937.SZ'"#,
            index_codes
                .iter()
                .map(|code| format!("'{code}'"))
                .collect::<Vec<_>>()
                .join(",")
        )
    } else {
        format!(
            r#"SELECT DISTINCT cal.trade_date,c.ts_code
            FROM calendar cal
            JOIN index_monthly_constituents c ON {index_predicate}
             AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date)
            JOIN daily_aggregated d ON d.trade_date=cal.trade_date AND d.ts_code=c.ts_code
            WHERE d.open>0 AND d.high>0 AND d.low>0 AND d.close>0 AND d.amount_cny>0 AND d.volume_share>0
              AND d.observation_status='complete_trading' {exclude_overlap}
              AND c.ts_code <> '000937.SZ'"#
        )
    };
    let query = format!(
        r#"WITH calendar AS (
          SELECT trade_date,row_number() over(order by trade_date) n
          FROM observed_calendar WHERE is_observed_market_day
        ), universe AS ({universe_sql})
        SELECT f.trade_date::VARCHAR,ce.trade_date::VARCHAR,f.ts_code,{columns},
          CASE WHEN d1.qfq_open>0 AND d2.qfq_open>0 AND i1.open>0 AND i2.open>0 AND d1.amount_cny>0 AND d2.amount_cny>0 AND d1.observation_status='complete_trading' AND d2.observation_status='complete_trading'
            THEN d2.qfq_open/d1.qfq_open-i2.open/i1.open ELSE CAST('NaN' AS DOUBLE) END h1,
          CASE WHEN d1.qfq_open>0 AND d6.qfq_open>0 AND i1.open>0 AND i6.open>0 AND d1.amount_cny>0 AND d6.amount_cny>0 AND d1.observation_status='complete_trading' AND d6.observation_status='complete_trading'
            THEN d6.qfq_open/d1.qfq_open-i6.open/i1.open ELSE CAST('NaN' AS DOUBLE) END h5
        FROM read_parquet('{feature_glob}') f
        JOIN universe u USING(trade_date,ts_code)
        JOIN calendar c ON c.trade_date=f.trade_date
        LEFT JOIN calendar ce ON ce.n=c.n+1 LEFT JOIN calendar c2 ON c2.n=c.n+2 LEFT JOIN calendar c6 ON c6.n=c.n+6
        LEFT JOIN daily_qfq d1 ON d1.ts_code=f.ts_code AND d1.trade_date=ce.trade_date
        LEFT JOIN daily_qfq d2 ON d2.ts_code=f.ts_code AND d2.trade_date=c2.trade_date
        LEFT JOIN daily_qfq d6 ON d6.ts_code=f.ts_code AND d6.trade_date=c6.trade_date
        LEFT JOIN index_daily i1 ON i1.index_code='000905.SH' AND i1.trade_date=ce.trade_date
        LEFT JOIN index_daily i2 ON i2.index_code='000905.SH' AND i2.trade_date=c2.trade_date
        LEFT JOIN index_daily i6 ON i6.index_code='000905.SH' AND i6.trade_date=c6.trade_date
        ORDER BY f.trade_date,f.ts_code"#,
    );
    let mut statement = conn.prepare(&query)?;
    let factor_count = factors.len();
    let mapped = statement.query_map([], |row| {
        let mut x = Vec::with_capacity(factor_count);
        for column in 0..factor_count {
            x.push(row.get::<_, Option<f64>>(3 + column)?.unwrap_or(f64::NAN));
        }
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
            x,
            row.get::<_, f64>(3 + factor_count)?,
            row.get::<_, f64>(4 + factor_count)?,
        ))
    })?;
    let mut panel = Panel {
        feature_count: factor_count,
        ..Panel::default()
    };
    let mut current = String::new();
    let mut day = Vec::new();
    for item in mapped {
        let (date, execution, code, x, h1, h5) = item?;
        if !current.is_empty() && date != current {
            append_day(&mut panel, current, std::mem::take(&mut day));
        }
        current = date;
        day.push(DayRow {
            code,
            execution,
            x,
            h1,
            h5,
        });
    }
    append_day(&mut panel, current, day);
    Ok(panel)
}

fn date_row_bounds(panel: &Panel, begin: usize, end: usize) -> (usize, usize) {
    (panel.ranges[begin].0, panel.ranges[end - 1].1)
}

fn extract_labeled(
    panel: &Panel,
    begin: usize,
    end: usize,
    horizon: usize,
    winsor: bool,
) -> Matrix {
    let (row_begin, row_end) = date_row_bounds(panel, begin, end);
    let labels = match (horizon, winsor) {
        (1, false) => &panel.raw_h1,
        (1, true) => &panel.win_h1,
        (5, false) => &panel.raw_h5,
        (5, true) => &panel.win_h5,
        _ => unreachable!(),
    };
    let mut x = Vec::new();
    let mut y = Vec::new();
    for row in row_begin..row_end {
        if labels[row].is_finite() {
            x.extend_from_slice(
                &panel.x[row * panel.feature_count..(row + 1) * panel.feature_count],
            );
            y.push(labels[row]);
        }
    }
    Matrix {
        rows: y.len(),
        x,
        y,
    }
}

fn extract_test(panel: &Panel, begin: usize, end: usize) -> (Vec<f32>, usize, usize) {
    let (row_begin, row_end) = date_row_bounds(panel, begin, end);
    (
        panel.x[row_begin * panel.feature_count..row_end * panel.feature_count].to_vec(),
        row_begin,
        row_end,
    )
}

fn dataset(matrix: &Matrix, feature_count: usize, reference: DatasetHandle) -> Result<Dataset> {
    let mut handle = ptr::null_mut();
    let params = CString::new("max_bin=255 feature_pre_filter=false")?;
    lgb_check(unsafe {
        LGBM_DatasetCreateFromMat(
            matrix.x.as_ptr().cast(),
            FLOAT32,
            i32::try_from(matrix.rows)?,
            i32::try_from(feature_count)?,
            1,
            params.as_ptr(),
            reference,
            &mut handle,
        )
    })?;
    let label = CString::new("label")?;
    lgb_check(unsafe {
        LGBM_DatasetSetField(
            handle,
            label.as_ptr(),
            matrix.y.as_ptr().cast(),
            i32::try_from(matrix.y.len())?,
            FLOAT32,
        )
    })?;
    Ok(Dataset(handle))
}

fn params(threads: usize) -> Result<CString> {
    Ok(CString::new(format!(
        "objective=regression metric=l2 learning_rate=0.03 bagging_fraction=0.8 bagging_freq=1 feature_pre_filter=false verbosity=-1 seed={SEED} feature_fraction_seed={SEED} bagging_seed={SEED} data_random_seed={SEED} deterministic=true force_row_wise=true num_threads={} num_leaves=31 min_data_in_leaf=20 feature_fraction=1.0 lambda_l2=0.0",
        threads.max(1)
    ))?)
}

fn new_booster(train: &Dataset, threads: usize) -> Result<Booster> {
    let mut handle = ptr::null_mut();
    let parameters = params(threads)?;
    lgb_check(unsafe { LGBM_BoosterCreate(train.0, parameters.as_ptr(), &mut handle) })?;
    Ok(Booster(handle))
}

fn choose_rounds(train: &Matrix, valid: &Matrix, p: usize, threads: usize) -> Result<usize> {
    let train_data = dataset(train, p, ptr::null_mut())?;
    let valid_data = dataset(valid, p, train_data.0)?;
    let booster = new_booster(&train_data, threads)?;
    lgb_check(unsafe { LGBM_BoosterAddValidData(booster.0, valid_data.0) })?;
    let mut best = f64::INFINITY;
    let mut best_iteration = 1;
    let mut stale = 0;
    for iteration in 1..=2000 {
        let mut finished = 0;
        lgb_check(unsafe { LGBM_BoosterUpdateOneIter(booster.0, &mut finished) })?;
        let mut length = 0;
        let mut score = 0.0;
        lgb_check(unsafe { LGBM_BoosterGetEval(booster.0, 1, &mut length, &mut score) })?;
        if score < best {
            best = score;
            best_iteration = iteration;
            stale = 0;
        } else {
            stale += 1;
        }
        if stale >= 50 || finished != 0 {
            break;
        }
    }
    Ok(best_iteration)
}

fn fit_predict_save(
    train: &Matrix,
    test: &[f32],
    test_rows: usize,
    p: usize,
    rounds: usize,
    threads: usize,
    model_path: &Path,
) -> Result<Vec<f64>> {
    let train_data = dataset(train, p, ptr::null_mut())?;
    let booster = new_booster(&train_data, threads)?;
    for _ in 0..rounds {
        let mut finished = 0;
        lgb_check(unsafe { LGBM_BoosterUpdateOneIter(booster.0, &mut finished) })?;
        if finished != 0 {
            break;
        }
    }
    let mut prediction = vec![0.0; test_rows];
    let mut length = 0_i64;
    let empty = CString::new("")?;
    lgb_check(unsafe {
        LGBM_BoosterPredictForMat(
            booster.0,
            test.as_ptr().cast(),
            FLOAT32,
            i32::try_from(test_rows)?,
            i32::try_from(p)?,
            1,
            PREDICT_NORMAL,
            0,
            i32::try_from(rounds)?,
            empty.as_ptr(),
            &mut length,
            prediction.as_mut_ptr(),
        )
    })?;
    if usize::try_from(length)? != test_rows {
        bail!("prediction length mismatch: {length} vs {test_rows}")
    }
    let filename = CString::new(model_path.to_string_lossy().as_bytes())?;
    lgb_check(unsafe {
        LGBM_BoosterSaveModel(booster.0, 0, i32::try_from(rounds)?, 0, filename.as_ptr())
    })?;
    Ok(prediction)
}

fn month_id(value: &str) -> Result<i32> {
    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")?;
    Ok(date.year() * 12 + i32::try_from(date.month0())?)
}

fn windows(panel: &Panel, start: &str, end: &str) -> Result<Vec<(usize, usize)>> {
    let start_month = month_id(start)?;
    let mut out = Vec::new();
    for index in 0..panel.dates.len() {
        let date = &panel.dates[index];
        if date.as_str() < start || date.as_str() > end || index < TRAIN_DAYS + LABEL_LAG {
            continue;
        }
        let first_in_month = index == 0 || panel.dates[index - 1][..7] != date[..7];
        if !first_in_month || (month_id(date)? - start_month) % 3 != 0 {
            continue;
        }
        let end_month = month_id(date)? + 3;
        let mut finish = index;
        while finish < panel.dates.len()
            && panel.dates[finish].as_str() <= end
            && month_id(&panel.dates[finish])? < end_month
        {
            finish += 1;
        }
        if finish > index {
            out.push((index, finish));
        }
    }
    Ok(out)
}

fn zscore(values: &[f64]) -> Vec<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return vec![f64::NAN; values.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let sd = (finite
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / finite.len() as f64)
        .sqrt();
    values
        .iter()
        .map(|value| {
            if value.is_finite() && sd > 1e-12 {
                (value - mean) / sd
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let index_codes = args
        .index_code
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if index_codes.is_empty()
        || index_codes.iter().any(|code| {
            code.is_empty()
                || !code
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '.')
        })
    {
        bail!("invalid --index-code list: {}", args.index_code)
    }
    fs::create_dir_all(&args.output)?;
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(args.feature_root.join("manifest.json")).context("read feature manifest")?,
    )?;
    let factors = manifest["factor_ids"]
        .as_array()
        .context("manifest.factor_ids is not an array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .context("non-string factor id")
                .map(str::to_string)
        })
        .collect::<Result<Vec<_>>>()?;
    let panel = load_panel(&args, &factors)?;
    let rolling = windows(&panel, &args.oos_start, &args.oos_end)?;
    if rolling.is_empty() {
        bail!("no rolling windows")
    }
    let tsv = args.output.join("predictions.tsv");
    let mut writer = BufWriter::new(File::create(&tsv)?);
    writeln!(
        writer,
        "trade_date\tts_code\traw_h1\traw_h5\tpred_h1\tpred_h5\texecution_date"
    )?;
    let model_root = args.output.join("models");
    fs::create_dir_all(&model_root)?;
    let mut logs = Vec::new();
    for (test_begin, test_end) in rolling {
        let train_begin = test_begin - LABEL_LAG - TRAIN_DAYS;
        let early_end = train_begin + 687;
        let valid_begin = train_begin + 693;
        let train_end = test_begin - LABEL_LAG;
        let (test_x, row_begin, row_end) = extract_test(&panel, test_begin, test_end);
        let test_rows = row_end - row_begin;
        let signal = panel.dates[test_begin].clone();
        let quarter_dir = model_root.join(format!("month={}", &signal[..7]));
        fs::create_dir_all(&quarter_dir)?;
        let mut outputs = Vec::new();
        let mut rounds_record = Vec::new();
        let mut training_rows = Vec::new();
        for horizon in [1, 5] {
            let early_train = extract_labeled(&panel, train_begin, early_end, horizon, true);
            let validation = extract_labeled(&panel, valid_begin, train_end, horizon, false);
            let rounds =
                choose_rounds(&early_train, &validation, panel.feature_count, args.threads)?;
            let final_train = extract_labeled(&panel, train_begin, train_end, horizon, true);
            let model_path = quarter_dir.join(format!(
                "raw105_{}_h{horizon}.txt",
                args.index_code.replace('.', "_")
            ));
            let prediction = fit_predict_save(
                &final_train,
                &test_x,
                test_rows,
                panel.feature_count,
                rounds,
                args.threads,
                &model_path,
            )?;
            outputs.push(prediction);
            rounds_record.push(rounds);
            training_rows.push(final_train.rows);
        }
        let mut offset = 0;
        for date_index in test_begin..test_end {
            let (begin, end) = panel.ranges[date_index];
            let count = end - begin;
            let z1 = zscore(&outputs[0][offset..offset + count]);
            let z5 = zscore(&outputs[1][offset..offset + count]);
            for local in 0..count {
                let row = begin + local;
                if let Some(execution) = &panel.executions[row]
                    && z1[local].is_finite()
                    && z5[local].is_finite()
                {
                    writeln!(
                        writer,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        panel.dates[date_index],
                        panel.codes[row],
                        outputs[0][offset + local],
                        outputs[1][offset + local],
                        z1[local],
                        z5[local],
                        execution
                    )?;
                }
            }
            offset += count;
        }
        logs.push(WindowLog {
            signal,
            test_end: panel.dates[test_end - 1].clone(),
            training_rows_h1: training_rows[0],
            training_rows_h5: training_rows[1],
            rounds_h1: rounds_record[0],
            rounds_h5: rounds_record[1],
        });
        fs::write(
            args.output.join("training_log.json"),
            serde_json::to_string_pretty(&logs)? + "\n",
        )?;
    }
    drop(writer);
    let conn = Connection::open_in_memory()?;
    let src = sql_quote(&tsv);
    let dst = sql_quote(&args.output.join("predictions.parquet"));
    conn.execute_batch(&format!(
        "COPY (SELECT CAST(trade_date AS DATE) trade_date,ts_code,raw_h1::DOUBLE raw_h1,raw_h5::DOUBLE raw_h5,pred_h1::DOUBLE pred_h1,pred_h5::DOUBLE pred_h5,CAST(execution_date AS DATE) execution_date FROM read_csv('{src}',delim='\\t',header=true)) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)"
    ))?;
    fs::remove_file(tsv)?;
    fs::write(
        args.output.join("manifest.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "engine": "quant-lgbm-train-rust-v1",
            "index_code": args.index_code,
            "legacy_qfq_universe": args.legacy_qfq_universe,
            "factor_count": factors.len(),
            "training_days": TRAIN_DAYS,
            "label_lag": LABEL_LAG,
            "oos_start": args.oos_start,
            "oos_end": args.oos_end,
            "features": args.feature_root,
            "rows": panel.codes.len(),
            "dates": panel.dates.len(),
            "windows": logs.len(),
            "lightgbm_library": "lib_lightgbm.dylib C API"
        }))? + "\n",
    )?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "output": args.output,
            "index_code": args.index_code,
            "rows": panel.codes.len(),
            "dates": panel.dates.len(),
            "windows": logs.len()
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_and_clipping_are_deterministic() {
        let values = [0.0, 1.0, 2.0, 100.0];
        assert_eq!(clipped(&values), vec![0.0, 1.0, 2.0, 100.0]);
        assert_eq!(nearest_quantile(&values, 0.5), 2.0);
    }

    #[test]
    fn population_zscore_matches_prediction_normalization() {
        let z = zscore(&[1.0, 2.0, 3.0]);
        assert!(z[0] < 0.0 && z[2] > 0.0);
        assert!(z.iter().sum::<f64>().abs() < 1e-12);
    }
}
