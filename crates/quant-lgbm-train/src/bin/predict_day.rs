use anyhow::{Context, Result, bail};
use clap::Parser;
use duckdb::Connection;
use serde::Serialize;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::path::{Path, PathBuf};

const FLOAT32: c_int = 0;
const PREDICT_NORMAL: c_int = 0;
type BoosterHandle = *mut c_void;

#[link(name = "_lightgbm")]
unsafe extern "C" {
    fn LGBM_GetLastError() -> *const c_char;
    fn LGBM_BoosterCreateFromModelfile(
        filename: *const c_char,
        out_num_iterations: *mut c_int,
        out: *mut BoosterHandle,
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
    fn LGBM_BoosterFree(handle: BoosterHandle) -> c_int;
}

#[derive(Parser, Debug)]
#[command(about = "Apply a trained LightGBM model to one 125-factor daily parquet")]
struct Args {
    /// Wide parquet containing trade_date, ts_code and the exact manifest factors.
    #[arg(long)]
    features: PathBuf,
    /// Feature-cache manifest that freezes factor order.
    #[arg(long)]
    manifest: PathBuf,
    /// Explicit LightGBM text model. Mutually exclusive with --model-root.
    #[arg(
        long,
        conflicts_with = "model_root",
        required_unless_present = "model_root"
    )]
    model: Option<PathBuf>,
    /// Rolling-model root containing month=YYYY-MM directories. The latest
    /// quarter not later than the feature date is selected automatically.
    #[arg(long, conflicts_with = "model", required_unless_present = "model")]
    model_root: Option<PathBuf>,
    #[arg(long, default_value = "raw105_000300_SH,000905_SH_h1.txt")]
    model_name: String,
    /// Forecast horizon written to raw_hN and pred_hN columns.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=10))]
    horizon: u8,
    #[arg(long)]
    output: PathBuf,
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
struct OutputManifest {
    engine: &'static str,
    feature_file: String,
    feature_manifest: String,
    model: String,
    factor_count: usize,
    rows: usize,
    trade_date: String,
}

fn lgb_check(status: c_int) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(LGBM_GetLastError()) }.to_string_lossy();
    bail!("LightGBM C API: {message}")
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

fn nearest_quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[(q * sorted.len().saturating_sub(1) as f64).round() as usize]
}

/// Training parity: per-date 1/99 winsorization followed by sample z-score.
fn standardize(values: &[Vec<f64>], factors: usize) -> Vec<f32> {
    let mut output = vec![f32::NAN; values.len() * factors];
    for factor in 0..factors {
        let mut finite = values
            .iter()
            .map(|row| row[factor])
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        if finite.len() < 2 {
            continue;
        }
        finite.sort_by(f64::total_cmp);
        let lo = nearest_quantile(&finite, 0.01);
        let hi = nearest_quantile(&finite, 0.99);
        let clipped = values
            .iter()
            .map(|row| row[factor].clamp(lo, hi))
            .collect::<Vec<_>>();
        let usable = clipped
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        let mean = usable.iter().sum::<f64>() / usable.len() as f64;
        let variance =
            usable.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (usable.len() - 1) as f64;
        let sd = variance.sqrt();
        if sd <= 1e-12 {
            continue;
        }
        for (row, value) in clipped.into_iter().enumerate() {
            if value.is_finite() {
                output[row * factors + factor] = ((value - mean) / sd) as f32;
            }
        }
    }
    output
}

fn zscore(values: &[f64]) -> Vec<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return vec![f64::NAN; values.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let sd = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64).sqrt();
    values
        .iter()
        .map(|v| {
            if v.is_finite() && sd > 1e-12 {
                (v - mean) / sd
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn select_model(root: &Path, signal_date: &str, name: &str) -> Result<PathBuf> {
    let signal_month = &signal_date[..7];
    let mut eligible = fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let value = entry.file_name().to_string_lossy().to_string();
            let month = value.strip_prefix("month=")?;
            (month.len() == 7 && month <= signal_month && entry.path().join(name).is_file())
                .then(|| (month.to_string(), entry.path().join(name)))
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|a, b| a.0.cmp(&b.0));
    eligible
        .pop()
        .map(|(_, path)| path)
        .with_context(|| format!("no model {name} at or before {signal_month}"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let factors = manifest["factor_ids"]
        .as_array()
        .context("manifest.factor_ids missing")?
        .iter()
        .map(|v| {
            v.as_str()
                .context("factor id is not a string")
                .map(str::to_owned)
        })
        .collect::<Result<Vec<_>>>()?;
    if factors.len() != 125 {
        bail!(
            "production model requires 125 factors, manifest has {}",
            factors.len()
        );
    }
    let columns = factors
        .iter()
        .map(|v| ident(v).map(|id| format!("{id}::DOUBLE")))
        .collect::<Result<Vec<_>>>()?
        .join(",");
    let conn = Connection::open_in_memory()?;
    let sql = format!(
        "SELECT trade_date::VARCHAR,ts_code,{columns} FROM read_parquet('{}') ORDER BY ts_code",
        quote(&args.features)
    );
    let mut statement = conn.prepare(&sql)?;
    let mapped = statement.query_map([], |row| {
        let mut x = Vec::with_capacity(factors.len());
        for i in 0..factors.len() {
            x.push(row.get::<_, Option<f64>>(2 + i)?.unwrap_or(f64::NAN));
        }
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, x))
    })?;
    let rows = mapped.collect::<std::result::Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        bail!("feature parquet contains no rows");
    }
    let trade_date = rows[0].0.clone();
    if rows.iter().any(|row| row.0 != trade_date) {
        bail!("expected exactly one trade_date");
    }
    let codes = rows.iter().map(|row| row.1.clone()).collect::<Vec<_>>();
    let raw = rows.iter().map(|row| row.2.clone()).collect::<Vec<_>>();
    let matrix = standardize(&raw, factors.len());

    let model = match (&args.model, &args.model_root) {
        (Some(path), None) => path.clone(),
        (None, Some(root)) => select_model(root, &trade_date, &args.model_name)?,
        _ => unreachable!("clap validates model source"),
    };
    let filename = CString::new(model.to_string_lossy().as_bytes())?;
    let mut handle = std::ptr::null_mut();
    let mut iterations = 0;
    lgb_check(unsafe {
        LGBM_BoosterCreateFromModelfile(filename.as_ptr(), &mut iterations, &mut handle)
    })?;
    let booster = Booster(handle);
    let mut prediction = vec![0.0; rows.len()];
    let mut length = 0_i64;
    let parameters = CString::new("")?;
    lgb_check(unsafe {
        LGBM_BoosterPredictForMat(
            booster.0,
            matrix.as_ptr().cast(),
            FLOAT32,
            i32::try_from(rows.len())?,
            i32::try_from(factors.len())?,
            1,
            PREDICT_NORMAL,
            0,
            -1,
            parameters.as_ptr(),
            &mut length,
            prediction.as_mut_ptr(),
        )
    })?;
    if usize::try_from(length)? != rows.len() {
        bail!("prediction length mismatch");
    }
    let normalized = zscore(&prediction);

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = args.output.with_extension("parquet.tmp");
    let values = codes
        .iter()
        .zip(prediction.iter().zip(&normalized))
        .map(|(code, (raw, pred))| {
            format!(
                "('{}','{}',{raw},{pred})",
                trade_date,
                code.replace('\'', "''")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let raw_column = format!("raw_h{}", args.horizon);
    let normalized_column = format!("pred_h{}", args.horizon);
    conn.execute_batch(&format!("COPY (SELECT trade_date::DATE trade_date,ts_code,{raw_column}::DOUBLE {raw_column},{normalized_column}::DOUBLE {normalized_column} FROM (VALUES {values}) v(trade_date,ts_code,{raw_column},{normalized_column}) ORDER BY ts_code) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD)", quote(&temp)))?;
    if args.output.exists() {
        fs::remove_file(&args.output)?;
    }
    fs::rename(&temp, &args.output)?;
    let sidecar = args.output.with_extension("manifest.json");
    fs::write(
        sidecar,
        serde_json::to_vec_pretty(&OutputManifest {
            engine: "quant-lgbm-predict-day-rust-v1",
            feature_file: args.features.display().to_string(),
            feature_manifest: args.manifest.display().to_string(),
            model: model.display().to_string(),
            factor_count: factors.len(),
            rows: rows.len(),
            trade_date,
        })?,
    )?;
    println!(
        "{}",
        serde_json::json!({"output":args.output,"rows":rows.len(),"model_iterations":iterations})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_matches_training_contract() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let x = standardize(&rows, 1);
        assert!((x[0] + 1.0).abs() < 1e-6 && x[1].abs() < 1e-6 && (x[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn model_selection_never_uses_a_future_quarter() {
        let root = tempfile::tempdir().unwrap();
        for month in ["2026-04", "2026-07", "2026-10"] {
            let dir = root.path().join(format!("month={month}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("h1.txt"), month).unwrap();
        }
        let selected = select_model(root.path(), "2026-08-27", "h1.txt").unwrap();
        assert!(selected.to_string_lossy().contains("month=2026-07"));
    }
}
