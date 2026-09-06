//! Wide-format daily parquet writer with output validation and atomic moves.
use anyhow::{bail, Context, Result};
use arrow::array::{Date32Array, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use chrono::NaiveDate;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::schema::{FACTOR_NAMES, N_FACTORS};

pub struct WideRow {
    pub ts_code: String,
    pub values: [Option<f64>; N_FACTORS],
}

pub struct DynamicWideRow {
    pub ts_code: String,
    pub values: Vec<Option<f64>>,
}

pub fn write_day_dynamic(
    output_root: &Path,
    trade_date: &str,
    names: &[&str],
    rows: &[DynamicWideRow],
) -> Result<PathBuf> {
    if names.is_empty() {
        bail!("dynamic factor set has no columns");
    }
    for (i, row) in rows.iter().enumerate() {
        if row.values.len() != names.len() {
            bail!("{trade_date}: factor vector length mismatch");
        }
        if i > 0 && row.ts_code <= rows[i - 1].ts_code {
            bail!("{trade_date}: rows not strictly sorted by ts_code");
        }
        if row.values.iter().flatten().any(|v| !v.is_finite()) {
            bail!("{trade_date}: non-finite dynamic factor");
        }
    }
    let mut fields = vec![
        Field::new("trade_date", DataType::Date32, false),
        Field::new("ts_code", DataType::Utf8, false),
    ];
    fields.extend(
        names
            .iter()
            .map(|n| Field::new(*n, DataType::Float64, true)),
    );
    let date = NaiveDate::parse_from_str(trade_date, "%Y-%m-%d").context("parse trade_date")?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let dates = vec![(date - epoch).num_days() as i32; rows.len()];
    let codes: Vec<String> = rows.iter().map(|r| r.ts_code.clone()).collect();
    let mut columns = vec![Vec::with_capacity(rows.len()); names.len()];
    for row in rows {
        for (col, value) in columns.iter_mut().zip(&row.values) {
            col.push(*value);
        }
    }
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        std::iter::once(Arc::new(Date32Array::from(dates)) as Arc<dyn arrow::array::Array>)
            .chain(std::iter::once(
                Arc::new(StringArray::from(codes)) as Arc<dyn arrow::array::Array>
            ))
            .chain(
                columns
                    .into_iter()
                    .map(|c| Arc::new(Float64Array::from(c)) as Arc<dyn arrow::array::Array>),
            )
            .collect(),
    )?;
    let final_path = day_file(output_root, trade_date);
    let staging = output_root
        .join("_staging")
        .join(format!("{trade_date}.parquet"));
    std::fs::create_dir_all(final_path.parent().unwrap())?;
    std::fs::create_dir_all(staging.parent().unwrap())?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(File::create(&staging)?, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    std::fs::rename(&staging, &final_path)?;
    Ok(final_path)
}

fn output_schema() -> Arc<Schema> {
    let mut fields = vec![
        Field::new("trade_date", DataType::Date32, false),
        Field::new("ts_code", DataType::Utf8, false),
    ];
    fields.extend(
        FACTOR_NAMES
            .iter()
            .map(|name| Field::new(*name, DataType::Float64, true)),
    );
    Arc::new(Schema::new(fields))
}

fn day_file(output_root: &Path, trade_date: &str) -> PathBuf {
    output_root
        .join(format!("year={}", &trade_date[..4]))
        .join(format!("{trade_date}.parquet"))
}

fn validate(rows: &[WideRow], trade_date: &str) -> Result<()> {
    for (index, row) in rows.iter().enumerate() {
        if index > 0 && row.ts_code <= rows[index - 1].ts_code {
            bail!("{trade_date}: rows not strictly sorted by ts_code");
        }
        for (slot, value) in row.values.iter().enumerate() {
            if let Some(value) = value {
                if !value.is_finite() {
                    bail!(
                        "{trade_date}: non-finite value in {} for {}",
                        FACTOR_NAMES[slot],
                        row.ts_code
                    );
                }
            }
        }
    }
    Ok(())
}

/// Write one day, staged then atomically moved into place.  The staging file
/// lives under `<output>/_staging` so an interrupted write can never leave a
/// half-written day at the final path.
pub fn write_day(output_root: &Path, trade_date: &str, rows: &[WideRow]) -> Result<PathBuf> {
    validate(rows, trade_date)?;
    let final_path = day_file(output_root, trade_date);
    let staging = output_root
        .join("_staging")
        .join(format!("{trade_date}.parquet"));
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir_all(staging.parent().unwrap())?;

    let date = NaiveDate::parse_from_str(trade_date, "%Y-%m-%d").context("parse trade_date")?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let days_since_epoch = (date - epoch).num_days() as i32;

    let mut dates = Vec::with_capacity(rows.len());
    let mut codes = Vec::with_capacity(rows.len());
    let mut columns: Vec<Vec<Option<f64>>> = (0..N_FACTORS)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for row in rows {
        dates.push(days_since_epoch);
        codes.push(row.ts_code.clone());
        for (column, value) in columns.iter_mut().zip(row.values.iter()) {
            column.push(*value);
        }
    }

    let batch =
        RecordBatch::try_new(
            output_schema(),
            std::iter::once(Arc::new(Date32Array::from(dates)) as Arc<dyn arrow::array::Array>)
                .chain(std::iter::once(
                    Arc::new(StringArray::from(codes)) as Arc<dyn arrow::array::Array>
                ))
                .chain(columns.into_iter().map(|column| {
                    Arc::new(Float64Array::from(column)) as Arc<dyn arrow::array::Array>
                }))
                .collect(),
        )?;

    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = File::create(&staging).with_context(|| format!("create {}", staging.display()))?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), Some(properties))?;
    writer.write(&batch)?;
    writer
        .close()
        .with_context(|| format!("close {}", staging.display()))?;
    std::fs::rename(&staging, &final_path)
        .with_context(|| format!("publish {}", final_path.display()))?;
    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_day_is_validated_and_published_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let mut rows = vec![
            WideRow {
                ts_code: "000001.SZ".into(),
                values: std::array::from_fn(|i| Some(i as f64)),
            },
            WideRow {
                ts_code: "000002.SZ".into(),
                values: std::array::from_fn(|i| if i == 3 { None } else { Some(1.0) }),
            },
        ];
        let path = write_day(temp.path(), "2024-01-02", &rows).unwrap();
        assert!(path.is_file());
        assert!(!temp
            .path()
            .join("_staging")
            .join("2024-01-02.parquet")
            .exists());

        rows[1].values[0] = Some(f64::NAN);
        assert!(write_day(temp.path(), "2024-01-03", &rows).is_err());
        rows[1].values[0] = Some(f64::INFINITY);
        assert!(write_day(temp.path(), "2024-01-03", &rows).is_err());
        rows[1].values[0] = Some(0.0);
        rows[1].ts_code = "000001.SZ".into(); // duplicate code breaks sort order
        assert!(write_day(temp.path(), "2024-01-03", &rows).is_err());
    }

    #[test]
    fn written_day_round_trips_through_arrow() {
        let temp = tempfile::tempdir().unwrap();
        let rows = vec![WideRow {
            ts_code: "600000.SH".into(),
            values: std::array::from_fn(|i| Some(i as f64)),
        }];
        let path = write_day(temp.path(), "2024-01-02", &rows).unwrap();
        let file = File::open(&path).unwrap();
        let batches: Vec<_> =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap()
                .map(std::result::Result::unwrap)
                .collect();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), N_FACTORS + 2);
        let codes = batch
            .column_by_name("ts_code")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(codes.value(0), "600000.SH");
        let dates = batch
            .column_by_name("trade_date")
            .unwrap()
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        let expected = (NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()
            - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
        .num_days() as i32;
        assert_eq!(dates.value(0), expected);
    }

    #[test]
    fn dynamic_writer_round_trips_45_candidate_columns() {
        let temp = tempfile::tempdir().unwrap();
        let names: Vec<String> = (0..45).map(|i| format!("candidate_{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let rows = vec![DynamicWideRow {
            ts_code: "600000.SH".into(),
            values: (0..45).map(|i| Some(i as f64)).collect(),
        }];
        let path = write_day_dynamic(temp.path(), "2024-01-02", &refs, &rows).unwrap();
        let file = File::open(path).unwrap();
        let batch = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(batch.num_columns(), 47);
        assert!(batch.column_by_name("candidate_44").is_some());
    }
}
