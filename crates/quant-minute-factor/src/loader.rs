//! Minute-bar day partition loader with strict shape validation.
use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::schema::SESSION_BARS;

/// One minute bar, projected to the columns the factors actually consume.
#[derive(Clone, Copy, Debug)]
pub struct Bar {
    pub minute_index: u8,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume_share: f64,
    pub amount: f64,
}

pub struct DayData {
    /// ts_code -> bars sorted by minute_index, validated to be 0..=240 once each.
    pub stocks: Vec<(String, Box<[Bar; SESSION_BARS]>)>,
    /// Stocks skipped for this day, keyed by exclusion reason.
    pub excluded: Vec<(String, &'static str)>,
}

fn day_partition(minute_root: &Path, trade_date: &str) -> PathBuf {
    let (year, month) = (&trade_date[..4], &trade_date[5..7]);
    minute_root
        .join(format!("year={year}"))
        .join(format!("month={month}"))
        .join(format!("trade_date={trade_date}"))
}

/// Column accessor that tolerates string-view / large-utf8 encodings by
/// casting to the canonical arrow type first.
fn string_column(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<arrow::array::StringArray> {
    let column = batch
        .column_by_name(name)
        .with_context(|| format!("column {name} missing from minute parquet"))?;
    let casted = cast(column, &DataType::Utf8).context("cast ts_code to Utf8")?;
    let array = casted
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .context("ts_code is not a string array")
        .map(|a| a.clone())?;
    Ok(array)
}

fn f64_column(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<arrow::array::Float64Array> {
    let column = batch
        .column_by_name(name)
        .with_context(|| format!("column {name} missing from minute parquet"))?;
    let casted = cast(column, &DataType::Float64).context("cast numeric column to Float64")?;
    Ok(casted
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .context("numeric column is not Float64")
        .map(|a| a.clone())?)
}

/// Load one trade date.  `Ok(None)` means the partition directory is absent.
pub fn load_day(minute_root: &Path, trade_date: &str) -> Result<Option<DayData>> {
    let partition = day_partition(minute_root, trade_date);
    if !partition.is_dir() {
        return Ok(None);
    }
    let mut parts: Vec<PathBuf> = std::fs::read_dir(&partition)?
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    parts.sort();
    if parts.is_empty() {
        return Ok(None);
    }

    let mut bars: HashMap<String, Box<[Bar; SESSION_BARS]>> = HashMap::new();
    let mut excluded: Vec<(String, &'static str)> = Vec::new();
    let mut counts: HashMap<String, u32> = HashMap::new();

    for part in &parts {
        let file = File::open(part).with_context(|| format!("open {}", part.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("read metadata {}", part.display()))?;
        let schema = builder.schema().clone();
        let projection: Vec<usize> = [
            "ts_code",
            "minute_index",
            "open",
            "high",
            "low",
            "close",
            "volume_share",
            "amount_cny",
        ]
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .with_context(|| format!("column {name} missing in {}", part.display()))
        })
        .collect::<Result<Vec<_>>>()?;
        let mask = parquet::arrow::ProjectionMask::roots(
            builder.metadata().file_metadata().schema_descr(),
            projection.iter().copied(),
        );
        let reader = builder
            .with_projection(mask)
            .with_batch_size(65_536)
            .build()?;
        for batch in reader {
            let batch = batch.with_context(|| format!("read batch {}", part.display()))?;
            let codes = string_column(&batch, "ts_code")?;
            let indices = cast(
                batch
                    .column_by_name("minute_index")
                    .context("minute_index")?,
                &DataType::UInt8,
            )?;
            let indices = indices
                .as_any()
                .downcast_ref::<arrow::array::UInt8Array>()
                .context("minute_index is not UInt8")
                .map(|a| a.clone())?;
            let opens = f64_column(&batch, "open")?;
            let highs = f64_column(&batch, "high")?;
            let lows = f64_column(&batch, "low")?;
            let closes = f64_column(&batch, "close")?;
            let volumes = f64_column(&batch, "volume_share")?;
            let amounts = f64_column(&batch, "amount_cny")?;
            for row in 0..batch.num_rows() {
                if codes.is_null(row) || indices.is_null(row) {
                    continue;
                }
                let code = codes.value(row).to_string();
                let index = indices.value(row);
                if index as usize >= SESSION_BARS {
                    excluded.push((code, "minute_index_out_of_range"));
                    continue;
                }
                *counts.entry(code.clone()).or_default() += 1;
                let slot = bars.entry(code).or_insert_with(|| {
                    Box::new(std::array::from_fn(|_| Bar {
                        minute_index: 0,
                        open: f64::NAN,
                        high: f64::NAN,
                        low: f64::NAN,
                        close: f64::NAN,
                        volume_share: 0.0,
                        amount: 0.0,
                    }))
                });
                slot[index as usize] = Bar {
                    minute_index: index,
                    open: opens.value(row),
                    high: highs.value(row),
                    low: lows.value(row),
                    close: closes.value(row),
                    volume_share: volumes.value(row),
                    amount: amounts.value(row),
                };
            }
        }
    }

    let mut stocks: Vec<(String, Box<[Bar; SESSION_BARS]>)> = Vec::with_capacity(bars.len());
    for (code, day_bars) in bars {
        if counts[&code] as usize != SESSION_BARS {
            excluded.push((code, "bar_count_invalid"));
            continue;
        }
        if day_bars.iter().any(|bar| {
            !(bar.open.is_finite()
                && bar.close.is_finite()
                && bar.high.is_finite()
                && bar.low.is_finite()
                && bar.open > 0.0
                && bar.close > 0.0
                && bar.high > 0.0
                && bar.low > 0.0)
        }) {
            excluded.push((code, "invalid_price"));
            continue;
        }
        stocks.push((code, day_bars));
    }
    stocks.sort_by(|left, right| left.0.cmp(&right.0));
    excluded.sort();
    excluded.dedup();
    // A day whose every stock failed validation is still a valid (empty)
    // output: the exclusions are recorded and rolling windows simply see a hole.
    Ok(Some(DayData { stocks, excluded }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_parquet(path: &Path, rows: &[(String, u8, f64, f64, i64, f64)]) {
        use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray, UInt8Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("ts_code", DataType::Utf8, false),
            Field::new("minute_index", DataType::UInt8, false),
            Field::new("open", DataType::Float64, true),
            Field::new("high", DataType::Float64, true),
            Field::new("low", DataType::Float64, true),
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
                Arc::new(UInt8Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2.max(r.3)).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2.min(r.3)).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.4).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.5).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn load_day_requires_full_241_bar_sessions() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp
            .path()
            .join("year=2024")
            .join("month=01")
            .join("trade_date=2024-01-02");
        std::fs::create_dir_all(&dir).unwrap();
        let mut rows = Vec::new();
        for i in 0..SESSION_BARS {
            rows.push((
                "full.A".to_string(),
                i as u8,
                10.0,
                10.0 + i as f64 * 0.01,
                100,
                1_000.0,
            ));
        }
        for i in 0..200 {
            rows.push(("short.B".to_string(), i as u8, 5.0, 5.0, 100, 500.0));
        }
        write_test_parquet(&dir.join("part.parquet"), &rows);
        let day = load_day(temp.path(), "2024-01-02").unwrap().unwrap();
        assert_eq!(day.stocks.len(), 1);
        assert_eq!(day.stocks[0].0, "full.A");
        assert_eq!(
            day.excluded,
            vec![("short.B".to_string(), "bar_count_invalid")]
        );
    }

    #[test]
    fn load_day_flags_nonpositive_prices() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp
            .path()
            .join("year=2024")
            .join("month=01")
            .join("trade_date=2024-01-02");
        std::fs::create_dir_all(&dir).unwrap();
        let mut rows = Vec::new();
        for i in 0..SESSION_BARS {
            rows.push((
                "bad.C".to_string(),
                i as u8,
                10.0,
                if i == 5 { 0.0 } else { 10.0 },
                100,
                1_000.0,
            ));
        }
        write_test_parquet(&dir.join("part.parquet"), &rows);
        let day = load_day(temp.path(), "2024-01-02").unwrap().unwrap();
        assert!(day.stocks.is_empty());
        assert_eq!(day.excluded, vec![("bad.C".to_string(), "invalid_price")]);
    }

    #[test]
    fn missing_partition_returns_none() {
        let temp = tempfile::tempdir().unwrap();
        assert!(load_day(temp.path(), "2024-01-02").unwrap().is_none());
    }
}
