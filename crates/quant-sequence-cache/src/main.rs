//! Convert a sorted, standardized daily panel into a compact mmap contract.
//!
//! The output deliberately stores each stock-day only once.  `sample_rows.u64`
//! contains the ending row of every complete market-calendar sequence, so the
//! Python trainer can gather `[end-L+1, ..., end]` in vectorized batches.
use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, ArrayRef, Date32Array, Float32Array, Int32Array, StringArray, UInt64Array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    factor_ids: PathBuf,
    #[arg(long, default_value_t = 20)]
    sequence_length: usize,
    #[arg(long)]
    fingerprint: String,
}

#[derive(Serialize)]
struct Manifest {
    version: u32,
    status: String,
    fingerprint: String,
    rows: u64,
    samples: u64,
    sequence_length: usize,
    factor_count: usize,
    input_channels: usize,
    factor_ids: Vec<String>,
    values: String,
    missing: String,
    targets: String,
    sample_rows: String,
    metadata: String,
    missing_indicator: String,
}

fn cast_array(batch: &RecordBatch, name: &str, dtype: &DataType) -> Result<ArrayRef> {
    cast(
        batch
            .column_by_name(name)
            .with_context(|| format!("missing column {name}"))?,
        dtype,
    )
    .with_context(|| format!("cast column {name}"))
}

fn flush_metadata(
    writer: &mut ArrowWriter<File>,
    schema: Arc<Schema>,
    rows: &mut Vec<u64>,
    days: &mut Vec<i32>,
    dates: &mut Vec<i32>,
    codes: &mut Vec<String>,
    h1: &mut Vec<Option<f32>>,
    h5: &mut Vec<Option<f32>>,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(std::mem::take(rows))) as ArrayRef,
            Arc::new(Int32Array::from(std::mem::take(days))) as ArrayRef,
            Arc::new(Date32Array::from(std::mem::take(dates))) as ArrayRef,
            Arc::new(StringArray::from(std::mem::take(codes))) as ArrayRef,
            Arc::new(Float32Array::from(std::mem::take(h1))) as ArrayRef,
            Arc::new(Float32Array::from(std::mem::take(h5))) as ArrayRef,
        ],
    )?;
    writer.write(&batch)?;
    Ok(())
}

fn build(args: &Args) -> Result<Manifest> {
    if args.sequence_length < 1 {
        bail!("sequence length must be positive");
    }
    if args.output.exists() {
        bail!("output already exists: {}", args.output.display());
    }
    let factor_ids: Vec<String> =
        serde_json::from_reader(File::open(&args.factor_ids).context("open factor id JSON")?)?;
    if factor_ids.is_empty() {
        bail!("factor list is empty");
    }
    let mut unique = factor_ids.clone();
    unique.sort();
    unique.dedup();
    if unique.len() != factor_ids.len() {
        bail!("factor list contains duplicates");
    }

    let input = File::open(&args.input).context("open input parquet")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input)?;
    let expected: Vec<String> = ["trade_date", "ts_code", "day_index"]
        .into_iter()
        .map(str::to_string)
        .chain(factor_ids.iter().cloned())
        .chain(["excess_h1", "excess_h5"].into_iter().map(str::to_string))
        .collect();
    let actual: Vec<String> = builder
        .schema()
        .fields()
        .iter()
        .map(|x| x.name().clone())
        .collect();
    if actual != expected {
        bail!("input schema/order differs from the declared sequence contract");
    }

    fs::create_dir_all(&args.output)?;
    let mut values = BufWriter::new(File::create(args.output.join("values.f32"))?);
    let mut missing = BufWriter::new(File::create(args.output.join("missing.u8"))?);
    let mut targets = BufWriter::new(File::create(args.output.join("targets.f32"))?);
    let mut samples = BufWriter::new(File::create(args.output.join("sample_rows.u64"))?);

    let metadata_schema = Arc::new(Schema::new(vec![
        Field::new("row_index", DataType::UInt64, false),
        Field::new("day_index", DataType::Int32, false),
        Field::new("trade_date", DataType::Date32, false),
        Field::new("ts_code", DataType::Utf8, false),
        Field::new("excess_h1", DataType::Float32, true),
        Field::new("excess_h5", DataType::Float32, true),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut metadata = ArrowWriter::try_new(
        File::create(args.output.join("metadata.parquet"))?,
        metadata_schema.clone(),
        Some(props),
    )?;

    let mut meta_rows = Vec::with_capacity(65_536);
    let mut meta_days = Vec::with_capacity(65_536);
    let mut meta_dates = Vec::with_capacity(65_536);
    let mut meta_codes = Vec::with_capacity(65_536);
    let mut meta_h1 = Vec::with_capacity(65_536);
    let mut meta_h5 = Vec::with_capacity(65_536);
    let mut history: VecDeque<(u64, i32)> = VecDeque::with_capacity(args.sequence_length);
    let mut previous_code: Option<String> = None;
    let mut previous_day: Option<i32> = None;
    let mut row_index = 0_u64;
    let mut sample_count = 0_u64;

    let reader = builder.with_batch_size(65_536).build()?;
    for batch in reader {
        let batch = batch?;
        let dates = cast_array(&batch, "trade_date", &DataType::Date32)?;
        let dates = dates.as_any().downcast_ref::<Date32Array>().unwrap();
        let codes = cast_array(&batch, "ts_code", &DataType::Utf8)?;
        let codes = codes.as_any().downcast_ref::<StringArray>().unwrap();
        let day_indices = cast_array(&batch, "day_index", &DataType::Int32)?;
        let day_indices = day_indices.as_any().downcast_ref::<Int32Array>().unwrap();
        let h1 = cast_array(&batch, "excess_h1", &DataType::Float32)?;
        let h1 = h1.as_any().downcast_ref::<Float32Array>().unwrap();
        let h5 = cast_array(&batch, "excess_h5", &DataType::Float32)?;
        let h5 = h5.as_any().downcast_ref::<Float32Array>().unwrap();
        let feature_arrays: Vec<ArrayRef> = factor_ids
            .iter()
            .map(|name| cast_array(&batch, name, &DataType::Float32))
            .collect::<Result<_>>()?;
        let feature_arrays: Vec<&Float32Array> = feature_arrays
            .iter()
            .map(|array| array.as_any().downcast_ref::<Float32Array>().unwrap())
            .collect();

        for i in 0..batch.num_rows() {
            if dates.is_null(i) || codes.is_null(i) || day_indices.is_null(i) {
                bail!("null sequence key at input row {row_index}");
            }
            let code = codes.value(i);
            let day = day_indices.value(i);
            if previous_code.as_deref() == Some(code) {
                if previous_day.is_some_and(|prior| day <= prior) {
                    bail!("duplicate or unsorted key for {code} at day index {day}");
                }
            } else {
                if previous_code.as_deref().is_some_and(|prior| code <= prior) {
                    bail!("stocks are not strictly sorted");
                }
                history.clear();
            }

            for array in &feature_arrays {
                let finite = !array.is_null(i) && array.value(i).is_finite();
                let value = if finite { array.value(i) } else { 0_f32 };
                values.write_all(&value.to_le_bytes())?;
                missing.write_all(&[u8::from(!finite)])?;
            }
            let target = |array: &Float32Array| {
                if !array.is_null(i) && array.value(i).is_finite() {
                    array.value(i)
                } else {
                    f32::NAN
                }
            };
            let y1 = target(h1);
            let y5 = target(h5);
            targets.write_all(&y1.to_le_bytes())?;
            targets.write_all(&y5.to_le_bytes())?;

            history.push_back((row_index, day));
            if history.len() > args.sequence_length {
                history.pop_front();
            }
            if history.len() == args.sequence_length
                && day - history.front().unwrap().1 == args.sequence_length as i32 - 1
            {
                samples.write_all(&row_index.to_le_bytes())?;
                sample_count += 1;
            }

            meta_rows.push(row_index);
            meta_days.push(day);
            meta_dates.push(dates.value(i));
            meta_codes.push(code.to_string());
            meta_h1.push(y1.is_finite().then_some(y1));
            meta_h5.push(y5.is_finite().then_some(y5));
            if meta_rows.len() == 65_536 {
                flush_metadata(
                    &mut metadata,
                    metadata_schema.clone(),
                    &mut meta_rows,
                    &mut meta_days,
                    &mut meta_dates,
                    &mut meta_codes,
                    &mut meta_h1,
                    &mut meta_h5,
                )?;
            }
            previous_code = Some(code.to_string());
            previous_day = Some(day);
            row_index += 1;
        }
    }
    flush_metadata(
        &mut metadata,
        metadata_schema,
        &mut meta_rows,
        &mut meta_days,
        &mut meta_dates,
        &mut meta_codes,
        &mut meta_h1,
        &mut meta_h5,
    )?;
    metadata.close()?;
    values.flush()?;
    missing.flush()?;
    targets.flush()?;
    samples.flush()?;

    Ok(Manifest {
        version: 1,
        status: "complete".into(),
        fingerprint: args.fingerprint.clone(),
        rows: row_index,
        samples: sample_count,
        sequence_length: args.sequence_length,
        factor_count: factor_ids.len(),
        input_channels: factor_ids.len() * 2,
        factor_ids,
        values: "values.f32".into(),
        missing: "missing.u8".into(),
        targets: "targets.f32".into(),
        sample_rows: "sample_rows.u64".into(),
        metadata: "metadata.parquet".into(),
        missing_indicator: "1 means missing or non-finite; values are zero-filled".into(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest = build(&args)?;
    let path = args.output.join("manifest.json.tmp");
    fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    fs::rename(path, args.output.join("manifest.json"))?;
    println!("{}", serde_json::to_string(&manifest)?);
    Ok(())
}
