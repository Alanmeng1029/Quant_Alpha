mod engine;
mod formulas;
mod panel;
mod registry;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use duckdb::Connection;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "quant-daily-factor",
    about = "Native Rust raw Daily60 production"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate that a frozen factor list is exactly the production Daily60 registry.
    ValidateRegistry {
        #[arg(long)]
        factor_ids: PathBuf,
    },
    /// Build Daily60. Enabled only after all formulas pass oracle parity.
    Build(BuildArgs),
    /// Development-only partial build used for formula oracle parity.
    DevBuild(BuildArgs),
}

#[derive(clap::Args)]
struct BuildArgs {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    date: String,
}

fn validate(path: &PathBuf) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let actual = registry::parse_ids(&text);
    if actual != registry::FACTOR_NAMES {
        bail!("factor registry differs from frozen production Daily60 order");
    }
    println!(
        "{}",
        serde_json::json!({"factor_count":actual.len(),"status":"ok"})
    );
    Ok(())
}

fn build(args: BuildArgs) -> Result<()> {
    if formulas::implemented_names().len() != 60 {
        bail!(
            "Daily60 build is release-gated: {}/60 native formulas implemented",
            formulas::implemented_names().len()
        )
    }
    build_inner(args, true)
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn build_inner(args: BuildArgs, production: bool) -> Result<()> {
    let panel = panel::load(&args.catalog, &args.date)?;
    let names = if production {
        registry::FACTOR_NAMES.to_vec()
    } else {
        formulas::implemented_names()
    };
    let mut columns = Vec::with_capacity(names.len());
    for name in &names {
        let values = formulas::compute(name, &panel)
            .with_context(|| format!("formula not implemented: {name}"))?;
        columns.push(panel.target(&values));
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = args.output.with_extension("parquet.tmp");
    let tsv = args.output.with_extension("tsv.tmp");
    let mut writer = BufWriter::new(File::create(&tsv)?);
    writeln!(writer, "trade_date\tts_code\t{}", names.join("\t"))?;
    for (row, code_index) in panel.target_members.iter().enumerate() {
        write!(writer, "{}\t{}", args.date, panel.codes[*code_index])?;
        for values in &columns {
            let value = values[row];
            if value.is_finite() {
                write!(writer, "\t{value:.17}")?;
            } else {
                write!(writer, "\t")?;
            }
        }
        writeln!(writer)?;
    }
    drop(writer);
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(&format!(
        "COPY (SELECT * REPLACE(trade_date::DATE AS trade_date) FROM read_csv('{}',delim='\\t',header=true,nullstr='')) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD)",
        quote(&tsv), quote(&temporary)
    ))?;
    if args.output.exists() {
        fs::remove_file(&args.output)?;
    }
    fs::rename(&temporary, &args.output)?;
    fs::remove_file(tsv)?;
    let manifest = serde_json::json!({
        "engine":"quant-daily-factor-rust-v1", "production":production,
        "trade_date":args.date, "factor_count":names.len(), "factor_ids":names,
        "rows":panel.target_members.len(), "calendar_sessions":panel.dates.len(),
        "calculation_codes":panel.codes.len()
    });
    fs::write(
        args.output.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("{manifest}");
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::ValidateRegistry { factor_ids } => validate(&factor_ids),
        Command::Build(args) => build(args),
        Command::DevBuild(args) => build_inner(args, false),
    }
}
