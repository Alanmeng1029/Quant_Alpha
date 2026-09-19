use anyhow::{Context, Result, bail};
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Build a wide feature cache from long daily factors and wide minute factors")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    daily_root: PathBuf,
    #[arg(long)]
    daily_ids: PathBuf,
    #[arg(long, value_parser = parse_source)]
    minute_source: Vec<(PathBuf, PathBuf)>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "000300.SH,000905.SH,000852.SH")]
    index_code: String,
    #[arg(long, default_value = "raw")]
    price_basis: String,
    #[arg(long, default_value = "2018-01-01")]
    start: String,
    #[arg(long, default_value = "2026-08-28")]
    end: String,
    #[arg(long, default_value_t = 4096)]
    memory_limit_mb: usize,
}

fn parse_source(value: &str) -> Result<(PathBuf, PathBuf), String> {
    let (root, ids) = value.split_once('=').ok_or("expected DATASET=IDS_FILE")?;
    Ok((PathBuf::from(root), PathBuf::from(ids)))
}

fn ids(path: &Path) -> Result<Vec<String>> {
    let values = fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .lines()
        .filter_map(|line| line.split('#').next().map(str::trim))
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if values.is_empty() {
        bail!("empty factor list: {}", path.display());
    }
    Ok(values)
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}
fn ident(value: &str) -> Result<String> {
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        bail!("unsafe identifier: {value}");
    }
    Ok(format!("\"{value}\""))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !matches!(args.price_basis.as_str(), "qfq" | "raw") {
        bail!("--price-basis must be qfq or raw");
    }
    let index_codes = args
        .index_code
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if index_codes.is_empty()
        || index_codes.iter().any(|code| {
            code.is_empty() || !code.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.')
        })
    {
        bail!("invalid --index-code list: {}", args.index_code);
    }
    let index_values = index_codes
        .iter()
        .map(|code| format!("'{code}'"))
        .collect::<Vec<_>>()
        .join(",");
    let daily_ids = ids(&args.daily_ids)?;
    let minute = args
        .minute_source
        .iter()
        .map(|(root, file)| Ok((root.clone(), ids(file)?)))
        .collect::<Result<Vec<_>>>()?;
    let mut all_ids = daily_ids.clone();
    for (_, values) in &minute {
        all_ids.extend(values.iter().cloned());
    }
    if all_ids.len() != 105 {
        bail!("expected 105 factors, found {}", all_ids.len());
    }

    fs::create_dir_all(&args.output)?;
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(&args.catalog, config)?;
    conn.execute_batch(&format!(
        "SET threads=1; SET memory_limit='{}MB'; SET preserve_insertion_order=false;",
        args.memory_limit_mb
    ))?;
    let start_year: i32 = args.start[..4].parse()?;
    let end_year: i32 = args.end[..4].parse()?;
    let mut rows = 0_i64;
    for year in start_year..=end_year {
        let lower = if year == start_year {
            args.start.clone()
        } else {
            format!("{year}-01-01")
        };
        let upper = if year == end_year {
            args.end.clone()
        } else {
            format!("{year}-12-31")
        };
        let unions = daily_ids.iter().map(|factor| {
            let stem = factor.strip_suffix("_v1").unwrap_or(factor);
            let path = args.daily_root.join(stem).join("v1/factor.parquet");
            if !path.exists() { bail!("missing {}", path.display()); }
            Ok(format!("SELECT trade_date,ts_code,factor_value,'{factor}' factor_id FROM read_parquet('{}') WHERE trade_date BETWEEN DATE '{lower}' AND DATE '{upper}'", quote(&path)))
        }).collect::<Result<Vec<_>>>()?.join(" UNION ALL ");
        let daily_cols = daily_ids
            .iter()
            .map(|factor| {
                Ok(format!(
                    "max(f.factor_value) FILTER(WHERE f.factor_id='{factor}')::FLOAT AS {}",
                    ident(factor)?
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let universe = if args.price_basis == "raw" {
            format!(
                "universe AS (SELECT DISTINCT cal.trade_date,c.ts_code FROM observed_calendar cal JOIN index_monthly_constituents c ON c.index_code IN ({index_values}) AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date) JOIN daily_aggregated d ON d.trade_date=cal.trade_date AND d.ts_code=c.ts_code WHERE cal.is_observed_market_day AND cal.trade_date BETWEEN DATE '{lower}' AND DATE '{upper}' AND d.open>0 AND d.high>0 AND d.low>0 AND d.close>0 AND d.volume_share>0 AND d.amount_cny>0 AND d.observation_status='complete_trading' AND c.ts_code<>'000937.SZ')"
            )
        } else {
            format!(
                "universe AS (SELECT DISTINCT trade_date,ts_code FROM index_trading_universe WHERE index_code IN ({index_values}) AND trade_date BETWEEN DATE '{lower}' AND DATE '{upper}' AND ts_code<>'000937.SZ')"
            )
        };
        let mut ctes = vec![universe, format!("factors AS ({unions})")];
        let mut joins = Vec::new();
        let mut minute_cols = Vec::new();
        for (i, (root, values)) in minute.iter().enumerate() {
            let selected = values
                .iter()
                .map(|v| ident(v))
                .collect::<Result<Vec<_>>>()?
                .join(",");
            ctes.push(format!("m{i} AS (SELECT trade_date,ts_code,{selected} FROM read_parquet('{}/year={year}/*.parquet') WHERE trade_date BETWEEN DATE '{lower}' AND DATE '{upper}')", quote(root)));
            joins.push(format!("LEFT JOIN m{i} USING(trade_date,ts_code)"));
            minute_cols.extend(
                values
                    .iter()
                    .map(|v| ident(v).map(|id| format!("max(m{i}.{id})::FLOAT AS {id}")))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let columns = daily_cols
            .into_iter()
            .chain(minute_cols)
            .collect::<Vec<_>>()
            .join(",");
        let year_dir = args.output.join(format!("year={year}"));
        fs::create_dir_all(&year_dir)?;
        let destination = year_dir.join("features.parquet");
        let temporary = year_dir.join("features.parquet.tmp");
        if temporary.exists() {
            fs::remove_file(&temporary)?;
        }
        let query = format!(
            "WITH {} SELECT u.trade_date,u.ts_code,{columns} FROM universe u LEFT JOIN factors f USING(trade_date,ts_code) {} GROUP BY u.trade_date,u.ts_code ORDER BY u.trade_date,u.ts_code",
            ctes.join(","),
            joins.join(" ")
        );
        conn.execute_batch(&format!(
            "COPY ({query}) TO '{}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)",
            quote(&temporary)
        ))?;
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(&temporary, &destination)?;
        let count: i64 = conn.query_row(
            &format!(
                "SELECT count(*) FROM read_parquet('{}')",
                quote(&destination)
            ),
            [],
            |row| row.get(0),
        )?;
        rows += count;
        eprintln!("{year}: {count} rows");
    }
    fs::write(
        args.output.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "version": 2, "factor_ids": all_ids, "daily_factor_ids": daily_ids,
            "minute_factor_sources": minute.iter().map(|(root, values)| json!({"dataset":root,"factor_ids":values})).collect::<Vec<_>>(),
            "price_basis": args.price_basis, "index_code": args.index_code, "start": args.start, "end": args.end, "rows": rows,
            "builder": "quant-lgbm-train/build-feature-cache-rust-v1"
        }))?,
    )?;
    println!(
        "{}",
        json!({"rows":rows,"factors":105,"output":args.output})
    );
    Ok(())
}
