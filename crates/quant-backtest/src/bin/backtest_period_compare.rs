use anyhow::Result;
use clap::Parser;
use duckdb::Connection;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Compare two portfolio ledgers with the same compounded-active-return metrics")]
struct Args {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Serialize)]
struct PeriodMetrics {
    days: usize,
    portfolio_return: f64,
    csi500_return: f64,
    compounded_excess_return: f64,
    excess_sharpe_243: f64,
    average_buy_turnover: f64,
}

#[derive(Serialize)]
struct Comparison {
    baseline: BTreeMap<String, PeriodMetrics>,
    candidate: BTreeMap<String, PeriodMetrics>,
}

fn sql_quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn load(conn: &Connection, path: &Path) -> Result<BTreeMap<String, PeriodMetrics>> {
    let metrics = "count(*), \
         exp(sum(ln(1+net_return)))-1, \
         exp(sum(ln(1+csi500_return)))-1, \
         exp(sum(ln(1+net_return-csi500_return)))-1, \
         avg(net_return-csi500_return)/stddev_samp(net_return-csi500_return)*sqrt(243), \
         avg(buy_turnover)";
    let path = sql_quote(path);
    let sql = format!(
        "SELECT 'ALL', {metrics} FROM read_parquet('{path}') \
         UNION ALL \
         SELECT CAST(year(execution_date) AS VARCHAR), {metrics} \
         FROM read_parquet('{path}') GROUP BY 1 ORDER BY 1"
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            PeriodMetrics {
                days: row.get::<_, i64>(1)? as usize,
                portfolio_return: row.get(2)?,
                csi500_return: row.get(3)?,
                compounded_excess_return: row.get(4)?,
                excess_sharpe_243: row.get(5)?,
                average_buy_turnover: row.get(6)?,
            },
        ))
    })?;
    Ok(rows.collect::<duckdb::Result<BTreeMap<_, _>>>()?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let conn = Connection::open_in_memory()?;
    let result = Comparison {
        baseline: load(&conn, &args.baseline)?,
        candidate: load(&conn, &args.candidate)?,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
