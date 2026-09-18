use anyhow::Result;
use clap::Parser;
use duckdb::{AccessMode, Config, Connection};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Compare two prediction rankings inside the point-in-time CSI500 universe")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Serialize)]
struct PeriodComparison {
    period: String,
    days: usize,
    mean_rank_correlation: f64,
    mean_top100_overlap: f64,
    mean_common_names: f64,
}

fn quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(&args.catalog, config)?;
    conn.execute_batch("SET threads=1; SET memory_limit='2GB';")?;
    let baseline = quote(&args.baseline);
    let candidate = quote(&args.candidate);
    let sql = format!(
        r#"
        WITH common AS (
          SELECT b.trade_date, b.ts_code,
                 b.pred_h1 AS b_h1, b.pred_h5 AS b_h5,
                 c.pred_h1 AS c_h1, c.pred_h5 AS c_h5
          FROM read_parquet('{baseline}') b
          JOIN read_parquet('{candidate}') c USING (trade_date, ts_code)
          JOIN index_trading_universe u
            ON u.trade_date=CAST(b.trade_date AS DATE) AND u.ts_code=b.ts_code
          WHERE u.index_code='000905.SH'
            AND b.pred_h1 IS NOT NULL AND b.pred_h5 IS NOT NULL
            AND c.pred_h1 IS NOT NULL AND c.pred_h5 IS NOT NULL
        ), standardized AS (
          SELECT trade_date, ts_code,
                 0.5 * (b_h1-avg(b_h1) OVER w)/nullif(stddev_samp(b_h1) OVER w,0)
                   + 0.5 * (b_h5-avg(b_h5) OVER w)/nullif(stddev_samp(b_h5) OVER w,0) AS b_score,
                 0.5 * (c_h1-avg(c_h1) OVER w)/nullif(stddev_samp(c_h1) OVER w,0)
                   + 0.5 * (c_h5-avg(c_h5) OVER w)/nullif(stddev_samp(c_h5) OVER w,0) AS c_score
          FROM common WINDOW w AS (PARTITION BY trade_date)
        ), ranked AS (
          SELECT *,
                 row_number() OVER (PARTITION BY trade_date ORDER BY b_score DESC, ts_code) AS b_rank,
                 row_number() OVER (PARTITION BY trade_date ORDER BY c_score DESC, ts_code) AS c_rank
          FROM standardized
        ), daily AS (
          SELECT trade_date,
                 corr(b_rank, c_rank) AS rank_corr,
                 sum(CASE WHEN b_rank<=100 AND c_rank<=100 THEN 1 ELSE 0 END)/100.0 AS top100_overlap,
                 count(*) AS common_names
          FROM ranked GROUP BY trade_date
        ), periods AS (
          SELECT 'ALL' AS period, count(*) AS days, avg(rank_corr) AS rank_corr,
                 avg(top100_overlap) AS top100_overlap, avg(common_names) AS common_names
          FROM daily
          UNION ALL
          SELECT CAST(year(CAST(trade_date AS DATE)) AS VARCHAR), count(*), avg(rank_corr),
                 avg(top100_overlap), avg(common_names)
          FROM daily GROUP BY 1
        )
        SELECT period, days, rank_corr, top100_overlap, common_names
        FROM periods ORDER BY period
        "#
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map([], |row| {
        Ok(PeriodComparison {
            period: row.get(0)?,
            days: row.get::<_, i64>(1)? as usize,
            mean_rank_correlation: row.get(2)?,
            mean_top100_overlap: row.get(3)?,
            mean_common_names: row.get(4)?,
        })
    })?;
    let result = rows.collect::<duckdb::Result<Vec<_>>>()?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
