use crate::engine::{Series, Shape};
use anyhow::{Context, Result, bail};
use duckdb::{AccessMode, Config, Connection};
use std::collections::HashMap;
use std::path::Path;

pub struct Panel {
    pub shape: Shape,
    pub dates: Vec<String>,
    pub codes: Vec<String>,
    pub open: Series,
    pub high: Series,
    pub low: Series,
    pub close: Series,
    pub volume: Series,
    pub vwap: Series,
    pub index_open: Series,
    pub index_close: Series,
    pub target_members: Vec<usize>,
}

impl Panel {
    pub fn target(&self, values: &[f64]) -> Vec<f64> {
        let date = self.shape.dates - 1;
        self.target_members
            .iter()
            .map(|code| values[self.shape.at(date, *code)])
            .collect()
    }
}

pub fn load(catalog: &Path, target: &str) -> Result<Panel> {
    let config = Config::default().access_mode(AccessMode::ReadOnly)?;
    let conn = Connection::open_with_flags(catalog, config)?;
    let has_market_view: bool = conn.query_row(
        "SELECT count(*)>0 FROM duckdb_views() WHERE view_name='market_daily_aggregated'",
        [],
        |row| row.get(0),
    )?;
    let daily_source = if has_market_view {
        "market_daily_aggregated"
    } else {
        "daily_aggregated"
    };
    let calendar_source = if has_market_view {
        "(SELECT trade_date FROM observed_calendar WHERE is_observed_market_day UNION SELECT DISTINCT trade_date FROM market_daily_aggregated)"
    } else {
        "(SELECT trade_date FROM observed_calendar WHERE is_observed_market_day)"
    };
    // Recursive EWM factors must carry effectively all prior state.  A 1024-session
    // burn-in makes the slowest decay (alpha=1/20) smaller than 2e-23 while
    // keeping the dense panel bounded for a daily production run.
    let mut dates=conn.prepare(&format!("SELECT trade_date::VARCHAR FROM (SELECT trade_date FROM {calendar_source} WHERE trade_date<=?::DATE ORDER BY trade_date DESC LIMIT 1025) ORDER BY trade_date"))?
        .query_map([target],|r|r.get(0))?.collect::<std::result::Result<Vec<String>,_>>()?;
    if dates.last().map(String::as_str) != Some(target) {
        bail!("{target} is not an observed market date")
    }
    if dates.len() < 255 {
        bail!(
            "need 254 prior sessions, found {}",
            dates.len().saturating_sub(1)
        )
    }
    let begin = &dates[0];
    let mut codes=conn.prepare(&format!("SELECT DISTINCT ts_code FROM {daily_source} WHERE trade_date BETWEEN ?::DATE AND ?::DATE AND open>0 AND high>0 AND low>0 AND close>0 AND volume_share>0 ORDER BY ts_code"))?
        .query_map([begin,target],|r|r.get(0))?.collect::<std::result::Result<Vec<String>,_>>()?;
    let shape = Shape {
        dates: dates.len(),
        codes: codes.len(),
    };
    let date_index = dates
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect::<HashMap<_, _>>();
    let code_index = codes
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect::<HashMap<_, _>>();
    let mut open = vec![f64::NAN; shape.len()];
    let mut high = open.clone();
    let mut low = open.clone();
    let mut close = open.clone();
    let mut volume = open.clone();
    let mut vwap = open.clone();
    let mut stmt=conn.prepare(&format!("SELECT trade_date::VARCHAR,ts_code,open::DOUBLE,high::DOUBLE,low::DOUBLE,close::DOUBLE,volume_share::DOUBLE,amount_cny/nullif(volume_share,0)::DOUBLE FROM {daily_source} WHERE trade_date BETWEEN ?::DATE AND ?::DATE AND open>0 AND high>0 AND low>0 AND close>0 AND volume_share>0 ORDER BY trade_date,ts_code"))?;
    for row in stmt.query_map([begin, target], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, f64>(2)?,
            r.get::<_, f64>(3)?,
            r.get::<_, f64>(4)?,
            r.get::<_, f64>(5)?,
            r.get::<_, f64>(6)?,
            r.get::<_, Option<f64>>(7)?,
        ))
    })? {
        let (d, c, o, h, l, cl, v, vw) = row?;
        let i = shape.at(date_index[&d], code_index[&c]);
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = cl;
        volume[i] = v;
        vwap[i] = vw.unwrap_or(f64::NAN);
    }
    let mut index_by_date = HashMap::new();
    for row in conn.prepare("SELECT trade_date::VARCHAR,avg(open)::DOUBLE,avg(close)::DOUBLE FROM index_daily WHERE index_code IN ('000300.SH','000905.SH') AND trade_date BETWEEN ?::DATE AND ?::DATE GROUP BY trade_date")?.query_map([begin,target],|r|Ok((r.get::<_,String>(0)?,r.get::<_,f64>(1)?,r.get::<_,f64>(2)?)))?{let(d,o,c)=row?;index_by_date.insert(d,(o,c));}
    let mut index_open = vec![f64::NAN; shape.len()];
    let mut index_close = index_open.clone();
    for (d, date) in dates.iter().enumerate() {
        if let Some((o, c)) = index_by_date.get(date) {
            for code in 0..shape.codes {
                let i = shape.at(d, code);
                index_open[i] = *o;
                index_close[i] = *c
            }
        }
    }
    // The legacy index feed can lag the full-market FTShare feed.  Extend the
    // benchmark causally with the equal-weight cross-sectional return so the
    // sole benchmark-dependent Daily60 formula remains finite on fresh days.
    for d in 1..shape.dates {
        let probe = shape.at(d, 0);
        if index_close[probe].is_finite() {
            continue;
        }
        let previous = index_close[shape.at(d - 1, 0)];
        if !previous.is_finite() || previous <= 0.0 {
            continue;
        }
        let mut close_log_sum = 0.0;
        let mut open_log_sum = 0.0;
        let mut count = 0usize;
        for code in 0..shape.codes {
            let prior_close = close[shape.at(d - 1, code)];
            let current_close = close[shape.at(d, code)];
            let current_open = open[shape.at(d, code)];
            if prior_close > 0.0 && current_close > 0.0 && current_open > 0.0 {
                close_log_sum += (current_close / prior_close).ln();
                open_log_sum += (current_open / prior_close).ln();
                count += 1;
            }
        }
        if count > 0 {
            let benchmark_close = previous * (close_log_sum / count as f64).exp();
            let benchmark_open = previous * (open_log_sum / count as f64).exp();
            for code in 0..shape.codes {
                let i = shape.at(d, code);
                index_open[i] = benchmark_open;
                index_close[i] = benchmark_close;
            }
        }
    }
    let member_sql = format!(
        "SELECT DISTINCT c.ts_code FROM index_monthly_constituents c JOIN {daily_source} d ON d.ts_code=c.ts_code AND d.trade_date=?::DATE WHERE c.index_code IN ('000300.SH','000905.SH','000852.SH') AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=?::DATE) AND d.open>0 AND d.high>0 AND d.low>0 AND d.close>0 AND d.volume_share>0 AND d.amount_cny>0 AND d.observation_status='complete_trading' AND c.ts_code<>'000937.SZ' ORDER BY c.ts_code"
    );
    let members = conn
        .prepare(&member_sql)?
        .query_map([target, target], |r| r.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let target_members = members
        .iter()
        .map(|c| {
            code_index
                .get(c)
                .copied()
                .with_context(|| format!("eligible member {c} absent from calculation panel"))
        })
        .collect::<Result<Vec<_>>>()?;
    if target_members.is_empty() {
        bail!("empty target universe")
    }
    Ok(Panel {
        shape,
        dates: std::mem::take(&mut dates),
        codes: std::mem::take(&mut codes),
        open,
        high,
        low,
        close,
        volume,
        vwap,
        index_open,
        index_close,
        target_members,
    })
}
