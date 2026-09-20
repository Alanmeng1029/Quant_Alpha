//! Block pipeline: process a contiguous run of market days with warm-up.
//!
//! Each block owns an independent `RollingState`; blocks overlap only in their
//! read-only warm-up replay, so block layout cannot influence output values.
//! Within a day, per-stock factor math is parallel and independent; the state
//! advance happens sequentially after all finalizations, which keeps results
//! bit-identical for any thread count.
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::daily;
use crate::loader::{self, DayData};
use crate::state::RollingState;
use crate::writer::{self, WideRow};

/// Read-only catalog inputs shared by one block job.
pub struct MarketContext {
    /// Market calendar positions for every observed trading day, ascending.
    pub calendar: Vec<String>,
    /// Position of a trade date inside `calendar` for O(1) day stamps.
    positions: HashMap<String, u32>,
    /// Configured point-in-time index-union membership per trade date.
    pub universe: HashMap<String, HashSet<String>>,
}

impl MarketContext {
    pub fn day_index(&self, trade_date: &str) -> u32 {
        self.positions[trade_date]
    }

    pub fn contains(&self, trade_date: &str, code: &str) -> bool {
        self.universe
            .get(trade_date)
            .is_some_and(|codes| codes.contains(code))
    }
}

pub fn load_market_context(
    catalog: &Path,
    start: &str,
    end: &str,
    memory_limit_mb: usize,
    index_codes: &[String],
    raw_eligible_universe: bool,
) -> Result<MarketContext> {
    if index_codes.is_empty() {
        anyhow::bail!("at least one --index-codes entry is required");
    }
    let config = duckdb::Config::default()
        .access_mode(duckdb::AccessMode::ReadOnly)
        .with_context(|| format!("configure read-only catalog {}", catalog.display()))?;
    let conn = duckdb::Connection::open_with_flags(catalog, config)
        .with_context(|| format!("open catalog {}", catalog.display()))?;
    conn.execute_batch(&format!(
        "SET threads TO 1; SET memory_limit='{}MB'; SET preserve_insertion_order=false;",
        memory_limit_mb.max(1)
    ))?;

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

    let calendar: Vec<String> = conn
        .prepare(&format!(
            "SELECT trade_date::VARCHAR FROM {calendar_source} ORDER BY trade_date"
        ))?
        .query_map([], |row| row.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut positions = HashMap::with_capacity(calendar.len());
    for (index, date) in calendar.iter().enumerate() {
        positions.insert(date.clone(), index as u32);
    }

    let universe: HashMap<String, HashSet<String>> = {
        let placeholders = std::iter::repeat("?")
            .take(index_codes.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = if raw_eligible_universe {
            format!(
                "SELECT cal.trade_date::VARCHAR, c.ts_code FROM {calendar_source} cal \
             JOIN index_monthly_constituents c ON c.index_code IN ({placeholders}) \
              AND c.as_of_date=(SELECT max(c2.as_of_date) FROM index_monthly_constituents c2 WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date) \
             JOIN {daily_source} d ON d.trade_date=cal.trade_date AND d.ts_code=c.ts_code \
             WHERE cal.trade_date BETWEEN ?::DATE AND ?::DATE \
              AND d.open>0 AND d.high>0 AND d.low>0 AND d.close>0 AND d.volume_share>0 AND d.amount_cny>0 AND d.observation_status='complete_trading' \
             GROUP BY cal.trade_date,c.ts_code"
            )
        } else {
            format!(
                "SELECT trade_date::VARCHAR, ts_code FROM index_trading_universe \
             WHERE index_code IN ({placeholders}) AND trade_date BETWEEN ?::DATE AND ?::DATE \
             GROUP BY trade_date, ts_code"
            )
        };
        let mut statement = conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> = index_codes
            .iter()
            .map(|code| code as &dyn duckdb::ToSql)
            .collect();
        params.push(&start);
        params.push(&end);
        let rows = statement.query_map(params.as_slice(), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut universe: HashMap<String, HashSet<String>> = HashMap::new();
        for row in rows {
            let (date, code) = row?;
            universe.entry(date).or_default().insert(code);
        }
        universe
    };
    Ok(MarketContext {
        calendar,
        positions,
        universe,
    })
}

pub struct DayOutcome {
    pub status: String,
    pub rows: usize,
    pub excluded: std::collections::BTreeMap<String, usize>,
    pub elapsed_seconds: f64,
}

/// How a day participates in the rolling pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DayMode {
    /// Full pipeline: finalize rolling factors, write the wide parquet.
    Produce,
    /// Advance the rolling state only - no finalize, no output file.  Used
    /// for warm-up days, days predating the universe, and days whose output
    /// already exists: their history must still feed later days' baselines.
    Replay,
}

/// Compute, write, and roll forward one trade date.  A missing minute
/// partition records `missing_partition` so a source gap cannot silently
/// truncate windows.
pub fn process_day(
    minute_root: &Path,
    output_root: &Path,
    context: &MarketContext,
    state: &mut RollingState,
    pool: &rayon::ThreadPool,
    trade_date: &str,
    day_index: u32,
    mode: DayMode,
) -> Result<DayOutcome> {
    let started = std::time::Instant::now();
    let mut excluded = std::collections::BTreeMap::new();

    let day: Option<DayData> = loader::load_day(minute_root, trade_date)?;
    let rows: Vec<WideRow> = match day {
        None => {
            return Ok(DayOutcome {
                status: "missing_partition".to_string(),
                rows: 0,
                excluded,
                elapsed_seconds: started.elapsed().as_secs_f64(),
            });
        }
        Some(day) => {
            for (_, reason) in &day.excluded {
                *excluded.entry((*reason).to_string()).or_default() += 1;
            }
            if mode == DayMode::Replay {
                // State-only pass: compute_daily is pure, finalize is skipped
                // because nothing is written for this day.
                let raws: Vec<(String, daily::StockDayRaw)> = pool.install(|| {
                    day.stocks
                        .par_iter()
                        .map(|(code, bars)| (code.clone(), daily::compute_daily(bars)))
                        .collect()
                });
                for (code, raw) in raws {
                    state.advance(day_index, &code, &raw);
                }
                Vec::new()
            } else {
                let computed: Vec<(
                    String,
                    [Option<f64>; crate::schema::N_FACTORS],
                    daily::StockDayRaw,
                )> = pool.install(|| {
                    day.stocks
                        .par_iter()
                        .map(|(code, bars)| {
                            let raw = daily::compute_daily(bars);
                            // finalize reads baselines recorded strictly before
                            // today, so the shared state is read-only here.
                            let values = daily::finalize(code, day_index, &raw, state);
                            (code.clone(), values, raw)
                        })
                        .collect()
                });

                let mut rows: Vec<WideRow> = Vec::with_capacity(computed.len());
                for (code, values, raw) in computed {
                    if context.contains(trade_date, &code) {
                        rows.push(WideRow {
                            ts_code: code.clone(),
                            values,
                        });
                    }
                    state.advance(day_index, &code, &raw);
                }
                rows
            }
        }
    };

    let written = if mode == DayMode::Produce {
        let count = rows.len();
        writer::write_day(output_root, trade_date, &rows)?;
        count
    } else {
        0
    };

    Ok(DayOutcome {
        status: "ok".to_string(),
        rows: written,
        excluded,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

/// A production block: `[warmup_begin, target_end]` is replayed through the
/// rolling state, but only `[target_begin, target_end]` writes output.
#[derive(Debug)]
pub struct Block {
    pub warmup_begin: usize,
    pub target_begin: usize,
    pub target_end: usize,
}

/// Split the requested date range into contiguous blocks of `block_days`
/// market days, each extended by `warmup` prior market days.  Target ranges
/// tile the request without gaps or overlap.  `start`/`end` need not fall on
/// trading days: the range is clamped to the first trading day on or after
/// `start` and the last trading day on or before `end`.
pub fn plan_blocks(
    calendar: &[String],
    start: &str,
    end: &str,
    block_days: usize,
    warmup: usize,
) -> Vec<Block> {
    // ISO dates compare lexicographically, so partition_point is a range
    // clamp; a binary_search miss must never fall back to the calendar edge.
    let first = calendar.partition_point(|day| day.as_str() < start);
    let last = calendar.partition_point(|day| day.as_str() <= end);
    if first >= last {
        return Vec::new();
    }
    let last = last - 1;
    let mut blocks = Vec::new();
    let mut target_begin = first;
    while target_begin <= last {
        let target_end = (target_begin + block_days - 1).min(last);
        let warmup_begin = target_begin.saturating_sub(warmup);
        blocks.push(Block {
            warmup_begin,
            target_begin,
            target_end,
        });
        target_begin = target_end + 1;
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calendar(dates: &[&str]) -> MarketContext {
        let calendar: Vec<String> = dates.iter().map(|d| d.to_string()).collect();
        let positions = calendar
            .iter()
            .enumerate()
            .map(|(i, d)| (d.clone(), i as u32))
            .collect();
        MarketContext {
            calendar,
            positions,
            universe: HashMap::new(),
        }
    }

    #[test]
    fn blocks_tile_without_gaps_and_warmup_overlaps_history() {
        let context = calendar(&[
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
        ]);
        let blocks = plan_blocks(&context.calendar, "2024-01-01", "2024-01-12", 4, 2);
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            (
                blocks[0].warmup_begin,
                blocks[0].target_begin,
                blocks[0].target_end
            ),
            (0, 0, 3)
        );
        assert_eq!(
            (
                blocks[1].warmup_begin,
                blocks[1].target_begin,
                blocks[1].target_end
            ),
            (2, 4, 7)
        );
        assert_eq!(
            (
                blocks[2].warmup_begin,
                blocks[2].target_begin,
                blocks[2].target_end
            ),
            (6, 8, 9)
        );
        let mut covered = std::collections::BTreeSet::new();
        for block in &blocks {
            for day in block.target_begin..=block.target_end {
                covered.insert(day);
            }
        }
        assert_eq!(covered.len(), 10);
    }

    #[test]
    fn range_outside_calendar_yields_no_blocks() {
        let context = calendar(&["2024-01-01", "2024-01-02"]);
        assert!(plan_blocks(&context.calendar, "2023-12-25", "2023-12-29", 4, 2).is_empty());
    }

    #[test]
    fn weekend_boundaries_clamp_to_enclosed_trading_days() {
        // 2024-01-06/07 is a weekend; the request is 2024-01-01..2024-01-07.
        let context = calendar(&[
            "2023-12-29",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ]);
        let blocks = plan_blocks(&context.calendar, "2024-01-01", "2024-01-07", 4, 2);
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            (
                blocks[0].warmup_begin,
                blocks[0].target_begin,
                blocks[0].target_end
            ),
            (0, 1, 4)
        );
        assert_eq!(
            (
                blocks[1].warmup_begin,
                blocks[1].target_begin,
                blocks[1].target_end
            ),
            (3, 5, 5)
        );
        // A weekend-only range produces nothing instead of leaking history.
        assert!(plan_blocks(&context.calendar, "2024-01-06", "2024-01-07", 4, 2).is_empty());
        // A start on a weekend clamps forward to the next trading day.
        let blocks = plan_blocks(&context.calendar, "2024-01-06", "2024-01-07", 4, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn end_clamps_to_last_calendar_day() {
        let context = calendar(&["2024-01-01", "2024-01-02"]);
        let blocks = plan_blocks(&context.calendar, "2024-01-01", "2030-01-01", 4, 2);
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            (
                blocks[0].warmup_begin,
                blocks[0].target_begin,
                blocks[0].target_end
            ),
            (0, 0, 1)
        );
    }
}
