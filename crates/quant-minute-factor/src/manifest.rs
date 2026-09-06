//! Root manifest: self-describing dataset with per-date status for resume.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::schema::{formulas, MIN_BASELINE_OBS, MIN_SMART_PRIOR_OBS};

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone)]
pub struct DayStatus {
    pub status: String,
    pub rows: usize,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub excluded: BTreeMap<String, usize>,
    pub elapsed_seconds: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BuildParameters {
    pub block_days: usize,
    pub jobs: usize,
    pub threads_per_job: usize,
    pub memory_limit_mb: usize,
    pub warmup_days: usize,
    pub session_bars: usize,
    pub smart_window_days: usize,
    pub baseline_days: usize,
    pub min_smart_prior_observations: usize,
    pub min_baseline_observations: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub factor_set: String,
    pub causality: String,
    pub parameters: BuildParameters,
    pub sources: BTreeMap<String, String>,
    /// Regenerated from code on every run; not read back from the file.
    #[serde(default, skip_deserializing)]
    pub formulas: Vec<crate::schema::FactorFormula>,
    pub dates: BTreeMap<String, DayStatus>,
}

impl Manifest {
    pub fn new(parameters: BuildParameters, sources: BTreeMap<String, String>) -> Self {
        Self::new_named("core24", parameters, sources, formulas())
    }

    pub fn new_named(
        factor_set: &str,
        parameters: BuildParameters,
        sources: BTreeMap<String, String>,
        formulas: Vec<crate::schema::FactorFormula>,
    ) -> Self {
        Self {
            version: MANIFEST_VERSION,
            factor_set: factor_set.to_string(),
            causality: "Every factor uses minute bars of its own trade date plus strictly \
                earlier market-calendar days; rolling baselines exclude the signal day and \
                are aligned to the trading calendar so missing stock-days leave holes \
                instead of compressing the window. Warm-up blocks replay history without \
                writing output, which makes each output date independent of block layout."
                .to_string(),
            parameters,
            sources,
            formulas,
            dates: BTreeMap::new(),
        }
    }

    /// `Some(true)` = complete, `Some(false)` = recorded failure, `None` = unknown.
    pub fn date_complete(&self, trade_date: &str) -> Option<bool> {
        self.dates
            .get(trade_date)
            .map(|status| status.status == "ok")
    }

    pub fn record(&mut self, trade_date: &str, status: DayStatus) {
        self.dates.insert(trade_date.to_string(), status);
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }
}

pub fn default_parameters(
    block_days: usize,
    jobs: usize,
    threads_per_job: usize,
    memory_limit_mb: usize,
) -> BuildParameters {
    BuildParameters {
        block_days,
        jobs,
        threads_per_job,
        memory_limit_mb,
        warmup_days: crate::schema::BASELINE_DAYS,
        session_bars: crate::schema::SESSION_BARS,
        smart_window_days: crate::schema::SMART_WINDOW_DAYS,
        baseline_days: crate::schema::BASELINE_DAYS,
        min_smart_prior_observations: MIN_SMART_PRIOR_OBS,
        min_baseline_observations: MIN_BASELINE_OBS,
    }
}
