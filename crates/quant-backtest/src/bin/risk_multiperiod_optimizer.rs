use anyhow::{Context, Result, bail};
use clap::Parser;
use duckdb::{AccessMode, Config as DuckDbConfig, Connection};
use osqp::{CscMatrix, Problem, Settings, Status};
use serde::Serialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const FACTORS: [&str; 6] = [
    "market",
    "size",
    "beta",
    "residual_volatility",
    "momentum",
    "liquidity",
];
const OBJECTIVE_SCALE: f64 = 10_000.0;

#[derive(Parser)]
#[command(about = "Five-period alpha/turnover/factor-risk sparse QP")]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    predictions: PathBuf,
    #[arg(long)]
    risk_root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 2.0)]
    buy_bps: f64,
    #[arg(long, default_value_t = 2.0)]
    sell_bps: f64,
    #[arg(long, default_value_t = 0.01)]
    max_weight: f64,
    #[arg(long, default_value_t = 0.98)]
    invested_weight: f64,
    #[arg(long, default_value_t = 1.0)]
    risk_aversion: f64,
    #[arg(long)]
    start: Option<String>,
    #[arg(long)]
    end: Option<String>,
    #[arg(long)]
    max_days: Option<usize>,
    #[arg(long)]
    initial_weights: Option<PathBuf>,
    #[arg(long, default_value_t = 20_000)]
    solver_max_iterations: u32,
    #[arg(long, default_value_t = 1_000)]
    solver_initial_iterations: u32,
    #[arg(long, default_value_t = 5_000)]
    solver_retry_iterations: u32,
}

#[derive(Clone)]
struct Exposure {
    benchmark_weight: f64,
    values: [f64; 6],
}

#[derive(Serialize)]
struct Summary {
    optimizer: &'static str,
    risk_aversion: f64,
    buy_bps: f64,
    sell_bps: f64,
    max_weight: f64,
    invested_weight: f64,
    days: usize,
    mean_buy_turnover: f64,
    mean_sell_turnover: f64,
    mean_predicted_active_volatility_ann: f64,
    max_predicted_active_volatility_ann: f64,
    mean_holding_count: f64,
    solver_max_iterations: u32,
    solver_initial_iterations: u32,
    solver_retry_iterations: u32,
    solver_check_dualgap: bool,
}

fn solver_settings(max_iterations: u32) -> Settings {
    Settings::default()
        .verbose(false)
        .eps_abs(1e-6)
        .eps_rel(1e-6)
        .max_iter(max_iterations)
        // The turnover epigraph has many equivalent optima. OSQP's duality-gap
        // check keeps iterating after the primal/dual residuals are usable.
        .check_dualgap(false)
        .polishing(true)
}

fn quote_sql(value: &str) -> String {
    value.replace('\'', "''")
}

fn sparse_matrix(
    nrows: usize,
    ncols: usize,
    columns: Vec<Vec<(usize, f64)>>,
) -> CscMatrix<'static> {
    let mut indptr = Vec::with_capacity(ncols + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for mut column in columns {
        column.sort_by_key(|x| x.0);
        for (row, value) in column {
            if value != 0.0 {
                indices.push(row);
                data.push(value);
            }
        }
        indptr.push(indices.len());
    }
    CscMatrix {
        nrows,
        ncols,
        indptr: Cow::Owned(indptr),
        indices: Cow::Owned(indices),
        data: Cow::Owned(data),
    }
}

fn project_capped_simplex(values: &mut [f64], total: f64, cap: f64) {
    let mut lower = values
        .iter()
        .map(|value| value - cap)
        .fold(f64::INFINITY, f64::min);
    let mut upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for _ in 0..80 {
        let shift = 0.5 * (lower + upper);
        let projected_sum: f64 = values
            .iter()
            .map(|value| (value - shift).clamp(0.0, cap))
            .sum();
        if projected_sum > total {
            lower = shift;
        } else {
            upper = shift;
        }
    }
    let shift = 0.5 * (lower + upper);
    for value in values {
        *value = (*value - shift).clamp(0.0, cap);
    }
}

fn clean_capped_simplex(values: &mut [f64], total: f64, cap: f64, dust: f64) {
    project_capped_simplex(values, total, cap);
    let active: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value >= dust).then_some(index))
        .collect();
    assert!(
        active.len() as f64 * cap + 1e-12 >= total,
        "numerical cleanup removed too many portfolio names"
    );
    let mut retained: Vec<f64> = active.iter().map(|index| values[*index]).collect();
    project_capped_simplex(&mut retained, total, cap);
    values.fill(0.0);
    for (index, value) in active.into_iter().zip(retained) {
        values[index] = value;
    }
}

fn copy_tsv_to_parquet(conn: &Connection, source: &Path, output: &Path) -> Result<()> {
    let source = quote_sql(&source.to_string_lossy());
    let output = quote_sql(&output.to_string_lossy());
    conn.execute_batch(&format!(
        "COPY (SELECT * FROM read_csv('{source}', delim='\\t', header=true)) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)"
    ))?;
    Ok(())
}

type Predictions = BTreeMap<String, (String, HashMap<String, (f64, f64)>)>;
type Exposures = HashMap<(String, String), Exposure>;
type SpecificRisk = HashMap<(String, String), f64>;
type Covariances = HashMap<String, [[f64; 6]; 6]>;

fn load_predictions(conn: &Connection, args: &Args) -> Result<Predictions> {
    let path = quote_sql(&args.predictions.canonicalize()?.to_string_lossy());
    let start = args.start.as_deref().unwrap_or("1900-01-01");
    let end = args.end.as_deref().unwrap_or("2999-12-31");
    let query = format!(
        "SELECT trade_date::VARCHAR, execution_date::VARCHAR, ts_code, raw_h1::DOUBLE, raw_h5::DOUBLE FROM read_parquet('{path}') WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}' AND execution_date IS NOT NULL AND isfinite(raw_h1) AND isfinite(raw_h5) ORDER BY trade_date, ts_code"
    );
    let mut output: Predictions = BTreeMap::new();
    for row in conn.prepare(&query)?.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, f64>(3)?,
            row.get::<_, f64>(4)?,
        ))
    })? {
        let (date, execution, code, h1, h5) = row?;
        output
            .entry(date)
            .or_insert_with(|| (execution, HashMap::new()))
            .1
            .insert(code, (h1, h5));
    }
    Ok(output)
}

fn load_exposures(conn: &Connection, risk_root: &Path) -> Result<Exposures> {
    let path = quote_sql(
        &risk_root
            .join("optimizer_exposures.parquet")
            .canonicalize()?
            .to_string_lossy(),
    );
    let query = format!(
        "SELECT date::DATE::VARCHAR, ts_code, benchmark_weight::DOUBLE, market::DOUBLE, size::DOUBLE, beta::DOUBLE, residual_volatility::DOUBLE, momentum::DOUBLE, liquidity::DOUBLE FROM read_parquet('{path}')"
    );
    let mut output = HashMap::new();
    for row in conn.prepare(&query)?.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            Exposure {
                benchmark_weight: row.get(2)?,
                values: [
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                ],
            },
        ))
    })? {
        let (date, code, exposure) = row?;
        output.insert((date, code), exposure);
    }
    Ok(output)
}

fn load_specific_risk(conn: &Connection, risk_root: &Path) -> Result<SpecificRisk> {
    let path = quote_sql(
        &risk_root
            .join("specific_variance_daily.parquet")
            .canonicalize()?
            .to_string_lossy(),
    );
    let query = format!(
        "SELECT date::DATE::VARCHAR, ts_code, specific_variance::DOUBLE FROM read_parquet('{path}')"
    );
    let mut output = HashMap::new();
    for row in conn.prepare(&query)?.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })? {
        let (date, code, variance) = row?;
        output.insert((date, code), variance);
    }
    Ok(output)
}

fn load_covariances(conn: &Connection, risk_root: &Path) -> Result<Covariances> {
    let path = quote_sql(
        &risk_root
            .join("factor_covariance_daily.parquet")
            .canonicalize()?
            .to_string_lossy(),
    );
    let query = format!(
        "SELECT date::DATE::VARCHAR, factor_left, factor_right, covariance::DOUBLE FROM read_parquet('{path}')"
    );
    let factor_index: HashMap<&str, usize> = FACTORS
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index))
        .collect();
    let mut output: Covariances = HashMap::new();
    for row in conn.prepare(&query)?.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, f64>(3)?,
        ))
    })? {
        let (date, left, right, value) = row?;
        let i = *factor_index
            .get(left.as_str())
            .with_context(|| format!("unknown factor {left}"))?;
        let j = *factor_index
            .get(right.as_str())
            .with_context(|| format!("unknown factor {right}"))?;
        output.entry(date).or_insert([[0.0; 6]; 6])[i][j] = value;
    }
    Ok(output)
}

fn load_initial_weights(conn: &Connection, path: &Path) -> Result<HashMap<String, f64>> {
    let path = quote_sql(&path.canonicalize()?.to_string_lossy());
    let query = format!(
        "SELECT ts_code, target_weight::DOUBLE FROM read_parquet('{path}') WHERE trade_date = (SELECT max(trade_date) FROM read_parquet('{path}'))"
    );
    let mut output = HashMap::new();
    for row in conn.prepare(&query)?.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })? {
        let (code, weight) = row?;
        output.insert(code, weight);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn solve_day(
    codes: &[String],
    alpha: &[Vec<f64>],
    previous: &HashMap<String, f64>,
    eligible: &[bool],
    exposures: &[[f64; 6]],
    benchmark: &[f64],
    specific: &[f64],
    factor_covariance: &[[f64; 6]; 6],
    args: &Args,
) -> Result<(
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    usize,
    String,
    f64,
    f64,
)> {
    let n = codes.len();
    let horizon = alpha.len();
    let factors = FACTORS.len();
    let block = horizon * n;
    let factor_block = horizon * factors;
    let vars = 3 * block + factor_block;
    let r_budget = 0;
    let r_flow = r_budget + horizon;
    let r_factor = r_flow + block;
    let r_identity = r_factor + factor_block;
    let constraints = r_identity + vars;
    let w_index = |period: usize, name: usize| period * n + name;
    let buy_index = |period: usize, name: usize| block + period * n + name;
    let sell_index = |period: usize, name: usize| 2 * block + period * n + name;
    let z_index = |period: usize, factor: usize| 3 * block + period * factors + factor;

    let mut a_columns: Vec<Vec<(usize, f64)>> = vec![Vec::new(); vars];
    for period in 0..horizon {
        for name in 0..n {
            let w = w_index(period, name);
            a_columns[w].push((r_budget + period, 1.0));
            a_columns[w].push((r_flow + period * n + name, 1.0));
            if period + 1 < horizon {
                a_columns[w].push((r_flow + (period + 1) * n + name, -1.0));
            }
            for factor in 0..factors {
                a_columns[w].push((
                    r_factor + period * factors + factor,
                    -exposures[name][factor],
                ));
            }
            a_columns[w].push((r_identity + w, 1.0));
            let buy = buy_index(period, name);
            a_columns[buy].push((r_flow + period * n + name, -1.0));
            a_columns[buy].push((r_identity + buy, 1.0));
            let sell = sell_index(period, name);
            a_columns[sell].push((r_flow + period * n + name, 1.0));
            a_columns[sell].push((r_identity + sell, 1.0));
        }
        for factor in 0..factors {
            let z = z_index(period, factor);
            a_columns[z].push((r_factor + period * factors + factor, 1.0));
            a_columns[z].push((r_identity + z, 1.0));
        }
    }
    let a = sparse_matrix(constraints, vars, a_columns);

    let mut p_columns: Vec<Vec<(usize, f64)>> = vec![Vec::new(); vars];
    let curvature = 2.0 * args.risk_aversion * OBJECTIVE_SCALE;
    for period in 0..horizon {
        for name in 0..n {
            let w = w_index(period, name);
            p_columns[w].push((w, curvature * specific[name] + 1e-10));
            let buy = buy_index(period, name);
            let sell = sell_index(period, name);
            p_columns[buy].push((buy, 1e-8 * OBJECTIVE_SCALE));
            p_columns[sell].push((sell, 1e-8 * OBJECTIVE_SCALE));
        }
        for right in 0..factors {
            let column = z_index(period, right);
            for left in 0..=right {
                p_columns[column].push((
                    z_index(period, left),
                    curvature * factor_covariance[left][right],
                ));
            }
        }
    }
    let p = sparse_matrix(vars, vars, p_columns);

    let mut benchmark_factor = [0.0; 6];
    for name in 0..n {
        for factor in 0..factors {
            benchmark_factor[factor] += benchmark[name] * exposures[name][factor];
        }
    }
    let mut covariance_benchmark_factor = [0.0; 6];
    for left in 0..factors {
        for right in 0..factors {
            covariance_benchmark_factor[left] +=
                factor_covariance[left][right] * benchmark_factor[right];
        }
    }
    let mut q = vec![0.0; vars];
    for period in 0..horizon {
        for name in 0..n {
            q[w_index(period, name)] = OBJECTIVE_SCALE
                * (-alpha[period][name]
                    - 2.0 * args.risk_aversion * specific[name] * benchmark[name]);
            q[buy_index(period, name)] = OBJECTIVE_SCALE * args.buy_bps / 10_000.0;
            q[sell_index(period, name)] = OBJECTIVE_SCALE * args.sell_bps / 10_000.0;
        }
        for factor in 0..factors {
            q[z_index(period, factor)] =
                -2.0 * OBJECTIVE_SCALE * args.risk_aversion * covariance_benchmark_factor[factor];
        }
    }

    let mut lower = vec![f64::NEG_INFINITY; constraints];
    let mut upper = vec![f64::INFINITY; constraints];
    for period in 0..horizon {
        lower[r_budget + period] = args.invested_weight;
        upper[r_budget + period] = args.invested_weight;
        for name in 0..n {
            let row = r_flow + period * n + name;
            let rhs = if period == 0 {
                previous.get(&codes[name]).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            lower[row] = rhs;
            upper[row] = rhs;
        }
        for factor in 0..factors {
            let row = r_factor + period * factors + factor;
            lower[row] = 0.0;
            upper[row] = 0.0;
        }
    }
    for period in 0..horizon {
        for name in 0..n {
            let w = w_index(period, name);
            lower[r_identity + w] = 0.0;
            upper[r_identity + w] = if eligible[name] { args.max_weight } else { 0.0 };
            lower[r_identity + buy_index(period, name)] = 0.0;
            lower[r_identity + sell_index(period, name)] = 0.0;
        }
    }

    let checkpoints = [
        args.solver_initial_iterations,
        args.solver_retry_iterations,
        args.solver_max_iterations,
    ];
    let mut checkpoint_index = 0usize;
    let mut total_iterations = 0usize;
    let (solution, status, primal_residual, dual_residual) = loop {
        let settings = solver_settings(checkpoints[checkpoint_index]);
        let mut problem = Problem::new(p.clone(), &q, a.clone(), &lower, &upper, &settings)
            .context("set up risk-aware OSQP")?;
        let result = problem.solve();
        total_iterations += result.iter() as usize;
        let (primal_limit, dual_limit) = if checkpoint_index + 1 < checkpoints.len() {
            (1e-8, 1e-6)
        } else {
            (1e-5, 1e-2)
        };
        let accepted = match &result {
            Status::Solved(solution) => Some((
                solution.x().to_vec(),
                "solved",
                solution.pri_res(),
                solution.dua_res(),
            )),
            Status::SolvedInaccurate(solution) => Some((
                solution.x().to_vec(),
                "solved_inaccurate",
                solution.pri_res(),
                solution.dua_res(),
            )),
            Status::MaxIterationsReached(solution)
                if solution.pri_res() <= primal_limit && solution.dua_res() <= dual_limit =>
            {
                Some((
                    solution.x().to_vec(),
                    "max_iterations_near_optimal",
                    solution.pri_res(),
                    solution.dua_res(),
                ))
            }
            _ => None,
        };
        if let Some(value) = accepted {
            break value;
        }
        let residuals = match &result {
            Status::MaxIterationsReached(solution) => (solution.pri_res(), solution.dua_res()),
            _ => bail!("risk-aware optimizer failed without a usable primal solution"),
        };
        checkpoint_index += 1;
        if checkpoint_index >= checkpoints.len() {
            bail!(
                "risk-aware optimizer reached iteration limit: primal_residual={}, dual_residual={}",
                residuals.0,
                residuals.1
            );
        }
    };
    let mut weights = vec![vec![0.0; n]; horizon];
    let mut buys = vec![vec![0.0; n]; horizon];
    let mut sells = vec![vec![0.0; n]; horizon];
    for period in 0..horizon {
        for name in 0..n {
            weights[period][name] = solution[w_index(period, name)];
            buys[period][name] = solution[buy_index(period, name)];
            sells[period][name] = solution[sell_index(period, name)];
        }
    }
    Ok((
        weights,
        buys,
        sells,
        total_iterations,
        status.to_string(),
        primal_residual,
        dual_residual,
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(args.risk_aversion >= 0.0
        && args.max_weight > 0.0
        && args.invested_weight > 0.0
        && args.invested_weight <= 1.0
        && args.solver_initial_iterations > 0
        && args.solver_initial_iterations <= args.solver_retry_iterations
        && args.solver_retry_iterations <= args.solver_max_iterations)
    {
        bail!("invalid optimizer parameters");
    }
    fs::create_dir_all(&args.output)?;
    let conn = Connection::open_with_flags(
        &args.catalog,
        DuckDbConfig::default().access_mode(AccessMode::ReadOnly)?,
    )?;
    let predictions = load_predictions(&conn, &args)?;
    let exposures = load_exposures(&conn, &args.risk_root)?;
    let specific_risk = load_specific_risk(&conn, &args.risk_root)?;
    let covariances = load_covariances(&conn, &args.risk_root)?;

    let target_tsv = args.output.join(".target_weights.tsv");
    let daily_tsv = args.output.join(".optimizer_daily.tsv");
    let mut target_writer = BufWriter::new(File::create(&target_tsv)?);
    let mut daily_writer = BufWriter::new(File::create(&daily_tsv)?);
    writeln!(
        target_writer,
        "trade_date\texecution_date\tts_code\ttarget_weight"
    )?;
    writeln!(
        daily_writer,
        "trade_date\texecution_date\tsolver_status\titerations\tprimal_residual\tdual_residual_scaled\tfirst_buy_turnover\tfirst_sell_turnover\tholding_count\tpredicted_active_variance\tpredicted_active_volatility_ann"
    )?;

    let total_days = predictions.len().min(args.max_days.unwrap_or(usize::MAX));
    let started = Instant::now();
    let mut previous = match &args.initial_weights {
        Some(path) => load_initial_weights(&conn, path)?,
        None => HashMap::new(),
    };
    let mut day_count = 0usize;
    let mut buy_sum = 0.0;
    let mut sell_sum = 0.0;
    let mut holding_sum = 0.0;
    let mut volatility_sum = 0.0;
    let mut max_volatility: f64 = 0.0;
    for (date, (execution_date, signals)) in predictions {
        if args.max_days.is_some_and(|limit| day_count >= limit) {
            break;
        }
        let covariance = match covariances.get(&date) {
            Some(value) => value,
            None => continue,
        };
        let mut current: Vec<String> = signals
            .keys()
            .filter(|code| {
                exposures.contains_key(&(date.clone(), (*code).clone()))
                    && specific_risk.contains_key(&(date.clone(), (*code).clone()))
            })
            .cloned()
            .collect();
        current.sort();
        if current.len() as f64 * args.max_weight + 1e-12 < args.invested_weight {
            bail!("{date}: only {} risk-covered eligible names", current.len());
        }
        let mut codes = current.clone();
        for code in previous.keys() {
            if !signals.contains_key(code) && !codes.contains(code) {
                codes.push(code.clone());
            }
        }
        codes.sort();
        let current_set: std::collections::HashSet<&str> =
            current.iter().map(String::as_str).collect();
        let eligible: Vec<bool> = codes
            .iter()
            .map(|code| current_set.contains(code.as_str()))
            .collect();
        let mut x = vec![[0.0; 6]; codes.len()];
        let mut specific = vec![0.0; codes.len()];
        let mut benchmark = vec![0.0; codes.len()];
        let benchmark_sum: f64 = current
            .iter()
            .map(|code| exposures[&(date.clone(), code.clone())].benchmark_weight)
            .sum();
        for (index, code) in codes.iter().enumerate() {
            if eligible[index] {
                let exposure = &exposures[&(date.clone(), code.clone())];
                x[index] = exposure.values;
                specific[index] = specific_risk[&(date.clone(), code.clone())];
                benchmark[index] = args.invested_weight * exposure.benchmark_weight / benchmark_sum;
            }
        }
        let h1: Vec<f64> = codes
            .iter()
            .map(|code| signals.get(code).map(|value| value.0).unwrap_or(0.0))
            .collect();
        let h5: Vec<f64> = codes
            .iter()
            .map(|code| signals.get(code).map(|value| value.1).unwrap_or(0.0))
            .collect();
        let mut alpha = vec![h1.clone()];
        for _ in 1..5 {
            alpha.push(
                h5.iter()
                    .zip(&h1)
                    .map(|(five, one)| (five - one) / 4.0)
                    .collect(),
            );
        }
        let (mut weights, _buys, _sells, iterations, status, primal_residual, dual_residual) =
            solve_day(
                &codes, &alpha, &previous, &eligible, &x, &benchmark, &specific, covariance, &args,
            )?;
        clean_capped_simplex(&mut weights[0], args.invested_weight, args.max_weight, 1e-6);
        let first = &weights[0];
        let buy_turnover: f64 = codes
            .iter()
            .zip(first)
            .map(|(code, weight)| (weight - previous.get(code).copied().unwrap_or(0.0)).max(0.0))
            .sum();
        let sell_turnover: f64 = codes
            .iter()
            .zip(first)
            .map(|(code, weight)| (previous.get(code).copied().unwrap_or(0.0) - weight).max(0.0))
            .sum();
        previous = codes
            .iter()
            .zip(first)
            .filter(|(_, weight)| **weight > 1e-10)
            .map(|(code, weight)| (code.clone(), *weight))
            .collect();
        for (code, weight) in &previous {
            writeln!(
                target_writer,
                "{date}\t{execution_date}\t{code}\t{weight:.12}"
            )?;
        }
        let active: Vec<f64> = first
            .iter()
            .zip(&benchmark)
            .map(|(weight, base)| weight - base)
            .collect();
        let mut factor_exposure = [0.0; 6];
        for name in 0..codes.len() {
            for factor in 0..6 {
                factor_exposure[factor] += active[name] * x[name][factor];
            }
        }
        let mut factor_variance = 0.0;
        for left in 0..6 {
            for right in 0..6 {
                factor_variance +=
                    factor_exposure[left] * covariance[left][right] * factor_exposure[right];
            }
        }
        let specific_variance: f64 = active
            .iter()
            .zip(&specific)
            .map(|(weight, variance)| weight * weight * variance)
            .sum();
        let active_variance = (factor_variance + specific_variance).max(0.0);
        let active_volatility = (active_variance * 252.0).sqrt();
        writeln!(
            daily_writer,
            "{date}\t{execution_date}\t{status}\t{iterations}\t{primal_residual:.12}\t{dual_residual:.12}\t{buy_turnover:.12}\t{sell_turnover:.12}\t{}\t{active_variance:.12}\t{active_volatility:.12}",
            previous.len()
        )?;
        day_count += 1;
        buy_sum += buy_turnover;
        sell_sum += sell_turnover;
        holding_sum += previous.len() as f64;
        volatility_sum += active_volatility;
        max_volatility = max_volatility.max(active_volatility);
        if day_count % 25 == 0 || day_count == total_days {
            target_writer.flush()?;
            daily_writer.flush()?;
            let elapsed = started.elapsed().as_secs_f64();
            let rate = day_count as f64 / elapsed.max(f64::EPSILON);
            let eta = (total_days.saturating_sub(day_count)) as f64 / rate.max(f64::EPSILON);
            eprintln!(
                "progress {day_count}/{total_days} days ({rate:.3} days/s, ETA {eta:.0}s), latest={date}"
            );
        }
    }
    target_writer.flush()?;
    daily_writer.flush()?;
    if day_count == 0 {
        bail!("no prediction dates had matching risk snapshots");
    }
    copy_tsv_to_parquet(
        &conn,
        &target_tsv,
        &args.output.join("target_weights.parquet"),
    )?;
    copy_tsv_to_parquet(
        &conn,
        &daily_tsv,
        &args.output.join("optimizer_daily.parquet"),
    )?;
    fs::remove_file(target_tsv)?;
    fs::remove_file(daily_tsv)?;
    let summary = Summary {
        optimizer: "five-period alpha minus turnover cost minus factor active risk",
        risk_aversion: args.risk_aversion,
        buy_bps: args.buy_bps,
        sell_bps: args.sell_bps,
        max_weight: args.max_weight,
        invested_weight: args.invested_weight,
        days: day_count,
        mean_buy_turnover: buy_sum / day_count as f64,
        mean_sell_turnover: sell_sum / day_count as f64,
        mean_predicted_active_volatility_ann: volatility_sum / day_count as f64,
        max_predicted_active_volatility_ann: max_volatility,
        mean_holding_count: holding_sum / day_count as f64,
        solver_max_iterations: args.solver_max_iterations,
        solver_initial_iterations: args.solver_initial_iterations,
        solver_retry_iterations: args.solver_retry_iterations,
        solver_check_dualgap: false,
    };
    fs::write(
        args.output.join("optimizer_summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_simplex_cleanup_enforces_budget_cap_and_dust() {
        let mut weights = vec![0.500001, 0.479999, 1e-9];
        clean_capped_simplex(&mut weights, 0.98, 0.50, 1e-6);
        assert!((weights.iter().sum::<f64>() - 0.98).abs() < 1e-12);
        assert!(
            weights
                .iter()
                .all(|weight| *weight >= 0.0 && *weight <= 0.50)
        );
        assert_eq!(weights[2], 0.0);
    }
}
