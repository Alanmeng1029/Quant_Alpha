use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Render a compact HTML comparison from Rust backtest summaries")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long = "entry")]
    entries: Vec<String>,
    #[arg(long, default_value = "Raw105 分池训练对比")]
    title: String,
    #[arg(
        long,
        default_value = "统一规则：CSI500 Top100 / 每日最多3只，CSI1000 Top100 / 每日最多5只，资金80/20；含费；超额为逐日组合收益减CSI500后复利，Sharpe年化因子243。"
    )]
    note: String,
}

#[derive(Serialize)]
struct Row {
    name: String,
    summary: String,
    total_return: f64,
    excess_return: f64,
    excess_sharpe: f64,
    excess_max_drawdown: f64,
    average_buy_turnover: f64,
    excess_2025: f64,
    excess_2026: f64,
}

fn number(value: &Value, path: &[&str]) -> Result<f64> {
    let mut current = value;
    for key in path {
        current = &current[*key];
    }
    current
        .as_f64()
        .with_context(|| format!("missing numeric JSON path {}", path.join(".")))
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn parse_entry(entry: &str) -> Result<Row> {
    let (name, path) = entry
        .split_once('=')
        .context("entry must be NAME=summary.json")?;
    let path = Path::new(path);
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(Row {
        name: name.to_string(),
        summary: path.display().to_string(),
        total_return: number(&value, &["total_return"])?,
        excess_return: number(&value, &["excess_curve_total_return"])?,
        excess_sharpe: number(&value, &["excess_sharpe_243"])?,
        excess_max_drawdown: number(&value, &["excess_max_drawdown"])?,
        average_buy_turnover: number(&value, &["average_buy_turnover"])?,
        excess_2025: number(&value, &["annual", "2025", "excess_curve_return"])?,
        excess_2026: number(&value, &["annual", "2026", "excess_curve_return"])?,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.entries.len() < 2 {
        bail!("provide at least two --entry values")
    }
    fs::create_dir_all(&args.output)?;
    let rows = args
        .entries
        .iter()
        .map(|entry| parse_entry(entry))
        .collect::<Result<Vec<_>>>()?;
    let table = rows
        .iter()
        .map(|row| {
            format!(
                "<tr><td>{}</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.3}</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.2}%</td><td>{:.2}%</td></tr>",
                escape(&row.name),
                100.0 * row.total_return,
                100.0 * row.excess_return,
                row.excess_sharpe,
                100.0 * row.excess_max_drawdown,
                100.0 * row.average_buy_turnover,
                100.0 * row.excess_2025,
                100.0 * row.excess_2026,
            )
        })
        .collect::<String>();
    let html = format!(
        r#"<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#18202a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border:1px solid #d9e1ea;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f4f6f8}}.note{{color:#5f6b7a;line-height:1.7}}</style><h1>{title}</h1><p class="note">{note}</p><table><tr><th>模型组合</th><th>总收益</th><th>超额复利</th><th>超额Sharpe</th><th>超额最大回撤</th><th>日均买入换手</th><th>2025超额</th><th>2026超额</th></tr>{table}</table>"#,
        title = escape(&args.title),
        note = escape(&args.note),
    );
    fs::write(args.output.join("report.html"), html)?;
    fs::write(
        args.output.join("comparison.json"),
        serde_json::to_string_pretty(&rows)? + "\n",
    )?;
    Ok(())
}
