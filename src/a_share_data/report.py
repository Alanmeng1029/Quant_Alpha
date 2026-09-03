"""Render the standard factor tear sheet from Rust result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from jinja2 import Template
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas


HTML_TEMPLATE = """<!doctype html><html><head><meta charset='utf-8'><title>{{ factor }}</title>
<style>body{font-family:Arial,sans-serif;margin:32px;color:#18212f}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid #d9e1ea;padding:12px;border-radius:6px}img{width:100%;margin:16px 0}.muted{color:#667085}</style>
</head><body><h1>{{ factor }} - 单因子预测报告</h1><p class='muted'>选股范围：{{ universe }}；所有超额收益相对中证500。{{ start }} 至 {{ end }}。</p>
<div class='grid'>{% for metric in metrics %}<div class='card'><div class='muted'>{{ metric.label }}</div><strong>{{ metric.value }}</strong></div>{% endfor %}</div>
<h2>Rank IC summary</h2><table border='1' cellspacing='0' cellpadding='6'><thead><tr><th>Label</th><th>Horizon</th><th>Mean Rank IC</th><th>Rank ICIR</th><th>Positive ratio</th></tr></thead><tbody>{% for row in ic_table %}<tr><td>{{ row.label }}</td><td>{{ row.horizon }}</td><td>{{ row.rank_ic }}</td><td>{{ row.rank_icir }}</td><td>{{ row.positive_ratio }}</td></tr>{% endfor %}</tbody></table>
{% for image in images %}<img src='{{ image }}'>{% endfor %}</body></html>"""


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _has_column(frame: pl.DataFrame, column: str) -> bool:
    return column in frame.columns


def _kind(frame: pl.DataFrame, preferred: str) -> str:
    """Use legacy raw artifacts only when rendering historical result folders."""
    if not _has_column(frame, "return_kind"):
        return preferred
    values = set(frame.get_column("return_kind").unique().to_list())
    return preferred if preferred in values else "raw"


def _research_kind(frame: pl.DataFrame) -> str:
    """Resolve the primary return label, preferring the Open-to-open protocol."""
    if _has_column(frame, "return_kind") and "open_to_open_raw" in set(frame.get_column("return_kind").unique().to_list()):
        return "open_to_open_raw"
    return _kind(frame, "close_to_close_raw")


def _research_label(frame: pl.DataFrame) -> str:
    return "Open-to-open" if _research_kind(frame) == "open_to_open_raw" else "Close-to-close"


def _series(frame: pl.DataFrame, expression: pl.Expr) -> tuple[list[object], list[float]]:
    values = frame.select(expression).to_series().to_list()
    return frame["trade_date"].to_list(), [float(value or 0.0) for value in values]


def _group_panel(groups: pl.DataFrame, kind: str) -> pl.DataFrame:
    return groups.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == kind)).group_by("group_number").agg(pl.col("mean_return").mean()).sort("group_number")


def _top_down(groups: pl.DataFrame, kind: str, high_minus_low: bool) -> pl.DataFrame:
    daily = groups.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == kind) & pl.col("group_number").is_in([1, 10])).pivot(on="group_number", index="trade_date", values="mean_return", aggregate_function="first").sort("trade_date")
    spread = pl.col("10") - pl.col("1") if high_minus_low else pl.col("1") - pl.col("10")
    return daily.with_columns(spread.alias("top_down_return")) if not daily.is_empty() else daily


def _rolling_icir(values: list[float], window: int = 252) -> list[float]:
    data = np.asarray(values, dtype=float)
    result = np.full(len(data), np.nan)
    for index in range(window - 1, len(data)):
        sample = data[index - window + 1:index + 1]
        std = sample.std(ddof=1)
        if std > 0: result[index] = sample.mean() / std * np.sqrt(252)
    return result.tolist()


def _save_page(pdf: canvas.Canvas, image: Path, factor: str) -> None:
    width, height = landscape(letter)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(36, height - 28, f"{factor} - factor evaluation")
    pdf.drawImage(str(image), 30, 30, width=width - 60, height=height - 70, preserveAspectRatio=True, anchor="c")
    pdf.showPage()


def _plot_report(output: Path, summary: dict[str, object], ic: pl.DataFrame, groups: pl.DataFrame, portfolio: pl.DataFrame, diagnostics: pl.DataFrame) -> list[Path]:
    images: list[Path] = []
    if not diagnostics.is_empty(): diagnostics = diagnostics.sort("trade_date")
    raw_kind = _research_kind(ic)
    research_label = _research_label(ic)
    raw_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == raw_kind)).sort("trade_date") if not ic.is_empty() else ic
    vwap_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == _kind(ic, "vwap_to_vwap_raw"))).sort("trade_date") if not ic.is_empty() else ic
    twap_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == _kind(ic, "twap_to_twap_raw"))).sort("trade_date") if not ic.is_empty() else ic
    raw_group_kind = _research_kind(groups)
    excess_group_kind = "open_to_open_excess_csi500" if raw_group_kind == "open_to_open_raw" else _kind(groups, "close_to_close_excess_csi500")
    raw_groups, excess_groups = _group_panel(groups, raw_group_kind), _group_panel(groups, excess_group_kind)
    raw_top_down, excess_top_down = _top_down(groups, raw_group_kind, True), _top_down(groups, excess_group_kind, True)
    figure, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    figure.suptitle(f"{summary['factor_id']} - Standard factor tear sheet", fontsize=16, fontweight="bold")
    if not raw_ic.is_empty():
        x, rank_ic = _series(raw_ic, pl.col("rank_ic").fill_null(0).cum_sum()); axes[0, 0].plot(x, rank_ic, label="Rank IC")
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title(f"Research: cumulative {research_label} Rank IC")
    if not raw_ic.is_empty():
        dates, values = _series(raw_ic, pl.col("rank_ic").fill_null(0)); axes[0, 1].plot(dates, _rolling_icir(values), linewidth=1)
    axes[0, 1].set_title("Research: 252-day rolling Rank ICIR")
    for data, label in ((vwap_ic, "VWAP-to-VWAP"), (twap_ic, "TWAP-to-TWAP")):
        if not data.is_empty(): axes[0, 2].plot(data["trade_date"].to_list(), data["rank_ic"].fill_null(0).cum_sum().to_list(), label=label)
    axes[0, 2].legend(fontsize=8); axes[0, 2].set_title("Executable: cumulative Rank IC")
    if not diagnostics.is_empty(): axes[1, 0].plot(diagnostics["trade_date"].to_list(), diagnostics["coverage_ratio"].fill_null(0).to_list(), linewidth=.8)
    axes[1, 0].set_ylim(0, 1.05); axes[1, 0].set_title("Factor coverage ratio")
    if not raw_groups.is_empty(): axes[1, 1].bar(raw_groups["group_number"].cast(pl.Utf8).to_list(), raw_groups["mean_return"].to_list())
    axes[1, 1].set_title(f"Mean {research_label} return by decile")
    if not excess_groups.is_empty(): axes[1, 2].bar(excess_groups["group_number"].cast(pl.Utf8).to_list(), excess_groups["mean_return"].to_list(), color="#e07a5f")
    axes[1, 2].set_title("Mean excess return vs CSI500 by decile")
    for group in range(1, 11):
        data = groups.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == raw_group_kind) & (pl.col("group_number") == group)).sort("trade_date")
        if not data.is_empty(): axes[2, 0].plot(data["trade_date"].to_list(), (1 + data["mean_return"].fill_null(0)).cum_prod().to_list(), label=f"G{group}")
    axes[2, 0].legend(fontsize=7, ncol=2); axes[2, 0].set_title(f"{research_label} cumulative returns: G1 through G10")
    for group in range(1, 11):
        data = groups.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == excess_group_kind) & (pl.col("group_number") == group)).sort("trade_date")
        if not data.is_empty(): axes[2, 1].plot(data["trade_date"].to_list(), (1 + data["mean_return"].fill_null(0)).cum_prod().to_list(), label=f"G{group}")
    axes[2, 1].legend(fontsize=7, ncol=2); axes[2, 1].set_title("Excess cumulative returns vs CSI500: G1 through G10")
    if not diagnostics.is_empty():
        axes[2, 2].plot(diagnostics["trade_date"].to_list(), diagnostics["factor_mean"].fill_null(0).to_list(), label="mean")
        axes[2, 2].plot(diagnostics["trade_date"].to_list(), diagnostics["factor_std"].fill_null(0).to_list(), label="std")
        axes[2, 2].legend(fontsize=8)
    axes[2, 2].set_title("1%-99% winsorized daily factor mean and std")
    for axis in axes.flat: axis.grid(alpha=.2); axis.tick_params(axis="x", rotation=25, labelsize=8)
    first = output / "tear_sheet.png"; figure.savefig(first, dpi=180); images.append(first); plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(16, 7), constrained_layout=True)
    for data, label in ((raw_top_down, "absolute"), (excess_top_down, "excess vs CSI500")):
        if not data.is_empty(): axes[0].plot(data["trade_date"].to_list(), (1 + data["top_down_return"].fill_null(0)).cum_prod().to_list(), label=label)
    axes[0].legend(); axes[0].set_title("Research directional spread: G10 − G1 (no costs)")
    for data, label in ((vwap_ic, "VWAP-to-VWAP"), (twap_ic, "TWAP-to-TWAP")):
        if not data.is_empty(): axes[1].plot(data["trade_date"].to_list(), data["rank_ic"].fill_null(0).to_list(), label=label)
    axes[1].legend(); axes[1].set_title("Executable daily Rank IC")
    table_rows = []
    for label, kind in ((research_label, raw_kind), ("VWAP", "vwap_to_vwap_raw"), ("TWAP", "twap_to_twap_raw")):
        for horizon in (1, 5, 10, 20):
            data = ic.filter((pl.col("return_kind") == _kind(ic, kind)) & (pl.col("horizon") == horizon))["rank_ic"].drop_nulls()
            if data.len() > 0:
                mean, std = data.mean(), data.std()
                table_rows.append([label, str(horizon), f"{mean:.4f}", f"{mean / std * np.sqrt(252):.2f}" if std and std != 0 else "n/a"])
    axes[2].axis("off")
    axes[2].table(cellText=table_rows, colLabels=["Label", "Days", "Rank IC", "ICIR"], loc="center", cellLoc="right")
    axes[2].set_title("Rank IC summary")
    for axis in axes: axis.grid(alpha=.2); axis.tick_params(axis="x", rotation=25)
    second = output / "portfolio_summary.png"; figure.savefig(second, dpi=180); images.append(second); plt.close(figure)
    document = canvas.Canvas(str(output / "report.pdf"), pagesize=landscape(letter))
    for image in images: _save_page(document, image, str(summary["factor_id"]))
    document.save()
    return images


def render(input_dir: Path) -> None:
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    ic, groups, portfolio, diagnostics = (_read(input_dir / name) for name in ("daily_ic.parquet", "group_returns.parquet", "portfolio_daily.parquet", "factor_diagnostics.parquet"))
    images = _plot_report(input_dir, summary, ic, groups, portfolio, diagnostics)
    raw_kind = _research_kind(ic)
    research_label = _research_label(ic)
    raw_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == raw_kind)) if not ic.is_empty() else ic
    vwap_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == _kind(ic, "vwap_to_vwap_raw"))) if not ic.is_empty() else ic
    twap_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == _kind(ic, "twap_to_twap_raw"))) if not ic.is_empty() else ic
    metrics = [
        {"label": "有效证券日", "value": f"{summary['rows']:,}"},
        {"label": f"{research_label} 1日 Rank IC", "value": f"{raw_ic['rank_ic'].mean():.4f}" if not raw_ic.is_empty() else "n/a"},
        {"label": "可执行 VWAP 1日 Rank IC", "value": f"{vwap_ic['rank_ic'].mean():.4f}" if not vwap_ic.is_empty() else "n/a"},
        {"label": "可执行 TWAP 1日 Rank IC", "value": f"{twap_ic['rank_ic'].mean():.4f}" if not twap_ic.is_empty() else "n/a"},
    ]
    ic_table = []
    for label, kind in ((f"Research {research_label}", raw_kind), ("Executable VWAP-to-VWAP", "vwap_to_vwap_raw"), ("Executable TWAP-to-TWAP", "twap_to_twap_raw")):
        for horizon in (1, 5, 10, 20):
            values = ic.filter((pl.col("return_kind") == _kind(ic, kind)) & (pl.col("horizon") == horizon))["rank_ic"].drop_nulls() if not ic.is_empty() else pl.Series()
            if values.len() == 0:
                continue
            mean, std = values.mean(), values.std()
            ic_table.append({"label": label, "horizon": horizon, "rank_ic": f"{mean:.4f}", "rank_icir": f"{mean / std * np.sqrt(252):.3f}" if std and std != 0 else "n/a", "positive_ratio": f"{(values > 0).mean():.1%}"})
    html = Template(HTML_TEMPLATE).render(factor=summary["factor_id"], universe=summary["universe"], start=summary["start"], end=summary["end"], cost=summary["transaction_cost_bps"], metrics=metrics, ic_table=ic_table, images=[image.name for image in images])
    (input_dir / "report.html").write_text(html, encoding="utf-8")
    print(json.dumps({"html": str(input_dir / "report.html"), "pdf": str(input_dir / "report.pdf")}, ensure_ascii=False))


def _batch_row(summary_path: Path) -> dict[str, object]:
    root = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ic = _read(root / "daily_ic.parquet")
    raw_ic = ic.filter((pl.col("horizon") == 1) & (pl.col("return_kind") == _kind(ic, "close_to_close_raw"))) if not ic.is_empty() else ic
    rank_mean = raw_ic["rank_ic"].mean() if not raw_ic.is_empty() else None
    rank_std = raw_ic["rank_ic"].std() if not raw_ic.is_empty() else None
    return {
        "factor_id": summary["factor_id"], "universe": summary["universe"], "run_path": str(root),
        "report_path": str(root / "report.html"), "close_h1_rank_ic": rank_mean,
        "close_h1_rank_icir": rank_mean / rank_std * np.sqrt(252) if rank_std and rank_std != 0 else None,
        "close_h1_positive_ratio": (raw_ic["rank_ic"] > 0).mean() if not raw_ic.is_empty() else None,
    }


def _batch_metrics(summary_path: Path) -> list[dict[str, object]]:
    root = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ic = _read(root / "daily_ic.parquet")
    if ic.is_empty():
        return []
    rows: list[dict[str, object]] = []
    for label, kind in (("Close", "close_to_close_raw"), ("VWAP", "vwap_to_vwap_raw"), ("TWAP", "twap_to_twap_raw")):
        resolved = _kind(ic, kind)
        for horizon in (1, 5, 10, 20):
            values = ic.filter((pl.col("return_kind") == resolved) & (pl.col("horizon") == horizon))["rank_ic"].drop_nulls()
            if values.len() == 0:
                continue
            mean, std = values.mean(), values.std()
            rows.append({
                "factor_id": summary["factor_id"], "universe": summary["universe"], "run_path": str(root),
                "label": label, "horizon": horizon, "mean_rank_ic": mean,
                "rank_icir": mean / std * np.sqrt(252) if std and std != 0 else None,
                "positive_rank_ic_ratio": (values > 0).mean(), "days": values.len(),
            })
    return rows


def _render_one(path: str) -> str:
    render(Path(path))
    return path


def render_batch(input_dir: Path, render_reports: bool, jobs: int) -> None:
    status_json = input_dir / "task_status.json"
    if status_json.exists():
        pl.DataFrame(json.loads(status_json.read_text(encoding="utf-8"))).write_parquet(input_dir / "task_status.parquet")
    summaries = sorted(input_dir.rglob("summary.json"))
    rows = [_batch_row(path) for path in summaries]
    frame = pl.DataFrame(rows).sort(["universe", "close_h1_rank_icir"], descending=[False, True]) if rows else pl.DataFrame()
    metric_rows = [row for path in summaries for row in _batch_metrics(path)]
    metrics = pl.DataFrame(metric_rows).sort(["universe", "label", "horizon", "rank_icir"], descending=[False, False, False, True]) if metric_rows else pl.DataFrame()
    frame.write_parquet(input_dir / "batch_summary.parquet")
    frame.write_csv(input_dir / "batch_summary.csv")
    metrics.write_parquet(input_dir / "batch_metrics.parquet")
    metrics.write_csv(input_dir / "batch_metrics.csv")
    columns = frame.columns
    body = "".join("<tr>" + "".join(f"<td>{row.get(column, '')}</td>" for column in columns) + "</tr>" for row in frame.to_dicts())
    status = json.loads(status_json.read_text(encoding="utf-8")) if status_json.exists() else []
    metric_body = "".join("<tr>" + "".join(f"<td>{row.get(column, '')}</td>" for column in metrics.columns) + "</tr>" for row in metrics.to_dicts())
    (input_dir / "batch_report.html").write_text(
        f"<!doctype html><meta charset='utf-8'><h1>Factor batch summary</h1>"
        f"<h2>Close 1-day ranking</h2><table border='1'><thead><tr>{''.join(f'<th>{column}</th>' for column in columns)}</tr></thead><tbody>{body}</tbody></table>"
        f"<h2>Rank IC and ICIR by label and horizon</h2><table border='1'><thead><tr>{''.join(f'<th>{column}</th>' for column in metrics.columns)}</tr></thead><tbody>{metric_body}</tbody></table>"
        f"<h2>Task status</h2><pre>{json.dumps(status, ensure_ascii=False, indent=2)}</pre>", encoding="utf-8")
    if render_reports and summaries:
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as executor:
            list(executor.map(_render_one, [str(path.parent) for path in summaries]))
    print(json.dumps({"rows": frame.height, "summary": str(input_dir / "batch_summary.parquet"), "report": str(input_dir / "batch_report.html")}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render HTML/PDF report from quant-backtest output")
    parser.add_argument("command", choices=["render", "batch"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--render-reports", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "render": render(args.input)
    else: render_batch(args.input, args.render_reports, args.jobs)


if __name__ == "__main__":
    main()
