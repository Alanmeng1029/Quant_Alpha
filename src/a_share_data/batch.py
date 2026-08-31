"""One-command orchestration for cached Rust factor batches and Python reports."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from a_share_data.report import render_batch


PROJECT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cached batch-factor-eval, then render its reports")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Build the release engine, evaluate factors, then render reports")
    run.add_argument("--catalog", required=True, type=Path)
    run.add_argument("--factor-root", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--batch-id", required=True, help="Stable ID enables cache and task resume")
    run.add_argument("--universes", default="csi300_csi500")
    run.add_argument("--tasks")
    run.add_argument("--csi300-config", type=Path, default=Path("configs/factor_eval.yaml"))
    run.add_argument("--csi500-config", type=Path, default=Path("configs/factor_eval_csi500.yaml"))
    run.add_argument("--csi300-csi500-config", type=Path, default=Path("configs/factor_eval_csi300_csi500.yaml"))
    run.add_argument("--all-config", type=Path, default=Path("configs/factor_eval_all.yaml"))
    run.add_argument("--rebuild-cache", action="store_true")
    run.add_argument("--report-jobs", type=int, default=2)
    run.add_argument("--skip-reports", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    build_started = time.perf_counter()
    subprocess.run(["cargo", "build", "--release", "-p", "quant-backtest"], cwd=PROJECT, check=True)
    build_seconds = time.perf_counter() - build_started
    binary = PROJECT / "target" / "release" / "quant-backtest"
    command = [
        str(binary), "batch-factor-eval", "--catalog", str(args.catalog), "--factor-root", str(args.factor_root),
        "--output", str(args.output), "--batch-id", args.batch_id, "--universes", args.universes,
        "--csi300-config", str(args.csi300_config), "--csi500-config", str(args.csi500_config),
        "--csi300-csi500-config", str(args.csi300_csi500_config),
        "--all-config", str(args.all_config),
    ]
    if args.tasks:
        command.extend(["--tasks", args.tasks])
    if args.rebuild_cache:
        command.append("--rebuild-cache")
    rust_started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT, check=False)
    rust_seconds = time.perf_counter() - rust_started
    batch_root = args.output / args.batch_id
    report_seconds = 0.0
    if batch_root.exists() and not args.skip_reports:
        report_started = time.perf_counter()
        render_batch(batch_root, render_reports=True, jobs=max(1, args.report_jobs))
        report_seconds = time.perf_counter() - report_started
    if batch_root.exists():
        (batch_root / "orchestration_timing.json").write_text(json.dumps({
            "cargo_release_build_seconds": build_seconds,
            "rust_batch_seconds": rust_seconds,
            "python_report_seconds": report_seconds,
            "total_elapsed_seconds": time.perf_counter() - started,
            "rust_exit_code": result.returncode,
        }, indent=2), encoding="utf-8")
    return result.returncode


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        status = run(args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
