"""Backtest the raw CSI500/CSI1000 80/20 candidate in one account."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb
import numpy as np
import polars as pl
from a_share_data.dual_sleeve import CSI500, CSI1000, SleeveRule, assign_sleeves, blend_joint_scores, decide
from a_share_data.policy import LimitedReplacementConfig, run_limited_replacement_policy
from a_share_data.predict import render_backtest_report


def membership(catalog: Path, dates: list[str]) -> dict[str, dict[str, str]]:
    lo, hi = min(dates), max(dates)
    with duckdb.connect(str(catalog), read_only=True) as c:
        rows = c.execute("""SELECT cal.trade_date::VARCHAR,c.index_code,c.ts_code
          FROM observed_calendar cal JOIN index_monthly_constituents c
            ON c.index_code IN ('000905.SH','000852.SH') AND c.as_of_date=(
              SELECT max(c2.as_of_date) FROM index_monthly_constituents c2
              WHERE c2.index_code=c.index_code AND c2.as_of_date<=cal.trade_date)
          WHERE cal.trade_date BETWEEN ?::DATE AND ?::DATE""", [lo, hi]).fetchall()
    raw: dict[str, dict[str, set[str]]] = {}
    for day, index, code in rows: raw.setdefault(day, {}).setdefault(index, set()).add(code)
    return {day: assign_sleeves(x.get(CSI500, set()), x.get(CSI1000, set())) for day, x in raw.items()}


def joint_scores(frame: pl.DataFrame) -> dict[str, float]:
    """Blend H1/H5 after same-day joint-universe standardization.

    The two LightGBM outputs do not share a stable numeric scale.  Averaging
    their raw values silently changes the intended 50/50 model weighting.
    """
    valid = frame.filter(
        pl.col("pred_h1").is_not_null()
        & pl.col("pred_h5").is_not_null()
        & pl.col("pred_h1").is_finite()
        & pl.col("pred_h5").is_finite()
    )
    return blend_joint_scores(valid["ts_code"], valid["pred_h1"], valid["pred_h5"])


def targets(
    prediction: Path, catalog: Path, output: Path,
    csi500_replacements: int, csi1000_replacements: int,
    shared_replacements: int | None = None,
    csi500_interval: int = 1, csi1000_interval: int = 1,
    csi500_holdings: int = 80, csi1000_holdings: int = 20,
    csi500_replacement_rate: float | None = None,
    csi1000_replacement_rate: float | None = None,
) -> Path:
    p = pl.read_parquet(prediction).sort(["trade_date", "ts_code"])
    days = [str(x) for x in p["trade_date"].unique().sort().to_list()]
    members = membership(catalog, days); held = {CSI500: set(), CSI1000: set()}; out=[]; audit=[]
    rules=(
        SleeveRule(CSI500,csi500_holdings,csi500_holdings,int(csi500_holdings*1.2)),
        SleeveRule(CSI1000,csi1000_holdings,csi1000_holdings,int(csi1000_holdings*1.2)),
    )
    for day_index, (day, frame) in enumerate(p.partition_by("trade_date", maintain_order=True, as_dict=True).items()):
        day_value = day[0] if isinstance(day, tuple) else day; key=str(day_value)
        m=members[key]
        # Standardize on the complete point-in-time CSI500 + CSI1000 cross
        # section, then apply membership.  Filtering first would give the two
        # sleeves different score scales and would no longer be a joint model.
        scores={code: score for code, score in joint_scores(frame).items() if code in m}
        if not held[CSI500] and not held[CSI1000]:
            ranked={s:sorted((c for c in scores if m[c]==s),key=lambda c:(-scores[c],c)) for s in (CSI500,CSI1000)}
            held={CSI500:set(ranked[CSI500][:csi500_holdings]),CSI1000:set(ranked[CSI1000][:csi1000_holdings])}; actions=[]
        else:
            forced={c for s in held.values() for c in s if m.get(c) not in (CSI500,CSI1000)}
            if shared_replacements is None:
                def rate_quota(rate: float | None, count: int, fallback: int, interval: int) -> int:
                    if rate is None:
                        return fallback if day_index % interval == 0 else 0
                    return int(np.floor((day_index + 1) * rate * count) - np.floor(day_index * rate * count))
                limits={
                    CSI500:rate_quota(csi500_replacement_rate,csi500_holdings,csi500_replacements,csi500_interval),
                    CSI1000:rate_quota(csi1000_replacement_rate,csi1000_holdings,csi1000_replacements,csi1000_interval),
                }
                held,actions=decide(
                    scores,m,held,forced,csi500_replacements+csi1000_replacements,
                    sleeve_limits=limits,rules=rules,
                )
            else:
                held,actions=decide(scores,m,held,forced,shared_replacements,rules=rules)
        selected=held[CSI500]|held[CSI1000]
        chosen=frame.filter(pl.col("ts_code").is_in(selected))
        chosen=chosen.with_columns(
            pl.lit(key).alias("decision_date"),
            pl.col("ts_code").replace_strict(m, default=None).alias("sleeve"),
        )
        out.append(chosen)
        audit.append({
            "trade_date":key,"csi500":len(held[CSI500]),"csi1000":len(held[CSI1000]),
            "buys":sum(a.action=="buy" for a in actions),
            "csi500_buys":sum(a.action=="buy" and a.sleeve==CSI500 for a in actions),
            "csi1000_buys":sum(a.action=="buy" and a.sleeve==CSI1000 for a in actions),
        })
    output.mkdir(parents=True,exist_ok=True); path=output/"dual_sleeve_targets.parquet"
    pl.concat(out).write_parquet(path,compression="zstd"); pl.DataFrame(audit).write_parquet(output/"decision_audit.parquet",compression="zstd")
    return path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--catalog",type=Path,required=True); ap.add_argument("--predictions",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--csi500-replacements",type=int,default=1); ap.add_argument("--csi1000-replacements",type=int,default=2)
    ap.add_argument("--csi500-rebalance-interval",type=int,default=1); ap.add_argument("--csi1000-rebalance-interval",type=int,default=1)
    ap.add_argument("--csi500-replacement-rate",type=float); ap.add_argument("--csi1000-replacement-rate",type=float)
    ap.add_argument("--csi500-holdings",type=int,default=80); ap.add_argument("--csi1000-holdings",type=int,default=20)
    ap.add_argument("--shared-replacements",type=int)
    ap.add_argument("--csi500-nav-weight",type=float,default=.80); ap.add_argument("--csi1000-nav-weight",type=float,default=.18); a=ap.parse_args()
    if a.csi500_rebalance_interval < 1 or a.csi1000_rebalance_interval < 1:
        raise ValueError("rebalance intervals must be positive")
    target=targets(
        a.predictions,a.catalog,a.output,a.csi500_replacements,a.csi1000_replacements,a.shared_replacements,
        a.csi500_rebalance_interval,a.csi1000_rebalance_interval,a.csi500_holdings,a.csi1000_holdings,
        a.csi500_replacement_rate,a.csi1000_replacement_rate,
    )
    replacement_limit=a.shared_replacements if a.shared_replacements is not None else a.csi500_replacements+a.csi1000_replacements
    total_holdings=a.csi500_holdings+a.csi1000_holdings
    min_position=min(a.csi500_nav_weight/a.csi500_holdings,a.csi1000_nav_weight/a.csi1000_holdings)
    cfg=LimitedReplacementConfig(target_holdings=total_holdings,entry_rank=total_holdings,exit_rank=total_holdings,max_daily_replacements=replacement_limit,
        cash_reserve=.02,max_weight=.03,rebalance_to_weight=.028,min_new_weight=min_position*.5,daily_buy_budget=.10,daily_sell_budget=.10,
        h1_weight=.5,entry_sizing="cash_balanced",rank_tilt=0,lot_size=100,initial_capital=10_000_000,buy_bps=2.1,sell_bps=7.1,
        sleeve_nav_targets={CSI500:a.csi500_nav_weight,CSI1000:a.csi1000_nav_weight},
        sleeve_target_holdings={CSI500:a.csi500_holdings,CSI1000:a.csi1000_holdings})
    result=run_limited_replacement_policy(a.catalog,target,a.output/"account",cfg)
    result["charged_report"] = render_backtest_report(
        a.output/"account"/"portfolio_daily.parquet",
        a.output/"report",
        "Raw CSI500/CSI1000 80/20 — revised OOS backtest",
    )
    (a.output/"run_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n")
    print(json.dumps(result,ensure_ascii=False,default=str))
if __name__=="__main__": main()
