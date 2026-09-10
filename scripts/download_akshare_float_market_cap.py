#!/usr/bin/env python3
"""Download CNInfo share-capital history through AkShare and create monthly caps.

The CNInfo ``已流通股份`` and ``总股本`` values are in ten-thousand shares.
Rows become usable on their announcement date, rather than their effective date,
so an exposure cannot use a later disclosure.  Market caps use the unadjusted
daily close because a forward-adjusted close cannot be multiplied by actual
shares outstanding.

The per-code cache makes an interrupted download safe to resume.  This is a
public-data acquisition utility, not a substitute for a licensed PIT vendor.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import akshare as ak
import duckdb
import numpy as np
import pandas as pd


def cninfo_history(code: str, start: str, end: str, cache: Path, retries: int) -> dict:
    target = cache / f"{code}.parquet"
    if target.exists():
        return {"code": code, "status": "cached"}
    error = None
    for attempt in range(retries):
        try:
            frame = ak.stock_share_change_cninfo(symbol=code, start_date=start, end_date=end)
            if frame.empty:
                frame = pd.DataFrame(columns=["公告日期", "变动日期", "已流通股份", "总股本"])
            frame.to_parquet(target, index=False)
            return {"code": code, "status": "downloaded", "rows": len(frame)}
        except Exception as exc:  # public endpoint can intermittently throttle
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30, 2 ** attempt))
    return {"code": code, "status": "failed", "error": error}


def normalize_history(code: str, path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    wanted = ["公告日期", "变动日期", "已流通股份", "总股本"]
    if raw.empty or not set(wanted).issubset(raw.columns):
        return pd.DataFrame(columns=["ts_code", "available_date", "float_shares", "total_shares"])
    out = raw[wanted].copy()
    # Announcement date is the point-in-time availability date.  Use change
    # date only when the source omitted an announcement date.
    out["available_date"] = pd.to_datetime(out["公告日期"], errors="coerce").fillna(pd.to_datetime(out["变动日期"], errors="coerce"))
    for source, target in [("已流通股份", "float_shares"), ("总股本", "total_shares")]:
        out[target] = pd.to_numeric(out[source], errors="coerce") * 10_000.0
    out = out.dropna(subset=["available_date"]).sort_values("available_date")
    out.loc[out["float_shares"] <= 0, "float_shares"] = np.nan
    out.loc[out["total_shares"] <= 0, "total_shares"] = np.nan
    out["ts_code"] = code
    return out[["ts_code", "available_date", "float_shares", "total_shares"]].drop_duplicates(["ts_code", "available_date"], keep="last")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index-code", default="000905.SH")
    p.add_argument("--start", default="20000101")
    p.add_argument("--end", default=time.strftime("%Y%m%d"))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--limit", type=int, help="limit codes; useful only for a connectivity smoke test")
    p.add_argument("--refresh", action="store_true", help="redownload even when a per-code cache exists")
    args = p.parse_args()
    cache = args.output / "cninfo_share_change_cache"
    cache.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.catalog), read_only=True)
    try:
        universe = con.execute("SELECT DISTINCT ts_code FROM index_trading_universe WHERE index_code=? ORDER BY ts_code", [args.index_code]).fetchdf()
        prices = con.execute("""
          WITH u AS (SELECT DISTINCT trade_date, ts_code FROM index_trading_universe WHERE index_code = ?)
          SELECT u.trade_date,u.ts_code,d.close
          FROM u JOIN daily_aggregated d USING(trade_date,ts_code)
          ORDER BY ts_code,trade_date
        """, [args.index_code]).fetchdf()
    finally:
        con.close()
    codes = [str(v).split(".")[0].zfill(6) for v in universe.ts_code]
    if args.limit is not None:
        codes = codes[:args.limit]
        prices = prices[prices.ts_code.str.split(".").str[0].isin(codes)].copy()
    if args.refresh:
        for code in codes:
            (cache / f"{code}.parquet").unlink(missing_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(cninfo_history, code, args.start, args.end, cache, args.retries) for code in codes]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); results.append(result)
            if result["status"] == "failed": print(json.dumps(result, ensure_ascii=False))
    history = pd.concat([normalize_history(code, cache / f"{code}.parquet") for code in codes if (cache / f"{code}.parquet").exists()], ignore_index=True)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).astype("datetime64[ns]")
    prices["code"] = prices.ts_code.str.split(".").str[0]
    panels = []
    for code, group in prices.groupby("code", sort=False):
        records = history.loc[history.ts_code == code].sort_values("available_date").copy()
        records["available_date"] = pd.to_datetime(records["available_date"]).astype("datetime64[ns]")
        if records.empty:
            group[["float_shares", "total_shares"]] = np.nan
            panels.append(group); continue
        panels.append(pd.merge_asof(group.sort_values("trade_date"), records.drop(columns="ts_code"), left_on="trade_date", right_on="available_date", direction="backward"))
    daily = pd.concat(panels, ignore_index=True)
    daily["float_mv"] = daily.close * daily.float_shares
    daily["total_mv"] = daily.close * daily.total_shares
    daily = daily[["trade_date", "ts_code", "close", "available_date", "float_shares", "total_shares", "float_mv", "total_mv"]]
    monthly = (daily.assign(month=daily.trade_date.dt.to_period("M"))
               .sort_values(["ts_code", "trade_date"])
               .groupby(["ts_code", "month"], as_index=False).tail(1)
               .drop(columns="month"))
    daily.to_parquet(args.output / "akshare_daily_market_cap.parquet", index=False)
    monthly.to_parquet(args.output / "akshare_monthly_market_cap.parquet", index=False)
    status = {name: sum(r["status"] == name for r in results) for name in {r["status"] for r in results}}
    metadata = {"source": "AkShare stock_share_change_cninfo / CNInfo", "index_code": args.index_code, "share_unit": "shares (source values multiplied by 10000)", "availability": "announcement date, falling back to change date only if unavailable", "price": "unadjusted daily_aggregated.close", "codes": len(codes), "download_status": status, "daily_rows": len(daily), "monthly_rows": len(monthly), "float_mv_coverage": float(daily.float_mv.notna().mean())}
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
