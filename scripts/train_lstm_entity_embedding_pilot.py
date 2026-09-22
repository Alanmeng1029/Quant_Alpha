"""One-quarter residual-LSTM pilot with industry/size/liquidity embeddings."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from a_share_data.sequence_data import SequenceCache
from a_share_data.sequence_train import _targets, require_mps


class EntityFactorLSTM(nn.Module):
    def __init__(self, channels: int, industry_count: int, hidden=(128, 64), dropout=.1):
        super().__init__()
        self.industry = nn.Embedding(industry_count, 8, padding_idx=0)
        self.size = nn.Embedding(11, 4, padding_idx=0)
        self.liquidity = nn.Embedding(11, 4, padding_idx=0)
        widths = (channels,) + tuple(hidden)
        self.recurrent = nn.ModuleList(
            nn.LSTM(widths[i], widths[i + 1], batch_first=True) for i in range(len(hidden)))
        self.residuals = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1], bias=False) for i in range(len(hidden)))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden[-1] + 16, 2)
        self.output_residual = nn.Linear(channels, 2)
        nn.init.zeros_(self.output_residual.weight)
        nn.init.zeros_(self.output_residual.bias)

    def forward(self, values, industry, size, liquidity):
        raw_last = values[:, -1]
        for i, layer in enumerate(self.recurrent):
            previous = values
            values, _ = layer(values)
            values = values + self.residuals[i](previous)
            if i + 1 < len(self.recurrent):
                values = self.dropout(values)
        entity = torch.cat((self.industry(industry), self.size(size),
                            self.liquidity(liquidity)), dim=1)
        return self.output(torch.cat((values[:, -1], entity), dim=1)) + self.output_residual(raw_last)


def categorical_inputs(cache: SequenceCache, sample_ids: np.ndarray, entity_data: Path):
    ends = np.asarray(cache.sample_rows[sample_ids], dtype=np.int64)
    attributes = pl.read_parquet(entity_data)
    by_row = {int(r[0]): tuple(map(int, r[1:])) for r in attributes.select(
        "row_index", "industry_id", "size_group", "liquidity_group").iter_rows()}
    categories = np.asarray([by_row[int(row)] for row in ends], dtype=np.int64)
    audit = json.loads(entity_data.with_suffix(".json").read_text())
    return categories, audit


def epoch(cache, ids, categories, values, masks, model, device, batch_size, optimizer, seed):
    training = optimizer is not None
    model.train(training)
    order = np.arange(len(ids));
    if training: np.random.default_rng(seed).shuffle(order)
    total = 0.; count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for start in range(0, len(order), batch_size):
            positions = order[start:start + batch_size]
            batch_ids = ids[positions]
            x, _, _ = cache.batch(batch_ids)
            cat = torch.from_numpy(categories[positions]).to(device)
            prediction = model(torch.from_numpy(x).to(device), cat[:, 0], cat[:, 1], cat[:, 2])
            target = torch.from_numpy(values[positions]).to(device)
            mask = torch.from_numpy(masks[positions]).to(device)
            losses = [(prediction[mask[:, h], h] - target[mask[:, h], h]).square().mean()
                      for h in range(2) if mask[:, h].any()]
            loss = torch.stack(losses).mean()
            if training:
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(positions); count += len(positions)
    if device.type == "mps": torch.mps.synchronize()
    return total / count


def rank_metrics(frame: pl.DataFrame, prefix: str):
    result = {}
    for horizon in ("h1", "h5"):
        daily = frame.select("trade_date", f"{prefix}_{horizon}", f"target_{horizon}").drop_nulls().group_by(
            "trade_date").agg(pl.corr(pl.col(f"{prefix}_{horizon}").rank(),
                                      pl.col(f"target_{horizon}").rank()).alias("ic"))
        result[horizon] = {"days": daily.height, "mean_rank_ic": float(daily["ic"].mean()),
                           "positive_ratio": float((daily["ic"] > 0).mean())}
    return result


def main(args):
    cache = SequenceCache(args.cache)
    windows = json.loads(args.windows.read_text())
    spec = next(x for x in windows if x["signal"].startswith(args.quarter))
    dates = cache.metadata.select("trade_date", "day_index").unique().sort("day_index")
    day_by_date = dict(dates.iter_rows())
    train_days = np.asarray([day_by_date[date.fromisoformat(x)] for x in spec["train_dates"]], np.int32)
    test_days = np.asarray([day_by_date[date.fromisoformat(x)] for x in spec["test_dates"]], np.int32)
    fit_ids = cache.sample_ids_for_days(train_days[:687]); valid_ids = cache.sample_ids_for_days(train_days[-63:])
    full_ids = cache.sample_ids_for_days(train_days); test_ids = cache.sample_ids_for_days(test_days)
    all_ids = np.unique(np.concatenate((fit_ids, valid_ids, full_ids, test_ids)))
    all_categories, category_audit = categorical_inputs(cache, all_ids, args.entity_data)
    category_by_id = {int(sample_id): category for sample_id, category in zip(all_ids, all_categories)}
    fit_cat, valid_cat, full_cat, test_cat = [
        np.asarray([category_by_id[int(sample_id)] for sample_id in ids], dtype=np.int64)
        for ids in (fit_ids, valid_ids, full_ids, test_ids)]
    fit_y, fit_mask, means, scales = _targets(cache, fit_ids, winsorize=True)
    valid_y, valid_mask, _, _ = _targets(cache, valid_ids, winsorize=False, means=means, scales=scales)
    device = require_mps(); seed = 20260908; torch.manual_seed(seed)
    model = EntityFactorLSTM(cache.factor_count * 2, category_audit["industry_categories"] + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-8, weight_decay=1e-4)
    best_loss=float("inf"); best_epoch=0; stale=0; history=[]
    for number in range(1, 31):
        train_loss=epoch(cache,fit_ids,fit_cat,fit_y,fit_mask,model,device,512,optimizer,seed+number)
        valid_loss=epoch(cache,valid_ids,valid_cat,valid_y,valid_mask,model,device,512,None,seed)
        history.append({"epoch":number,"training_loss":train_loss,"validation_loss":valid_loss})
        print(json.dumps(history[-1]),flush=True)
        if valid_loss < best_loss: best_loss=valid_loss;best_epoch=number;stale=0
        else:
            stale += 1
            if stale >= 4: break
    torch.manual_seed(seed)
    full_y, full_mask, final_means, final_scales = _targets(cache, full_ids, winsorize=True)
    final = EntityFactorLSTM(cache.factor_count * 2, category_audit["industry_categories"] + 1).to(device)
    final_opt = torch.optim.Adam(final.parameters(),lr=3e-5,eps=1e-8,weight_decay=1e-4)
    for number in range(1,best_epoch+1):
        epoch(cache,full_ids,full_cat,full_y,full_mask,final,device,512,final_opt,seed+10000+number)
    final.eval(); predictions=[]
    with torch.no_grad():
        for start in range(0,len(test_ids),512):
            ids=test_ids[start:start+512];x,_,_=cache.batch(ids)
            cat=torch.from_numpy(test_cat[start:start+512]).to(device)
            y=final(torch.from_numpy(x).to(device),cat[:,0],cat[:,1],cat[:,2]).cpu().numpy()
            predictions.append(y*final_scales+final_means)
    prediction=np.concatenate(predictions); ends=np.asarray(cache.sample_rows[test_ids],dtype=np.int64)
    target=np.asarray(cache.targets[ends],dtype=np.float32)
    result=cache.metadata.filter(pl.col("row_index").is_in(ends)).select("trade_date","ts_code").with_columns(
        pl.Series("embedding_h1",prediction[:,0]),pl.Series("embedding_h5",prediction[:,1]),
        pl.Series("target_h1",target[:,0]),pl.Series("target_h5",target[:,1]))
    baseline=pl.read_parquet(args.baseline).select("trade_date","ts_code",
        pl.col("raw_h1").alias("baseline_h1"),pl.col("raw_h5").alias("baseline_h5"))
    compared=result.join(baseline,on=["trade_date","ts_code"],how="inner")
    if compared.height != result.height: raise ValueError("baseline prediction keys differ from pilot")
    report={"quarter":args.quarter,"best_epoch":best_epoch,"best_validation_loss":best_loss,
            "history":history,"categories":category_audit,"samples":{"fit":len(fit_ids),
            "validation":len(valid_ids),"full":len(full_ids),"test":len(test_ids)},
            "models":{"baseline":rank_metrics(compared,"baseline"),
                      "entity_embedding":rank_metrics(compared,"embedding")}}
    args.output.mkdir(parents=True,exist_ok=True)
    compared.write_parquet(args.output/"predictions_and_targets.parquet",compression="zstd")
    torch.save({"state_dict":final.state_dict(),"industry_count":category_audit["industry_categories"]+1},
               args.output/"checkpoint.pt")
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    root=Path("results/predict/sequence-lstm-residual-raw125-v1")
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=root/"sequence_cache/7b65e9b210f84533")
    parser.add_argument("--windows",type=Path,default=root/"full/windows.json")
    parser.add_argument("--entity-data",type=Path,default=root/"entity_embedding_pilot/month=2025-04/entity_categories.parquet")
    parser.add_argument("--baseline",type=Path,default=root/"full/lstm/quarters/month=2025-04/predictions.parquet")
    parser.add_argument("--quarter",default="2025-04")
    parser.add_argument("--output",type=Path,default=root/"entity_embedding_pilot/month=2025-04")
    main(parser.parse_args())
