"""Paired three-seed 2025Q2 residual-LSTM industry/liquidity embedding pilot."""
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
from a_share_data.sequence_train import FactorLSTM, _targets, require_mps


class EntityConditionedLSTM(nn.Module):
    """Baseline-identical backbone plus an initially-zero gated entity residual."""
    def __init__(self, channels: int, industry_count: int):
        super().__init__()
        # Construct this first so its parameters exactly match FactorLSTM for the same seed.
        self.backbone = FactorLSTM(channels, hidden=(128, 64), dropout=.1,
                                   recurrent_residual=True, output_residual=True,
                                   zero_init_output_residual=True)
        self.industry = nn.Embedding(industry_count, 8)
        self.liquidity = nn.Embedding(11, 4)
        self.context = nn.Linear(64 + 12, 32)
        self.gate = nn.Linear(32, 2)
        self.value = nn.Linear(32, 2)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, values, industry, liquidity):
        raw_last = values[:, -1]
        hidden = values
        for index, layer in enumerate(self.backbone.recurrent):
            previous = hidden
            hidden, _ = layer(hidden)
            hidden = hidden + self.backbone.residuals[index](previous)
            if index + 1 < len(self.backbone.recurrent):
                hidden = self.backbone.dropout(hidden)
        last = hidden[:, -1]
        baseline = self.backbone.output(last) + self.backbone.output_residual(raw_last)
        entity = torch.cat((self.industry(industry), self.liquidity(liquidity)), dim=1)
        context = torch.nn.functional.gelu(self.context(torch.cat((last, entity), dim=1)))
        return baseline + torch.sigmoid(self.gate(context)) * self.value(context)


def category_arrays(cache, entity_path, sets):
    all_ids = np.unique(np.concatenate(sets)); ends = np.asarray(cache.sample_rows[all_ids], np.int64)
    frame = pl.read_parquet(entity_path)
    by_row = {int(a): (int(b), int(c)) for a, b, c in frame.iter_rows()}
    by_id = {int(i): by_row[int(row)] for i, row in zip(all_ids, ends)}
    return [np.asarray([by_id[int(i)] for i in ids], np.int64) for ids in sets]


def run_epoch(cache, ids, cats, targets, masks, model, device, optimizer, seed):
    training = optimizer is not None; model.train(training)
    order = np.arange(len(ids))
    if training: np.random.default_rng(seed).shuffle(order)
    total=0.; count=0
    with (torch.enable_grad() if training else torch.no_grad()):
        for start in range(0,len(order),512):
            pos=order[start:start+512]; x,_,_=cache.batch(ids[pos]); cat=torch.from_numpy(cats[pos]).to(device)
            xt=torch.from_numpy(x).to(device)
            pred=model(xt) if isinstance(model,FactorLSTM) else model(xt,cat[:,0],cat[:,1])
            y=torch.from_numpy(targets[pos]).to(device); mask=torch.from_numpy(masks[pos]).to(device)
            loss=torch.stack([(pred[mask[:,h],h]-y[mask[:,h],h]).square().mean()
                              for h in range(2) if mask[:,h].any()]).mean()
            if training:
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach().cpu())*len(pos);count+=len(pos)
    if device.type=="mps":torch.mps.synchronize()
    return total/count


def make_model(kind, channels, industry_count, seed, device):
    torch.manual_seed(seed)
    if kind=="baseline":
        model=FactorLSTM(channels,hidden=(128,64),dropout=.1,recurrent_residual=True,
                         output_residual=True,zero_init_output_residual=True)
    else:model=EntityConditionedLSTM(channels,industry_count)
    return model.to(device)


def fit(kind, cache, sets, cats, device, industry_count, seed):
    fit_ids,valid_ids,full_ids,test_ids=sets;fit_cat,valid_cat,full_cat,test_cat=cats
    fit_y,fit_mask,means,scales=_targets(cache,fit_ids,winsorize=True)
    valid_y,valid_mask,_,_=_targets(cache,valid_ids,winsorize=False,means=means,scales=scales)
    model=make_model(kind,cache.factor_count*2,industry_count,seed,device)
    # Construction of the candidate consumes extra RNG; reset training RNG for a fair backbone path.
    torch.manual_seed(seed+500000)
    opt=torch.optim.Adam(model.parameters(),lr=3e-5,eps=1e-8,weight_decay=1e-4)
    best=float("inf");best_epoch=0;stale=0;history=[]
    for number in range(1,31):
        tr=run_epoch(cache,fit_ids,fit_cat,fit_y,fit_mask,model,device,opt,seed+number)
        va=run_epoch(cache,valid_ids,valid_cat,valid_y,valid_mask,model,device,None,seed)
        history.append({"epoch":number,"training_loss":tr,"validation_loss":va})
        print(json.dumps({"seed":seed,"model":kind,**history[-1]}),flush=True)
        if va<best:best=va;best_epoch=number;stale=0
        else:
            stale+=1
            if stale>=4:break
    full_y,full_mask,final_means,final_scales=_targets(cache,full_ids,winsorize=True)
    final=make_model(kind,cache.factor_count*2,industry_count,seed,device)
    torch.manual_seed(seed+600000)
    final_opt=torch.optim.Adam(final.parameters(),lr=3e-5,eps=1e-8,weight_decay=1e-4)
    for number in range(1,best_epoch+1):
        run_epoch(cache,full_ids,full_cat,full_y,full_mask,final,device,final_opt,seed+10000+number)
    final.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(test_ids),512):
            ids=test_ids[start:start+512];x,_,_=cache.batch(ids);cat=torch.from_numpy(test_cat[start:start+512]).to(device)
            xt=torch.from_numpy(x).to(device);pred=final(xt) if kind=="baseline" else final(xt,cat[:,0],cat[:,1])
            out.append(pred.cpu().numpy()*final_scales+final_means)
    return np.concatenate(out),{"best_epoch":best_epoch,"best_validation_loss":best,"history":history},final


def metrics(frame,prefix):
    out={}
    for h in ("h1","h5"):
        daily=frame.select("trade_date",f"{prefix}_{h}",f"target_{h}").drop_nulls().group_by("trade_date").agg(
            pl.corr(pl.col(f"{prefix}_{h}").rank(),pl.col(f"target_{h}").rank()).alias("ic"))
        out[h]={"days":daily.height,"mean_rank_ic":float(daily["ic"].mean()),
                "positive_ratio":float((daily["ic"]>0).mean())}
    return out


def main(args):
    cache=SequenceCache(args.cache);spec=next(x for x in json.loads(args.windows.read_text()) if x["signal"].startswith(args.signal_month))
    dates=cache.metadata.select("trade_date","day_index").unique().sort("day_index");day_by_date=dict(dates.iter_rows())
    train=np.asarray([day_by_date[date.fromisoformat(x)] for x in spec["train_dates"]],np.int32)
    test=np.asarray([day_by_date[date.fromisoformat(x)] for x in spec["test_dates"]],np.int32)
    sets=[cache.sample_ids_for_days(train[:687]),cache.sample_ids_for_days(train[-63:]),
          cache.sample_ids_for_days(train),cache.sample_ids_for_days(test)]
    cats=category_arrays(cache,args.entity_data,sets);audit=json.loads(args.entity_data.with_suffix('.json').read_text())
    ends=np.asarray(cache.sample_rows[sets[-1]],np.int64);target=np.asarray(cache.targets[ends],np.float32)
    keys=cache.metadata.filter(pl.col("row_index").is_in(ends)).select("trade_date","ts_code")
    device=require_mps();runs=[];frames=[]
    for seed in args.seeds:
        predictions={};training={}
        for kind in args.models:
            pred,info,model=fit(kind,cache,sets,cats,device,audit["industry_embedding_count"],seed)
            predictions[kind]=pred;training[kind]=info
            torch.save(model.state_dict(),args.output/f"{kind}_seed{seed}.pt")
        expressions=[pl.Series("target_h1",target[:,0]),pl.Series("target_h5",target[:,1])]
        for kind in args.models:
            expressions.extend((pl.Series(f"{kind}_h1",predictions[kind][:,0]),
                                pl.Series(f"{kind}_h5",predictions[kind][:,1])))
        frame=keys.with_columns(expressions)
        result={"seed":seed,"training":training,
                "models":{kind:metrics(frame,kind) for kind in args.models}}
        runs.append(result);frames.append(frame.with_columns(pl.lit(seed).alias("seed")))
    summary={"signal_month":args.signal_month,"categories":audit,"runs":runs,"aggregate":{}}
    for kind in args.models:
        summary["aggregate"][kind]={h:{"mean_rank_ic":float(np.mean([r["models"][kind][h]["mean_rank_ic"] for r in runs])),
            "std_rank_ic":float(np.std([r["models"][kind][h]["mean_rank_ic"] for r in runs],ddof=1))} for h in ("h1","h5")}
    if "entity" in args.models and "baseline" in args.models:
        summary["aggregate"]["entity_minus_baseline"]={h:float(np.mean([r["models"]["entity"][h]["mean_rank_ic"]-r["models"]["baseline"][h]["mean_rank_ic"] for r in runs])) for h in ("h1","h5")}
    pl.concat(frames).write_parquet(args.output/"paired_predictions.parquet",compression="zstd")
    (args.output/"report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    root=Path("results/predict/sequence-lstm-residual-raw125-v1");p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,default=root/"sequence_cache/7b65e9b210f84533")
    p.add_argument("--windows",type=Path,default=root/"full/windows.json")
    p.add_argument("--entity-data",type=Path,default=root/"entity_embedding_pilot_v2/month=2025-04/entity_categories.parquet")
    p.add_argument("--output",type=Path,default=root/"entity_embedding_pilot_v2/month=2025-04")
    p.add_argument("--signal-month",default="2025-04")
    p.add_argument("--seeds",type=int,nargs="+",default=[20260908,20260909,20260910])
    p.add_argument("--models",nargs="+",choices=("baseline","entity"),default=["baseline","entity"])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);main(args)
