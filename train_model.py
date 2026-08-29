import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score
from sklearn.model_selection import TimeSeriesSplit

from ml.features import FEATURE_COLUMNS, binary_target, build_features, future_return
from ml.calibration import fit_calibrated as _fit_calibrated_shared

OKX = "https://www.okx.com"
INST = "ETH-USDT-SWAP"
MODEL_PATH = "ml/eth_direction_lgbm.joblib"
REPORT_PATH = "ml/walk_forward_report.json"
HORIZONS = [1, 12, 24]


def okx_get(path, params=None):
    r = requests.get(f"{OKX}{path}", params=params or {}, timeout=25,
                     headers={"User-Agent":"ETH-RADAR-FINAL/1.0"})
    r.raise_for_status()
    p = r.json()
    if p.get("code") != "0":
        raise RuntimeError(p.get("msg") or str(p))
    return p.get("data", [])


def history_1h(target_bars=9000):
    rows, after = [], None
    while len(rows) < target_bars:
        params={"instId":INST,"bar":"1H","limit":"100"}
        if after: params["after"] = str(after)
        chunk=okx_get("/api/v5/market/history-candles",params)
        if not chunk: break
        rows.extend(chunk)
        oldest=min(int(x[0]) for x in chunk)
        if after is not None and oldest >= int(after): break
        after=oldest
        time.sleep(0.08)
        if len(chunk)<100: break
    cols=["t","o","h","l","c","v","vol_ccy","qav","confirm"]
    df=pd.DataFrame(rows,columns=cols).drop_duplicates("t")
    for c in ["t","o","h","l","c","v","qav","confirm"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.sort_values("t").reset_index(drop=True)
    return df[df["confirm"]==1].copy()


def history_funding(min_time_ms, target_records=3000):
    rows, after=[], None
    while len(rows)<target_records:
        params={"instId":INST,"limit":"100"}
        if after: params["after"]=str(after)
        chunk=okx_get("/api/v5/public/funding-rate-history",params)
        if not chunk: break
        rows.extend(chunk)
        oldest=min(int(x["fundingTime"]) for x in chunk)
        if after is not None and oldest>=int(after): break
        after=oldest
        time.sleep(0.08)
        if len(chunk)<100 or oldest<=min_time_ms: break
    if not rows:
        return pd.DataFrame(columns=["fundingTime","fundingRate"])
    f=pd.DataFrame(rows)[["fundingTime","fundingRate"]].drop_duplicates("fundingTime")
    f["fundingTime"]=pd.to_numeric(f["fundingTime"],errors="coerce")
    f["fundingRate"]=pd.to_numeric(f["fundingRate"],errors="coerce")
    return f.sort_values("fundingTime").reset_index(drop=True)


def attach_real_funding(df):
    f=history_funding(int(df["t"].min()))
    out=df.sort_values("t").reset_index(drop=True).copy()
    if f.empty:
        out["funding_hist"]=0.0
        return out
    merged=pd.merge_asof(
        out[["t"]], f.rename(columns={"fundingTime":"t"}).sort_values("t"),
        on="t", direction="backward"
    )
    out["funding_hist"]=merged["fundingRate"].fillna(0.0).values
    return out


def make_base():
    return LGBMClassifier(
        n_estimators=650, learning_rate=0.02, num_leaves=24,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        class_weight=None, verbosity=-1,
    )


def fit_calibrated(X, y, horizon):
    return _fit_calibrated_shared(make_base, X, y, horizon)


def metrics(y_true,p):
    pred=(np.asarray(p)>=0.5).astype(int)
    yv=np.asarray(y_true).astype(int)
    return {
        "accuracy":float(accuracy_score(yv,pred)),
        "brier":float(brier_score_loss(yv,p)),
        "long_precision":float(precision_score(yv,pred,pos_label=1,zero_division=0)),
        "short_precision":float(precision_score(yv,pred,pos_label=0,zero_division=0)),
        "n":int(len(yv)),
        "up_rate":float(yv.mean()),
    }


def train_one(X_all, df, horizon):
    data=X_all.copy()
    data["target"]=binary_target(df,horizon)
    data["future_return"]=future_return(df,horizon)
    data=data.dropna().reset_index(drop=True)
    X=data[FEATURE_COLUMNS]
    y=data["target"].astype(int)

    # Untouched last 15%: final honest out-of-sample report.
    hold_n=max(400, int(len(X)*0.15))
    split=max(800, len(X)-hold_n)
    dev_X, dev_y=X.iloc[:split], y.iloc[:split]
    hold_X, hold_y=X.iloc[split:], y.iloc[split:]

    n_splits=max(3,min(6,len(dev_X)//500))
    splitter=TimeSeriesSplit(n_splits=n_splits,gap=horizon)
    fold_results=[]; oof_p=[]; oof_y=[]; oof_ret=[]
    for fold,(tr,va) in enumerate(splitter.split(dev_X),1):
        if len(tr)<600: continue
        model=fit_calibrated(dev_X.iloc[tr],dev_y.iloc[tr],horizon)
        p=model.predict_proba(dev_X.iloc[va])[:,1]
        fold_results.append({"fold":fold,**metrics(dev_y.iloc[va],p)})
        oof_p.extend(p.tolist()); oof_y.extend(dev_y.iloc[va].tolist())
        oof_ret.extend(data["future_return"].iloc[va].tolist())

    final_model=fit_calibrated(dev_X,dev_y,horizon)
    hold_p=final_model.predict_proba(hold_X)[:,1]
    holdout=metrics(hold_y,hold_p)

    # Expected-return table from walk-forward OOF only, min 30 observations.
    calibration=[]
    if oof_p:
        pp=np.asarray(oof_p); rr=np.asarray(oof_ret)
        buckets=np.clip((pp*10).astype(int),0,9)
        for b in range(10):
            mask=buckets==b
            if int(mask.sum())>=30:
                r=rr[mask]
                actual_up=np.mean(np.asarray(oof_y)[mask])
                calibration.append({
                    "bucket_p_low":round(b/10,2),"bucket_p_high":round((b+1)/10,2),
                    "n":int(mask.sum()),"actual_up_rate":round(float(actual_up),4),
                    "median_future_return_pct":round(float(np.median(r))*100,3),
                    "p05_future_return_pct":round(float(np.quantile(r,0.05))*100,3),
                    "p95_future_return_pct":round(float(np.quantile(r,0.95))*100,3),
                })

    report={
        "horizon_hours":horizon,"rows":int(len(X)),"dev_rows":int(len(dev_X)),
        "holdout_rows":int(len(hold_X)),"folds":fold_results,
        "walk_forward":metrics(oof_y,oof_p) if oof_p else {},
        "holdout":holdout,"calibration":calibration,
        "target":f"Price(t+{horizon}h) > Price(t)",
        "probability_calibration":"sigmoid / Platt, time-ordered calibration split with embargo",
        "holdout_note":"Последние 15% истории не использовались для fit/calibration модели.",
    }
    return final_model, report


def train_model():
    df=history_1h(9000)
    if len(df)<2500: raise RuntimeError(f"Недостаточно истории OKX: {len(df)}")
    df=attach_real_funding(df)
    X_all=build_features(df,funding_series=df["funding_hist"].values)
    models={}; reports={}
    for h in HORIZONS:
        m,r=train_one(X_all,df,h)
        models[f"h{h}"]=m; reports[f"h{h}"]=r
        print(f"h{h}: holdout acc={r['holdout']['accuracy']:.3f} brier={r['holdout']['brier']:.4f}")
    Path("ml").mkdir(exist_ok=True)
    joblib.dump(models,MODEL_PATH)
    Path(REPORT_PATH).write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
    return reports


if __name__=="__main__": train_model()
