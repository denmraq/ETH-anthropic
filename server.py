import json
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from deep_translator import GoogleTranslator

from ml.features import FEATURE_COLUMNS, build_features, live_overlay
from ml.risk import risk_plan
from ml import prediction_log

app=FastAPI(title="ETH RADAR FINAL (Anthropic + GPT merged)")
OKX="https://www.okx.com"; INST="ETH-USDT-SWAP"
RSS="https://app.chaingpt.org/rssfeeds-ethereum.xml"
MODEL_PATH="ml/eth_direction_lgbm.joblib"; REPORT_PATH="ml/walk_forward_report.json"
STATE_PATH=Path("runtime/signal_state.json"); LOG_PATH=Path("runtime/predictions.jsonl")
CACHE_TTL=15
models=joblib.load(MODEL_PATH) if Path(MODEL_PATH).exists() else {}
reports={}
if Path(REPORT_PATH).exists():
    try: reports=json.loads(Path(REPORT_PATH).read_text(encoding="utf-8"))
    except Exception: reports={}
model_training=False; training_error=None
_cache={"ts":0.0,"data":None}; _lock=threading.Lock(); _state_lock=threading.Lock()
_translation_cache={}


def okx_get(path,params=None):
    r=requests.get(f"{OKX}{path}",params=params or {},timeout=20,headers={"User-Agent":"ETH-RADAR-FINAL/1.0"})
    r.raise_for_status(); p=r.json()
    if p.get("code")!="0": raise RuntimeError(p.get("msg") or str(p))
    return p.get("data",[])


def frame(rows):
    cols=["t","o","h","l","c","v","vol_ccy","qav","confirm"]
    df=pd.DataFrame(rows,columns=cols)
    for c in ["t","o","h","l","c","v","qav","confirm"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.sort_values("t").reset_index(drop=True)


def fetch_live():
    h1=frame(okx_get("/api/v5/market/candles",{"instId":INST,"bar":"1H","limit":"160"}))
    m15=frame(okx_get("/api/v5/market/candles",{"instId":INST,"bar":"15m","limit":"80"}))
    book=okx_get("/api/v5/market/books",{"instId":INST,"sz":"50"})[0]
    trades=okx_get("/api/v5/market/trades",{"instId":INST,"limit":"500"})
    funding=okx_get("/api/v5/public/funding-rate",{"instId":INST})
    ticker=okx_get("/api/v5/market/ticker",{"instId":INST})[0]
    f=float(funding[0].get("fundingRate") or 0.0) if funding else 0.0
    price=float(ticker["last"])

    bids=[(float(x[0]),float(x[1])) for x in book.get("bids",[])]; asks=[(float(x[0]),float(x[1])) for x in book.get("asks",[])]
    bv=sum(s for _,s in bids[:20]); av=sum(s for _,s in asks[:20]); denom=bv+av
    ofi=(bv-av)/denom if denom else 0.0
    if bids and asks and bids[0][1]+asks[0][1]>0:
        micro=(asks[0][0]*bids[0][1]+bids[0][0]*asks[0][1])/(bids[0][1]+asks[0][1]); mid=(bids[0][0]+asks[0][0])/2
        micro_edge=(micro-mid)/mid if mid else 0.0
    else: micro_edge=0.0

    buy=sell=0.0
    for tr in trades:
        try:
            notion=float(tr.get("px",0))*float(tr.get("sz",0))
            if tr.get("side")=="buy": buy+=notion
            elif tr.get("side")=="sell": sell+=notion
        except Exception: pass
    tape_ratio=buy/(buy+sell) if buy+sell else 0.5
    return {"h1":h1,"m15":m15,"funding":f,"price":price,"ofi":ofi,"micro_edge":micro_edge,
            "tape_buy_ratio":tape_ratio,"trade_notional":buy+sell,"fetched_at":time.time()}


def live_market():
    with _lock:
        now=time.time()
        if _cache["data"] is not None and now-_cache["ts"]<CACHE_TTL: return _cache["data"]
        d=fetch_live(); _cache.update(ts=now,data=d); return d


def ensure_model():
    global model_training,models,reports,training_error
    if models or model_training: return
    model_training=True; training_error=None
    def worker():
        global model_training,models,reports,training_error
        try:
            from train_model import train_model
            train_model(); models=joblib.load(MODEL_PATH)
            reports=json.loads(Path(REPORT_PATH).read_text(encoding="utf-8"))
        except Exception as e:
            training_error=f"{type(e).__name__}: {e}"
        finally: model_training=False
    threading.Thread(target=worker,daemon=True).start()


@app.on_event("startup")
def startup():
    Path("runtime").mkdir(exist_ok=True); ensure_model()


def load_state():
    default={k:{"direction":None,"probability":0.5,"last_ts":None,"pending":None,"count":0} for k in ["1h","12h","24h"]}
    if not STATE_PATH.exists(): return default
    try:
        got=json.loads(STATE_PATH.read_text(encoding="utf-8")); default.update(got); return default
    except Exception: return default


def save_state(state):
    STATE_PATH.parent.mkdir(exist_ok=True); tmp=STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(STATE_PATH)


def stabilize(key,p,candle_ts,long_thr,short_thr):
    """Binary stabilizer. Reconsiders only on a new CLOSED control candle.
    Flip requires 2 distinct closed-candle confirmations outside threshold.
    State is persisted to disk rather than only global RAM.
    """
    with _state_lock:
        state=load_state(); s=state[key]; ts=str(int(candle_ts))
        raw="LONG" if p>=0.5 else "SHORT"
        if s.get("direction") is None:
            s.update(direction=raw,probability=float(p),last_ts=ts,pending=None,count=0); save_state(state); return s,True
        if s.get("last_ts")==ts: return s,False
        s["last_ts"]=ts; s["probability"]=float(p)
        candidate="LONG" if p>=long_thr else ("SHORT" if p<=short_thr else None)
        if candidate is None or candidate==s["direction"]:
            s["pending"]=None; s["count"]=0
        else:
            if s.get("pending")==candidate: s["count"]=int(s.get("count",0))+1
            else: s["pending"]=candidate; s["count"]=1
            if s["count"]>=2:
                s["direction"]=candidate; s["pending"]=None; s["count"]=0
        save_state(state); return s,True


def return_stats(h,p_up,direction):
    rep=reports.get(f"h{h}",{}); rows=rep.get("calibration",[])
    for r in rows:
        lo,hi=r["bucket_p_low"],r["bucket_p_high"]
        if lo<=p_up<hi or (hi>=1 and p_up==1):
            med,p05,p95=r["median_future_return_pct"],r["p05_future_return_pct"],r["p95_future_return_pct"]
            if direction=="SHORT": med,p05,p95=-med,-p95,-p05
            return {"expected_return_pct":round(med,2),"interval_pct":[round(p05,2),round(p95,2)],"n":r["n"],"actual_up_rate":r["actual_up_rate"]}
    return {"expected_return_pct":None,"interval_pct":None,"n":0,"actual_up_rate":None}


def translate_ru(text):
    if not text: return ""
    if text in _translation_cache: return _translation_cache[text]
    try: ru=GoogleTranslator(source="auto", target="ru").translate(text)
    except Exception: ru=text
    _translation_cache[text]=ru
    return ru


def append_log(obj):
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n")
    except Exception: pass


@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":bool(models),"model_training":model_training,
            "training_error":training_error,"horizons_loaded":sorted(models.keys()),"source":"OKX ETH-USDT-SWAP"}


@app.get("/api/v4/predict")
def predict():
    live=live_market(); h1=live["h1"]; m15=live["m15"]
    closed=h1[h1["confirm"]==1].copy(); closed15=m15[m15["confirm"]==1].copy()
    if len(closed)<100 or len(closed15)<4: raise RuntimeError("Недостаточно закрытых свечей OKX")
    last1h=int(closed["t"].iloc[-1]); last15=int(closed15["t"].iloc[-1]); now=datetime.now(timezone.utc)

    try: prediction_log.resolve_due_predictions(closed, now)
    except Exception: pass  # never let logging/reconciliation break a live prediction

    X=build_features(closed,funding_rate=live["funding"]); row=X.iloc[[-1]][FEATURE_COLUMNS]
    rv=float(np.log(closed["c"]).diff().rolling(24).std().iloc[-1]); rv=0.0 if not np.isfinite(rv) else rv
    last4=closed15.tail(4); recent_ret=float(last4["c"].iloc[-1]/last4["c"].iloc[0]-1.0)
    overlay=live_overlay(live["tape_buy_ratio"],live["ofi"],live["micro_edge"],recent_ret)

    thresholds={1:(0.55,0.45),12:(0.57,0.43),24:(0.58,0.42)}
    packs={}
    for h in [1,12,24]:
        m=models.get(f"h{h}"); backed=m is not None
        p_ml=float(m.predict_proba(row)[0,1]) if backed else 0.5
        # Only 1h gets live microstructure/tape overlay, capped at +/-5 pp.
        p_adj=float(np.clip(p_ml+(0.05*overlay if h==1 else 0.0),0.01,0.99))
        ctrl_ts=last15 if h==1 else last1h
        s,updated=stabilize(f"{h}h",p_adj,ctrl_ts,*thresholds[h])
        direction=s["direction"] or ("LONG" if p_adj>=0.5 else "SHORT")
        chosen=p_adj if direction=="LONG" else 1-p_adj
        stats=return_stats(h,p_adj,direction)
        risk=risk_plan(live["price"],p_adj,rv,direction,h)
        target_time=now+timedelta(hours=h)
        try:
            prediction_log.log_prediction(
                horizon_hours=h, direction=direction, probability_up=round(p_adj,4),
                entry_price=live["price"], now=now, target_time=target_time,
            )
        except Exception: pass
        packs[h]={
            "direction":direction,"probability":round(chosen*100,1),"probability_up":round(p_adj*100,1),
            "ml_probability_up":round(p_ml*100,1),"model_backed":backed,"stabilizer_updated":updated,
            "pending_flip":s.get("pending"),"pending_confirmations":s.get("count",0),
            "forecast_target_time":target_time.isoformat(),"return_stats":stats,"risk":risk,
        }

    result={"status":"ok","symbol":"ETH/USDT","price":round(live["price"],2),"source":"OKX ETH-USDT-SWAP",
        "generated_at":now.isoformat(),"as_of_closed_candle":{"time":datetime.fromtimestamp(last1h/1000,tz=timezone.utc).isoformat(),"close_price":round(float(closed["c"].iloc[-1]),2)},
        "horizon_1h":packs[1],"horizon_12h":packs[12],"horizon_24h":packs[24],
        "live_context":{"funding_rate":live["funding"],"recent_trade_buy_ratio":round(live["tape_buy_ratio"],4),
            "recent_trade_notional":round(live["trade_notional"],2),"orderbook_ofi":round(live["ofi"],4),
            "microprice_edge":round(live["micro_edge"],6),"recent_15m_1h_return":round(recent_ret,6),
            "microstructure_overlay":round(overlay,4),"overlay_note":"Только 1ч, эвристика максимум +/-5 п.п.; не обучаемая историческая фича."}}
    append_log({"generated_at":result["generated_at"],"price":result["price"],"h1":packs[1],"h12":packs[12],"h24":packs[24]})
    return result


@app.get("/api/v4/validation")
def validation():
    if not reports:
        return {"status":"not_ready","model_training":model_training,"training_error":training_error}
    return {"status":"ok","horizons":reports,"live_tracked_accuracy":prediction_log.live_accuracy_stats()}


@app.get("/api/v4/news")
def news():
    try:
        r=requests.get(RSS,timeout=20,headers={"User-Agent":"ETH-RADAR-FINAL/1.0"})
        r.raise_for_status(); root=ET.fromstring(r.content)
    except Exception as e:
        return {"status":"error","source":"ChainGPT Ethereum RSS","articles":[],"error":str(e)}
    out=[]
    for item in root.findall(".//item")[:6]:
        title=(item.findtext("title") or "").strip()
        desc=(item.findtext("description") or "").strip()
        link=(item.findtext("link") or "").strip()
        pub=(item.findtext("pubDate") or "").strip()
        text=(title+" "+desc).lower()
        pos=sum(k in text for k in ["approval","inflow","adoption","surge","rally","growth","institutional"])
        neg=sum(k in text for k in ["hack","exploit","outflow","lawsuit","ban","crash","liquidation"])
        sentiment="BULLISH" if pos>neg else ("BEARISH" if neg>pos else "NEUTRAL")
        out.append({"title":translate_ru(title),"link":link,"published":pub,"sentiment":sentiment})
    return {"status":"ok","source":"ChainGPT Ethereum RSS","articles":out,
            "note":"Новости не участвуют в ML-сигнале — справочный контекст."}


@app.get("/api/v4/log/status")
def log_status():
    n=0
    if LOG_PATH.exists():
        try:
            with LOG_PATH.open("r",encoding="utf-8") as f: n=sum(1 for _ in f)
        except Exception: pass
    pending_n=len(prediction_log._read_jsonl(prediction_log.PENDING_PATH))
    resolved_n=len(prediction_log._read_jsonl(prediction_log.RESOLVED_PATH))
    return {"status":"ok","logged_predictions":n,"path":str(LOG_PATH),
            "pending_reconciliation":pending_n,"resolved_reconciliation":resolved_n}

app.mount("/static",StaticFiles(directory="static"),name="static")
@app.get("/")
def root(): return FileResponse("static/index.html")
