import math


def risk_plan(price, probability_up, realized_vol, direction, horizon=1):
    """Risk-management heuristic, explicitly separate from model forecast."""
    chosen_p = probability_up if direction == "LONG" else 1.0-probability_up
    edge = max(0.0, chosen_p-0.5)
    kelly = min(0.25, edge*2.0)

    vol = max(float(realized_vol or 0.0), 0.002)
    scale = math.sqrt(max(1, int(horizon)))
    stop_pct = min(0.20, min(0.05*scale, max(0.006*scale, vol*2.2*scale)))
    tp_pct = min(0.35, min(0.10*scale, max(0.010*scale, stop_pct*1.6)))

    if direction == "LONG":
        stop, take = price*(1-stop_pct), price*(1+tp_pct)
    else:
        stop, take = price*(1+stop_pct), price*(1-tp_pct)

    return {
        "kelly_fraction": round(kelly,4),
        "stop_loss": round(stop,2),
        "take_profit": round(take,2),
        "stop_pct": round(stop_pct*100,2),
        "take_pct": round(tp_pct*100,2),
        "kill_switch": bool(vol > 0.035),
        "note": "SL/TP — risk heuristic, не прогноз цены модели",
    }
