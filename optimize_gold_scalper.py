"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — OPTIMIZER v4.0 (SIGNAL AUDIT)     ║
║  Test elk signaal individueel: wat helpt, wat moet eruit?    ║
║  Includes v1.3: VWAP, Session Levels, Double Pattern, ADX   ║
║  Output: Keep/Remove per signaal + beste combinatie          ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, math, time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy"); sys.exit(1)
try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); sys.exit(1)


class Direction(Enum):
    LONG = "buy"
    SHORT = "sell"

class TradePhase(Enum):
    OPEN = "open"
    TP1_HIT = "tp1_hit"
    CLOSED = "closed"


@dataclass
class Config:
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 0.5
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 3
    MAX_DAILY_TRADES: int = 25
    MAX_CONSECUTIVE_LOSSES: int = 5

    # Optimized core from v1
    ATR_SL_MULTIPLIER: float = 1.0
    MIN_SL_POINTS: float = 2.0
    MAX_SL_POINTS: float = 10.0
    RR_RATIO: float = 2.5
    PARTIAL_PERCENT: float = 0.50
    TP1_RR: float = 0.8
    MOVE_SL_TO_BE: bool = True

    SWING_LOOKBACK: int = 3
    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    EXHAUSTION_WICK_RATIO: float = 0.65
    ROUND_NUMBER_INTERVAL: float = 50.0
    ROUND_NUMBER_ZONE: float = 3.0
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    MR_CONFLUENCE_SCORE: int = 2

    MIN_CONFLUENCE: int = 3
    TRADE_COOLDOWN_BARS: int = 5
    LOSS_COOLDOWN_BARS: int = 9

    # ─── Toggleable signals (each can be ON/OFF) ─────────────
    SIG_EMA: bool = True           # 1. EMA 9/21 trend
    SIG_EMA50: bool = False        # 2. EMA 50 trend filter
    SIG_SWEEP: bool = True         # 3. Liquidity sweep
    SIG_OB: bool = True            # 4. Order block / engulfing
    SIG_FVG: bool = True           # 5. Fair value gap
    SIG_MOMENTUM: bool = True      # 6. Momentum candle
    SIG_EXHAUSTION: bool = True    # 7. Exhaustion candle
    SIG_WICK: bool = True          # 8. Wick rejection
    SIG_ROUND_NUM: bool = True     # 9. Round number
    SIG_MR: bool = True            # 10. Mean Reversion (BB+RSI)
    SIG_RSI_DIV: bool = True       # 11. RSI divergence
    SIG_VWAP: bool = True          # 12. VWAP (NEW v1.3)
    SIG_SESSION_LVL: bool = True   # 13. Session high/low (NEW v1.3)
    SIG_DOUBLE: bool = True        # 14. Double bottom/top (NEW v1.3)
    SIG_ADX: bool = True           # 15. ADX trend strength (NEW v1.3)

    # ADX settings
    ADX_THRESHOLD: float = 25.0    # only trend bonus above this
    ADX_BLOCK_BELOW: float = 0.0   # block trades below this (0=disabled)


# ═══════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════

def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["tr"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()

    # RSI 7
    d = df["Close"].diff()
    g = d.clip(lower=0).rolling(7).mean()
    l = (-d.clip(upper=0)).rolling(7).mean()
    rs = g / l.replace(0, np.nan)
    df["rsi"] = 100 - (100/(1+rs))

    # RSI 14
    d14 = df["Close"].diff()
    g14 = d14.clip(lower=0).rolling(14).mean()
    l14 = (-d14.clip(upper=0)).rolling(14).mean()
    rs14 = g14 / l14.replace(0, np.nan)
    df["rsi14"] = 100 - (100/(1+rs14))

    # Bollinger
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]

    # Candle metrics
    df["body"] = abs(df["Close"]-df["Open"])
    df["candle_range"] = df["High"]-df["Low"]
    df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]

    # VWAP (reset daily)
    if "Volume" in df.columns:
        df["avg_volume"] = df["Volume"].rolling(20).mean()
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = df["Volume"].replace(0, 1)
        df["cum_tp_vol"] = (typical * vol).cumsum()
        df["cum_vol"] = vol.cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
    else:
        df["Volume"] = 0
        df["avg_volume"] = 0
        df["vwap"] = df["Close"].rolling(20).mean()

    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df["tr"].rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()

    # Session tracking (rolling high/low per 10h blocks)
    df["session_high"] = df["High"].rolling(120, min_periods=10).max()  # ~10h of 5m
    df["session_low"] = df["Low"].rolling(120, min_periods=10).min()
    # Previous session = shifted
    df["prev_session_high"] = df["session_high"].shift(120)
    df["prev_session_low"] = df["session_low"].shift(120)

    return df


def detect_swings(df, lookback=3):
    sh, sl = [], []
    for i in range(lookback, len(df)-lookback):
        h, l = df["High"].iloc[i], df["Low"].iloc[i]
        if all(h >= df["High"].iloc[i+j] and h >= df["High"].iloc[i-j] for j in range(1, lookback+1)):
            sh.append(i)
        if all(l <= df["Low"].iloc[i+j] and l <= df["Low"].iloc[i-j] for j in range(1, lookback+1)):
            sl.append(i)
    return sh, sl


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION — all 15 signals toggleable
# ═══════════════════════════════════════════════════════════════════

def detect_signal(df, i, cfg, swing_highs, swing_lows):
    if i < 60:
        return None
    price = df["Close"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or atr <= 0:
        return None
    ema9, ema21, ema50 = df["ema9"].iloc[i], df["ema21"].iloc[i], df["ema50"].iloc[i]
    rsi = df["rsi"].iloc[i]
    if pd.isna(ema9) or pd.isna(ema21):
        return None
    ts = df.index[i]
    if not hasattr(ts, 'hour') or not (7 <= ts.hour < 17):
        return None

    confluence = 0
    reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}

    ch, cl, cc = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
    co = df["Open"].iloc[i]
    body = df["body"].iloc[i]
    total = df["candle_range"].iloc[i]

    # 1. EMA 9/21
    if cfg.SIG_EMA:
        if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
        else: votes[Direction.SHORT] += 1; reasons.append("ema")

    # 2. EMA 50
    if cfg.SIG_EMA50 and not pd.isna(ema50):
        if price > ema50 and ema9 > ema50: votes[Direction.LONG] += 1; reasons.append("ema50")
        elif price < ema50 and ema9 < ema50: votes[Direction.SHORT] += 1; reasons.append("ema50")

    # 3. Liquidity sweep
    if cfg.SIG_SWEEP:
        for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
            if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
                wick = min(cc, co) - cl
                if wick > atr*0.25: votes[Direction.LONG] += 2; reasons.append("sweep"); break
        for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
            if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
                wick = ch - max(cc, co)
                if wick > atr*0.25: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    # 4. Order block
    if cfg.SIG_OB and i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # 5. FVG
    if cfg.SIG_FVG and i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*0.25
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    # 6. Momentum
    if cfg.SIG_MOMENTUM and total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    # 7. Exhaustion
    if cfg.SIG_EXHAUSTION and total > 0:
        if df["upper_wick"].iloc[i]/total >= cfg.EXHAUSTION_WICK_RATIO: votes[Direction.SHORT] += 1; reasons.append("exh")
        elif df["lower_wick"].iloc[i]/total >= cfg.EXHAUSTION_WICK_RATIO: votes[Direction.LONG] += 1; reasons.append("exh")

    # 8. Wick rejection
    if cfg.SIG_WICK and total > 0:
        if df["lower_wick"].iloc[i]/total > 0.5 and cc > co: votes[Direction.LONG] += 1; reasons.append("wick")
        elif df["upper_wick"].iloc[i]/total > 0.5 and cc < co: votes[Direction.SHORT] += 1; reasons.append("wick")

    # 9. Round number
    if cfg.SIG_ROUND_NUM:
        nearest = round(price/cfg.ROUND_NUMBER_INTERVAL)*cfg.ROUND_NUMBER_INTERVAL
        if abs(price-nearest) <= cfg.ROUND_NUMBER_ZONE: confluence += 1; reasons.append("rn")

    # 10. Mean Reversion
    if cfg.SIG_MR:
        bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
        if not (pd.isna(bb_u) or pd.isna(rsi)):
            bbr = bb_u - bb_l
            if bbr > 0:
                pct_b = (price-bb_l)/bbr
                if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
                elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    # 11. RSI divergence
    if cfg.SIG_RSI_DIV:
        rsi14 = df["rsi14"].iloc[i]
        if not pd.isna(rsi14):
            if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
            elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # 12. VWAP (v1.3)
    if cfg.SIG_VWAP:
        vwap = df["vwap"].iloc[i]
        if not pd.isna(vwap) and vwap > 0:
            if price > vwap * 1.001: votes[Direction.LONG] += 1; reasons.append("vwap")
            elif price < vwap * 0.999: votes[Direction.SHORT] += 1; reasons.append("vwap")

    # 13. Previous session high/low (v1.3)
    if cfg.SIG_SESSION_LVL:
        psh = df["prev_session_high"].iloc[i]
        psl = df["prev_session_low"].iloc[i]
        if not (pd.isna(psh) or pd.isna(psl)) and psh > psl:
            buf = (psh-psl) * 0.05
            if ch >= psh-buf and cc < psh:
                votes[Direction.SHORT] += 2; reasons.append("sess_lvl")
            elif cl <= psl+buf and cc > psl:
                votes[Direction.LONG] += 2; reasons.append("sess_lvl")

    # 14. Double bottom/top (v1.3)
    if cfg.SIG_DOUBLE:
        recent_sh = [s for s in swing_highs if s < i and s > i-30]
        recent_sl = [s for s in swing_lows if s < i and s > i-30]
        tol = atr * 0.3
        if len(recent_sl) >= 2:
            l1, l2 = df["Low"].iloc[recent_sl[-2]], df["Low"].iloc[recent_sl[-1]]
            if abs(l1-l2) < tol and price > max(l1,l2): votes[Direction.LONG] += 1; reasons.append("dbl")
        if len(recent_sh) >= 2:
            h1, h2 = df["High"].iloc[recent_sh[-2]], df["High"].iloc[recent_sh[-1]]
            if abs(h1-h2) < tol and price < min(h1,h2): votes[Direction.SHORT] += 1; reasons.append("dbl")

    # 15. ADX (v1.3)
    if cfg.SIG_ADX:
        adx = df["adx"].iloc[i]
        if not pd.isna(adx):
            if cfg.ADX_BLOCK_BELOW > 0 and adx < cfg.ADX_BLOCK_BELOW:
                return None  # block ranging market
            if adx >= cfg.ADX_THRESHOLD:
                confluence += 1; reasons.append(f"adx{adx:.0f}")

    # Direction
    ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
    if ls > ss and ls >= 1: direction = Direction.LONG; confluence += ls
    elif ss > ls and ss >= 1: direction = Direction.SHORT; confluence += ss
    else: return None

    # EMA filter
    if cfg.SIG_EMA:
        if direction == Direction.LONG and ema9 <= ema21 and "MR" not in reasons: return None
        if direction == Direction.SHORT and ema9 >= ema21 and "MR" not in reasons: return None

    if confluence < cfg.MIN_CONFLUENCE: return None
    return direction, confluence, "|".join(reasons)


# ═══════════════════════════════════════════════════════════════════
#  FAST BACKTESTER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    direction: Direction; entry: float; sl: float; tp: float; tp1: float
    lots: float; bar: int; phase: TradePhase = TradePhase.OPEN
    pnl: float = 0.0; remaining: float = 0.0
    def __post_init__(self): self.remaining = self.lots

def run_backtest(df, cfg, sh, sl):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999

    for i in range(60, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date: daily_date = today; daily_trades = 0

        for t in list(active):
            sl_hit = (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl)
            if sl_hit:
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses += 1; llb = i; active.remove(t); closed.append(t); continue
            tp_hit = (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp)
            if tp_hit:
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses = 0; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                tp1_hit = (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1)
                if tp1_hit:
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*100
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p
                        t.remaining = round(t.remaining-cl, 2); t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

        eq = balance + sum(((price-t.entry) if t.direction==Direction.LONG else (t.entry-price))*t.remaining*100 for t in active)
        if eq > peak: peak = eq
        dd = (peak-eq)/peak*100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        if daily_trades >= cfg.MAX_DAILY_TRADES: continue
        if len(active) >= cfg.MAX_CONCURRENT_TRADES: continue
        if consec_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            if i - llb < cfg.LOSS_COOLDOWN_BARS*2: continue
            consec_losses = 0
        if i - ltb < cfg.TRADE_COOLDOWN_BARS: continue
        if i - llb < cfg.LOSS_COOLDOWN_BARS: continue
        if (cfg.START_BALANCE-balance)/cfg.START_BALANCE*100 >= cfg.MAX_TOTAL_DRAWDOWN_PERCENT: continue

        signal = detect_signal(df, i, cfg, sh, sl)
        if not signal: continue
        direction, score, reason = signal

        sl_dist = max(atr*cfg.ATR_SL_MULTIPLIER, cfg.MIN_SL_POINTS)
        sl_dist = min(sl_dist, cfg.MAX_SL_POINTS)
        entry = price + cfg.SIMULATED_SPREAD if direction==Direction.LONG else price
        if direction == Direction.LONG:
            s,t,t1 = entry-sl_dist, entry+sl_dist*cfg.RR_RATIO, entry+sl_dist*cfg.TP1_RR
        else:
            s,t,t1 = entry+sl_dist, entry-sl_dist*cfg.RR_RATIO, entry-sl_dist*cfg.TP1_RR

        lots = max(0.01, min(round((balance*cfg.RISK_PERCENT/100)/(sl_dist*100), 2), 0.5))
        active.append(Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1, lots=lots, bar=i))
        daily_trades += 1; ltb = i

    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            closed.append(t)

    if not closed: return {"trades":0,"pf":0,"wr":0,"pnl":0,"dd":0,"balance":balance}
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    return {
        "trades":len(closed),"wins":len(wins),"losses":len(losses),
        "wr":round(len(wins)/len(closed)*100,1),"pf":round(tw/tl,2) if tl>0 else 99,
        "pnl":round(balance-cfg.START_BALANCE,2),"return_pct":round((balance-cfg.START_BALANCE)/cfg.START_BALANCE*100,2),
        "dd":round(max_dd,2),"balance":round(balance,2),
        "avg_win":round(tw/len(wins),2) if wins else 0,"avg_loss":round(tl/len(losses),2) if losses else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL NAMES
# ═══════════════════════════════════════════════════════════════════

SIGNAL_NAMES = {
    "SIG_EMA": "EMA 9/21 Trend",
    "SIG_EMA50": "EMA 50 Filter",
    "SIG_SWEEP": "Liquidity Sweep",
    "SIG_OB": "Order Block",
    "SIG_FVG": "Fair Value Gap",
    "SIG_MOMENTUM": "Momentum Candle",
    "SIG_EXHAUSTION": "Exhaustion Candle",
    "SIG_WICK": "Wick Rejection",
    "SIG_ROUND_NUM": "Round Number",
    "SIG_MR": "Mean Reversion (BB+RSI)",
    "SIG_RSI_DIV": "RSI Divergence",
    "SIG_VWAP": "VWAP Direction",
    "SIG_SESSION_LVL": "Session High/Low",
    "SIG_DOUBLE": "Double Bottom/Top",
    "SIG_ADX": "ADX Trend Strength",
}


# ═══════════════════════════════════════════════════════════════════
#  OPTIMIZER v4 — SIGNAL AUDIT
# ═══════════════════════════════════════════════════════════════════

def optimize(df):
    print("\n" + "="*60)
    print("  🔧 OPTIMIZER v4.0 — SIGNAL AUDIT")
    print("  Welke signalen helpen? Welke moeten eruit?")
    print("="*60)

    df = calculate_indicators(df)
    sh, sl = detect_swings(df, 3)

    # ─── PHASE 1: BASELINE (all signals ON) ───────────────────
    print("\n  📊 Phase 1: Baseline (alle signalen AAN)")
    baseline_cfg = Config()
    baseline = run_backtest(df, baseline_cfg, sh, sl)
    print(f"  ✅ Baseline: {baseline['trades']}t | WR:{baseline['wr']}% | PF:{baseline['pf']} | ${baseline['pnl']:+.0f}")

    # ─── PHASE 2: TEST EACH SIGNAL INDIVIDUALLY ──────────────
    print(f"\n  📊 Phase 2: Elk signaal UIT — wat verandert er?")
    print(f"  {'Signal':<25} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL':>10} {'Impact':>10}")
    print(f"  {'-'*65}")

    signal_impact = {}
    for sig_key, sig_name in SIGNAL_NAMES.items():
        cfg = Config()
        setattr(cfg, sig_key, False)  # turn OFF this signal
        r = run_backtest(df, cfg, sh, sl)

        # Impact = verschil met baseline
        wr_diff = r["wr"] - baseline["wr"]
        pnl_diff = r["pnl"] - baseline["pnl"]
        pf_diff = r["pf"] - baseline["pf"]

        signal_impact[sig_key] = {
            "name": sig_name,
            "without": r,
            "wr_diff": wr_diff,
            "pnl_diff": pnl_diff,
            "pf_diff": pf_diff,
        }

        # If turning OFF improves results → signal hurts
        emoji = "🟢" if pnl_diff < 0 else "🔴" if pnl_diff > 50 else "⚪"
        print(f"  {emoji} {sig_name:<23} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} ${r['pnl']:>+9.0f} ${pnl_diff:>+9.0f}")

    # ─── PHASE 3: IDENTIFY HELPERS AND HURTERS ────────────────
    print(f"\n  📊 Phase 3: Signaal classificatie")

    helpers = []  # turning off makes it WORSE → keep
    hurters = []  # turning off makes it BETTER → remove
    neutral = []  # minimal impact

    for sig_key, data in signal_impact.items():
        if data["pnl_diff"] < -100:  # removing costs >$100 → keep it
            helpers.append((sig_key, data))
        elif data["pnl_diff"] > 100:  # removing gains >$100 → remove it
            hurters.append((sig_key, data))
        else:
            neutral.append((sig_key, data))

    helpers.sort(key=lambda x: x[1]["pnl_diff"])  # most impactful first
    hurters.sort(key=lambda x: x[1]["pnl_diff"], reverse=True)

    print(f"\n  ✅ HOUDEN (verwijderen maakt het slechter):")
    for sig_key, data in helpers:
        print(f"    🟢 {data['name']:<25} — verwijderen kost ${abs(data['pnl_diff']):.0f}")

    print(f"\n  ❌ VERWIJDEREN (verwijderen maakt het beter):")
    if hurters:
        for sig_key, data in hurters:
            print(f"    🔴 {data['name']:<25} — verwijderen wint ${data['pnl_diff']:.0f}")
    else:
        print(f"    Geen signalen gevonden die schaden!")

    print(f"\n  ⚪ NEUTRAAL (minimale impact):")
    for sig_key, data in neutral:
        print(f"    ⚪ {data['name']:<25} — ${data['pnl_diff']:+.0f}")

    # ─── PHASE 4: TEST BEST COMBO (only helpers) ─────────────
    print(f"\n  📊 Phase 4: Beste combo (alleen helpers)")

    best_cfg = Config()
    for sig_key, data in hurters:
        setattr(best_cfg, sig_key, False)  # turn off hurters

    best_result = run_backtest(df, best_cfg, sh, sl)
    print(f"  ✅ Only helpers: {best_result['trades']}t | WR:{best_result['wr']}% | PF:{best_result['pf']} | ${best_result['pnl']:+.0f}")

    # ─── PHASE 5: FINE-TUNE BEST COMBO ───────────────────────
    print(f"\n  📊 Phase 5: Fine-tune met core params")

    all_results = []
    count = 0

    for conf in [3, 4, 5]:
        for rr in [2.0, 2.5, 3.0]:
            for sl_m in [1.0, 1.5, 2.0]:
                for tp1 in [0.6, 0.8, 1.0]:
                    for cd_t in [3, 5, 8, 12]:
                        for cd_l in [5, 9, 15]:
                            count += 1
                            cfg = Config()
                            # Apply hurter removals
                            for sig_key, data in hurters:
                                setattr(cfg, sig_key, False)
                            cfg.MIN_CONFLUENCE = conf
                            cfg.RR_RATIO = rr
                            cfg.ATR_SL_MULTIPLIER = sl_m
                            cfg.TP1_RR = tp1
                            cfg.TRADE_COOLDOWN_BARS = cd_t
                            cfg.LOSS_COOLDOWN_BARS = cd_l

                            r = run_backtest(df, cfg, sh, sl)
                            r["params"] = {"conf":conf,"rr":rr,"sl":sl_m,"tp1":tp1,"cd_t":cd_t,"cd_l":cd_l}
                            all_results.append(r)

                            if count % 100 == 0:
                                print(f"    {count} tested...")

    print(f"  ✅ Phase 5: {count} combinations")

    viable = [r for r in all_results if r["trades"] >= 15 and r["pf"] >= 1.0]
    for r in viable:
        wr_bonus = max(0, r["wr"]-50)*0.5
        r["score"] = round((r["wr"]/100)*r["pf"]*math.sqrt(r["trades"])*(1-r["dd"]/100)+wr_bonus, 2)

    viable.sort(key=lambda x: x["score"], reverse=True)

    # ─── FINAL RESULTS ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🏆 TOP 10 BESTE SETTINGS (na signal audit)")
    print(f"{'='*60}")

    for i, r in enumerate(viable[:10]):
        p = r["params"]
        print(f"""
  #{i+1} — Score: {r['score']}
  ├── SL: ATR×{p['sl']} | Conf≥{p['conf']} | RR: 1:{p['rr']}
  ├── TP1: {p['tp1']}R | CD: {p['cd_t']}/{p['cd_l']}
  ├── Trades: {r['trades']} | WR: {r['wr']}% | PF: {r['pf']}
  ├── PnL: ${r['pnl']:+,.2f} | Return: {r['return_pct']:+.2f}%
  └── Max DD: {r['dd']:.1f}%""")

    # ─── RECOMMENDATION ──────────────────────────────────────
    if viable:
        best = viable[0]; bp = best["params"]
        print(f"""
{'='*60}
  ⭐ AANBEVOLEN LIVE SETTINGS:
{'='*60}

  SIGNALEN:""")
        for sig_key, sig_name in SIGNAL_NAMES.items():
            is_on = sig_key not in [h[0] for h in hurters]
            emoji = "✅" if is_on else "❌"
            print(f"    {emoji} {sig_name}")

        print(f"""
  PARAMETERS:
  ATR_SL_MULTIPLIER = {bp['sl']}
  RR_RATIO = {bp['rr']}
  TP1_RR = {bp['tp1']}
  MIN_CONFLUENCE = {bp['conf']}
  TRADE_COOLDOWN = {bp['cd_t']}
  LOSS_COOLDOWN = {bp['cd_l']}

  VERWACHT:
  WR: {best['wr']}% | PF: {best['pf']} | Trades: {best['trades']}
  DD: {best['dd']:.1f}% | Return: {best['return_pct']:+.2f}%

  VS BASELINE:
  WR: {baseline['wr']}% → {best['wr']}% ({best['wr']-baseline['wr']:+.1f}%)
  PF: {baseline['pf']} → {best['pf']} ({best['pf']-baseline['pf']:+.2f})
  PnL: ${baseline['pnl']:+.0f} → ${best['pnl']:+.0f}
  Trades: {baseline['trades']} → {best['trades']}
{'='*60}""")

    # Save
    output = {
        "baseline": {"wr":baseline["wr"],"pf":baseline["pf"],"pnl":baseline["pnl"],"trades":baseline["trades"]},
        "signal_audit": {k: {"name":v["name"],"pnl_impact":v["pnl_diff"],"wr_impact":v["wr_diff"],"verdict":"KEEP" if v["pnl_diff"]<-100 else "REMOVE" if v["pnl_diff"]>100 else "NEUTRAL"} for k,v in signal_impact.items()},
        "top_10": [{"rank":i+1,"params":r["params"],"wr":r["wr"],"pf":r["pf"],"pnl":r["pnl"],"trades":r["trades"],"dd":r["dd"]} for i,r in enumerate(viable[:10])],
        "total_tested": count,
    }
    with open("optimization_v4_results.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  📁 Saved: optimization_v4_results.json")
    return viable[:10]


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — OPTIMIZER v4.0                    ║
║     🔍 Signal Audit: wat helpt, wat moet eruit?             ║
║     🎯 + Fine-tune voor maximale winst                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print("📥 Downloading 60-day gold data...")
    df = yf.download("GC=F", period="60d", interval="5m", progress=True)
    if df.empty: df = yf.download("GC=F", period="60d", interval="1h", progress=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df[df["Close"]>0]
    print(f"✅ {len(df)} bars: {df.index[0]} → {df.index[-1]}")
    start = time.time()
    optimize(df)
    print(f"\n  ⏱️  Klaar in {time.time()-start:.0f}s")

if __name__ == "__main__":
    main()
