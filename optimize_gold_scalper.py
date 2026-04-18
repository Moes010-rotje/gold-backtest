“””
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — FULL WEEKLY ANALYZER + OPTIMIZER  ║
║  Phase 1: Wekelijkse breakdown                               ║
║  Phase 2: 30+ verbeteringen testen                           ║
║  Categories: Indicatoren, Risk Mgmt, Strategie, Sessions     ║
╚══════════════════════════════════════════════════════════════╝
“””

import sys, json, time, math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

try:
import pandas as pd
import numpy as np
except ImportError:
print(“pip install pandas numpy”); sys.exit(1)
try:
import yfinance as yf
except ImportError:
print(“pip install yfinance”); sys.exit(1)

class Direction(Enum):
LONG = “buy”
SHORT = “sell”

class TradePhase(Enum):
OPEN = “open”
TP1_HIT = “tp1_hit”

@dataclass
class Config:
LABEL: str = “v1.5.1 BASE”
START_BALANCE: float = 5000.0
COMMISSION_PER_LOT: float = 7.0
SIMULATED_SPREAD: float = 0.30
RISK_PERCENT: float = 1.0
MAX_DAILY_LOSS_PERCENT: float = 3.0
MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
MAX_CONCURRENT_TRADES: int = 3
MAX_DAILY_TRADES: int = 30
MAX_CONSECUTIVE_LOSSES: int = 5

```
ATR_SL_MULTIPLIER: float = 2.5
MIN_SL_POINTS: float = 2.0
MAX_SL_POINTS: float = 10.0
RR_RATIO: float = 2.0
PARTIAL_PERCENT: float = 0.67
TP1_RR: float = 0.4
MOVE_SL_TO_BE: bool = True
SWING_LOOKBACK: int = 3
ENGULF_BODY_RATIO: float = 0.60
MOMENTUM_CANDLE_ATR: float = 0.8
RSI_OVERSOLD: float = 25.0
RSI_OVERBOUGHT: float = 75.0
MR_CONFLUENCE_SCORE: int = 2
MIN_CONFLUENCE: int = 3
TRADE_COOLDOWN_BARS: int = 5
LOSS_COOLDOWN_BARS: int = 12
ADX_THRESHOLD: float = 25.0
SWEEP_WICK_ATR_MIN: float = 0.25
FVG_MIN_ATR: float = 0.25
DOUBLE_TOLERANCE_ATR: float = 0.3

# ─── EXTRA TOGGLES ───────────────────────────────────────
# Indicators
USE_MACD: bool = False
USE_STOCH: bool = False
USE_EMA50: bool = False
USE_HTF_TREND: bool = False
USE_VWAP: bool = False
USE_RSI_FILTER: bool = False       # block overbought buy / oversold sell
USE_BB_SQUEEZE: bool = False       # only trade when BB expanding

# Risk management
USE_REGIME_FILTER: bool = False    # skip low volatility
REGIME_LOW_VOL: float = 0.6
USE_LOSS_PAUSE: bool = False       # pause after consecutive losses
LOSS_PAUSE_BARS: int = 60
USE_DAILY_TARGET: bool = False     # stop trading after hitting daily target
DAILY_TARGET_USD: float = 200.0
USE_SCALING: bool = False          # scale lots based on confidence
SCALE_HIGH_CONF: float = 1.5      # multiply lots if high confluence

# Strategy
USE_TREND_ONLY: bool = False       # only trade with EMA21 trend
USE_COUNTER_TREND_BLOCK: bool = False  # block trades against 1H trend
TP2_ENABLED: bool = False          # second partial at 1.0R
TP2_RR: float = 1.0
TP2_PERCENT: float = 0.50          # close 50% of remaining at TP2

# Session
SKIP_HOURS: list = None            # hours to skip
LONDON_ONLY: bool = False          # only London session
NY_ONLY: bool = False              # only NY session

def __post_init__(self):
    if self.SKIP_HOURS is None:
        self.SKIP_HOURS = []
```

def calculate_indicators(df):
df = df.copy()
df[“ema9”] = df[“Close”].ewm(span=9).mean()
df[“ema21”] = df[“Close”].ewm(span=21).mean()
df[“ema50”] = df[“Close”].ewm(span=50).mean()
df[“tr”] = np.maximum(df[“High”]-df[“Low”], np.maximum(abs(df[“High”]-df[“Close”].shift(1)), abs(df[“Low”]-df[“Close”].shift(1))))
df[“atr”] = df[“tr”].rolling(14).mean()
df[“atr50”] = df[“tr”].rolling(50).mean()

```
d = df["Close"].diff()
g = d.clip(lower=0).rolling(7).mean()
l = (-d.clip(upper=0)).rolling(7).mean()
rs = g / l.replace(0, np.nan)
df["rsi"] = 100 - (100/(1+rs))

d14 = df["Close"].diff()
g14 = d14.clip(lower=0).rolling(14).mean()
l14 = (-d14.clip(upper=0)).rolling(14).mean()
rs14 = g14 / l14.replace(0, np.nan)
df["rsi14"] = 100 - (100/(1+rs14))

df["bb_mid"] = df["Close"].rolling(20).mean()
df["bb_std"] = df["Close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

df["body"] = abs(df["Close"]-df["Open"])
df["candle_range"] = df["High"]-df["Low"]
df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]

# ADX
plus_dm = df["High"].diff(); minus_dm = -df["Low"].diff()
plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
atr14 = df["tr"].rolling(14).mean()
plus_di = 100*(plus_dm.rolling(14).mean()/atr14.replace(0,np.nan))
minus_di = 100*(minus_dm.rolling(14).mean()/atr14.replace(0,np.nan))
dx = 100*abs(plus_di-minus_di)/(plus_di+minus_di).replace(0,np.nan)
df["adx"] = dx.rolling(14).mean()

# MACD
ema12 = df["Close"].ewm(span=12).mean(); ema26 = df["Close"].ewm(span=26).mean()
df["macd"] = ema12 - ema26; df["macd_sig"] = df["macd"].ewm(span=9).mean()
df["macd_hist"] = df["macd"] - df["macd_sig"]

# Stochastic
low14 = df["Low"].rolling(14).min(); high14 = df["High"].rolling(14).max()
df["stoch_k"] = ((df["Close"]-low14)/(high14-low14).replace(0,np.nan))*100
df["stoch_d"] = df["stoch_k"].rolling(3).mean()

# VWAP
if "Volume" in df.columns and df["Volume"].sum() > 0:
    tp = (df["High"]+df["Low"]+df["Close"])/3
    vol = df["Volume"].replace(0,1)
    df["vwap"] = (tp*vol).cumsum()/vol.cumsum()
else:
    df["vwap"] = df["Close"].rolling(50).mean()

# EMA21 slope (trend direction)
df["ema21_slope"] = df["ema21"] - df["ema21"].shift(20)

return df
```

def detect_swings(df, lookback=3):
sh, sl = [], []
for i in range(lookback, len(df)-lookback):
h, l = df[“High”].iloc[i], df[“Low”].iloc[i]
if all(h >= df[“High”].iloc[i+j] and h >= df[“High”].iloc[i-j] for j in range(1,lookback+1)): sh.append(i)
if all(l <= df[“Low”].iloc[i+j] and l <= df[“Low”].iloc[i-j] for j in range(1,lookback+1)): sl.append(i)
return sh, sl

def detect_signal(df, i, cfg, swing_highs, swing_lows):
if i < 60: return None
price = df[“Close”].iloc[i]
atr = df[“atr”].iloc[i]
if pd.isna(atr) or atr <= 0: return None
ema9, ema21 = df[“ema9”].iloc[i], df[“ema21”].iloc[i]
rsi = df[“rsi”].iloc[i]
if pd.isna(ema9) or pd.isna(ema21): return None
ts = df.index[i]
if not hasattr(ts,‘hour’): return None
hour = ts.hour

```
# Session filters
if cfg.LONDON_ONLY and not (7 <= hour < 12): return None
if cfg.NY_ONLY and not (12 <= hour < 17): return None
if not (cfg.LONDON_ONLY or cfg.NY_ONLY) and not (7 <= hour < 17): return None
if hour in cfg.SKIP_HOURS: return None

# Regime filter
if cfg.USE_REGIME_FILTER:
    atr50 = df["atr50"].iloc[i]
    if not pd.isna(atr50) and atr50 > 0 and atr < atr50*cfg.REGIME_LOW_VOL: return None

# BB squeeze filter
if cfg.USE_BB_SQUEEZE:
    bw = df["bb_width"].iloc[i]; bw_prev = df["bb_width"].iloc[i-5] if i >= 5 else bw
    if not pd.isna(bw) and not pd.isna(bw_prev) and bw < bw_prev: return None

confluence = 0; reasons = []
votes = {Direction.LONG: 0, Direction.SHORT: 0}
ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

# 1. EMA
if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
else: votes[Direction.SHORT] += 1; reasons.append("ema")

# 2. Sweep
for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
    if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
        wick = min(cc,co)-cl
        if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.LONG] += 2; reasons.append("sweep"); break
for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
    if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
        wick = ch-max(cc,co)
        if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

# 3. OB
if i >= 2:
    prev,curr = df.iloc[i-1],df.iloc[i]
    pb,cb = abs(prev["Close"]-prev["Open"]),abs(curr["Close"]-curr["Open"])
    if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
        votes[Direction.LONG] += 1; reasons.append("ob")
    elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
        votes[Direction.SHORT] += 1; reasons.append("ob")

# 4. FVG
if i >= 2:
    c1h,c3l = df["High"].iloc[i-2],df["Low"].iloc[i]
    c1l,c3h = df["Low"].iloc[i-2],df["High"].iloc[i]
    mg = atr*cfg.FVG_MIN_ATR
    if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
    elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

# 5. Momentum
if total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
    if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
    else: votes[Direction.SHORT] += 1; reasons.append("mom")

# 6. Mean Reversion
bb_u,bb_l = df["bb_upper"].iloc[i],df["bb_lower"].iloc[i]
if not (pd.isna(bb_u) or pd.isna(rsi)):
    bbr = bb_u-bb_l
    if bbr > 0:
        pct_b = (price-bb_l)/bbr
        if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
        elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

# 7. RSI div
rsi14 = df["rsi14"].iloc[i]
if not pd.isna(rsi14):
    if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
    elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

# 8. Double
rsl = [s for s in swing_lows if s < i and s > i-30]
rsh = [s for s in swing_highs if s < i and s > i-30]
tol = atr*cfg.DOUBLE_TOLERANCE_ATR
if len(rsl) >= 2:
    l1,l2 = df["Low"].iloc[rsl[-2]],df["Low"].iloc[rsl[-1]]
    if abs(l1-l2) < tol and price > max(l1,l2): votes[Direction.LONG] += 1; reasons.append("dbl")
if len(rsh) >= 2:
    h1,h2 = df["High"].iloc[rsh[-2]],df["High"].iloc[rsh[-1]]
    if abs(h1-h2) < tol and price < min(h1,h2): votes[Direction.SHORT] += 1; reasons.append("dbl")

# 9. ADX
adx = df["adx"].iloc[i]
if not pd.isna(adx) and adx >= cfg.ADX_THRESHOLD: confluence += 1; reasons.append("adx")

# ─── EXTRA INDICATORS ─────────────────────────────────────

# 10. MACD
if cfg.USE_MACD:
    mh = df["macd_hist"].iloc[i]; mh_p = df["macd_hist"].iloc[i-1]
    if not pd.isna(mh) and not pd.isna(mh_p):
        if mh > 0 and mh > mh_p: votes[Direction.LONG] += 1; reasons.append("macd")
        elif mh < 0 and mh < mh_p: votes[Direction.SHORT] += 1; reasons.append("macd")

# 11. Stochastic
if cfg.USE_STOCH:
    sk,sd = df["stoch_k"].iloc[i],df["stoch_d"].iloc[i]
    if not pd.isna(sk) and not pd.isna(sd):
        if sk < 20 and sk > sd: votes[Direction.LONG] += 1; reasons.append("stoch")
        elif sk > 80 and sk < sd: votes[Direction.SHORT] += 1; reasons.append("stoch")

# 12. EMA50
if cfg.USE_EMA50:
    ema50 = df["ema50"].iloc[i]
    if not pd.isna(ema50):
        if price > ema50 and ema9 > ema50: votes[Direction.LONG] += 1; reasons.append("ema50")
        elif price < ema50 and ema9 < ema50: votes[Direction.SHORT] += 1; reasons.append("ema50")

# 13. HTF trend (EMA21 slope)
if cfg.USE_HTF_TREND:
    slope = df["ema21_slope"].iloc[i]
    if not pd.isna(slope):
        if slope > 0: votes[Direction.LONG] += 1; reasons.append("htf")
        elif slope < 0: votes[Direction.SHORT] += 1; reasons.append("htf")

# 14. VWAP
if cfg.USE_VWAP:
    vwap = df["vwap"].iloc[i]
    if not pd.isna(vwap) and vwap > 0:
        if price > vwap*1.001: votes[Direction.LONG] += 1; reasons.append("vwap")
        elif price < vwap*0.999: votes[Direction.SHORT] += 1; reasons.append("vwap")

# Direction
ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
if ls > ss and ls >= 1: direction = Direction.LONG; confluence += ls
elif ss > ls and ss >= 1: direction = Direction.SHORT; confluence += ss
else: return None

# EMA filter
if direction == Direction.LONG and ema9 <= ema21 and "MR" not in reasons: return None
if direction == Direction.SHORT and ema9 >= ema21 and "MR" not in reasons: return None

# RSI overextension filter
if cfg.USE_RSI_FILTER and not pd.isna(rsi):
    if direction == Direction.LONG and rsi > 70: return None
    if direction == Direction.SHORT and rsi < 30: return None

# Trend-only filter
if cfg.USE_TREND_ONLY:
    slope = df["ema21_slope"].iloc[i]
    if not pd.isna(slope):
        if direction == Direction.LONG and slope < 0: return None
        if direction == Direction.SHORT and slope > 0: return None

if confluence < cfg.MIN_CONFLUENCE: return None
return direction, confluence, "|".join(reasons)
```

@dataclass
class Trade:
direction: Direction; entry: float; sl: float; tp: float; tp1: float
lots: float; bar: int; sl_dist: float; day: str = “”; hour: int = 0
phase: TradePhase = TradePhase.OPEN; pnl: float = 0.0; remaining: float = 0.0
reasons: str = “”; confluence: int = 0
def **post_init**(self): self.remaining = self.lots

def run_backtest(df, cfg, sh, sl):
balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
active, closed = [], []; daily_trades = 0; daily_date = “”
consec_losses = 0; ltb = -999; llb = -999
loss_pause_until = -999; daily_pnl_today = 0.0

```
for i in range(60, len(df)):
    price,high,low = df["Close"].iloc[i],df["High"].iloc[i],df["Low"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or price <= 0: continue
    ts = df.index[i]
    today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
    if today != daily_date:
        daily_date = today; daily_trades = 0; daily_pnl_today = 0.0

    for t in list(active):
        if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
            t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
            t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl; daily_pnl_today += t.pnl
            consec_losses += 1; llb = i; active.remove(t); closed.append(t)
            if cfg.USE_LOSS_PAUSE and consec_losses >= 3: loss_pause_until = i + cfg.LOSS_PAUSE_BARS
            continue
        if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
            t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
            t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl; daily_pnl_today += t.pnl
            consec_losses = 0; active.remove(t); closed.append(t); continue
        if t.phase == TradePhase.OPEN:
            if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                if cl >= 0.01:
                    p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*100
                    p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p; daily_pnl_today += p
                    t.remaining = round(t.remaining-cl,2); t.phase = TradePhase.TP1_HIT
                    if cfg.MOVE_SL_TO_BE: t.sl = t.entry
        # TP2 (second partial)
        if cfg.TP2_ENABLED and t.phase == TradePhase.TP1_HIT:
            tp2_price = t.entry + t.sl_dist*cfg.TP2_RR if t.direction==Direction.LONG else t.entry - t.sl_dist*cfg.TP2_RR
            if (t.direction==Direction.LONG and high>=tp2_price) or (t.direction==Direction.SHORT and low<=tp2_price):
                cl2 = round(t.remaining*cfg.TP2_PERCENT,2)
                if cl2 >= 0.01:
                    p2 = ((tp2_price-t.entry) if t.direction==Direction.LONG else (t.entry-tp2_price))*cl2*100
                    p2 -= cfg.COMMISSION_PER_LOT*cl2; balance += p2; t.pnl += p2; daily_pnl_today += p2
                    t.remaining = round(t.remaining-cl2,2)

    eq = balance + sum(((price-t.entry) if t.direction==Direction.LONG else (t.entry-price))*t.remaining*100 for t in active)
    if eq > peak: peak = eq
    dd = (peak-eq)/peak*100 if peak > 0 else 0
    if dd > max_dd: max_dd = dd

    if daily_trades >= cfg.MAX_DAILY_TRADES: continue
    if len(active) >= cfg.MAX_CONCURRENT_TRADES: continue
    if consec_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
        if i-llb < cfg.LOSS_COOLDOWN_BARS*2: continue
        consec_losses = 0
    if i-ltb < cfg.TRADE_COOLDOWN_BARS: continue
    if i-llb < cfg.LOSS_COOLDOWN_BARS: continue
    if cfg.USE_LOSS_PAUSE and i < loss_pause_until: continue
    if cfg.USE_DAILY_TARGET and daily_pnl_today >= cfg.DAILY_TARGET_USD: continue
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

    lots = max(0.01, min(round((balance*cfg.RISK_PERCENT/100)/(sl_dist*100),2), 0.5))
    if cfg.USE_SCALING and score >= 5: lots = min(round(lots*cfg.SCALE_HIGH_CONF,2), 0.5)
    trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1,
                  lots=lots, bar=i, sl_dist=sl_dist, day=today,
                  hour=ts.hour if hasattr(ts,'hour') else 0, reasons=reason, confluence=score)
    active.append(trade); daily_trades += 1; ltb = i

if active:
    lp = df["Close"].iloc[-1]
    for t in active:
        t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
        closed.append(t)

return closed, balance, max_dd
```

def calc_stats(closed, cfg):
if not closed: return None
wins = [t for t in closed if t.pnl > 0]; losses = [t for t in closed if t.pnl <= 0]
tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
# Weekly
weekly_pnl = defaultdict(float)
daily_pnl = defaultdict(float)
for t in closed:
weekly_pnl[t.day[:10]] += t.pnl  # daily first
daily_pnl[t.day] += t.pnl
# Group into weeks
week_totals = defaultdict(float)
for d, p in daily_pnl.items():
try:
dt = datetime.strptime(d, “%Y-%m-%d”)
yr,wk,_ = dt.isocalendar()
week_totals[f”{yr}-W{wk:02d}”] += p
except: pass
green_w = sum(1 for p in week_totals.values() if p >= 0)
red_w = sum(1 for p in week_totals.values() if p < 0)
trading_days = len([p for p in daily_pnl.values() if abs(p) > 0.01])
avg_daily = sum(daily_pnl.values())/max(trading_days,1)

```
return {
    "label": cfg.LABEL, "trades": len(closed),
    "wr": round(len(wins)/max(len(closed),1)*100,1),
    "pf": round(tw/max(tl,0.01),2),
    "pnl": round(sum(t.pnl for t in closed),2),
    "dd": round(0,2),  # filled by caller
    "avg_daily": round(avg_daily,2),
    "green_weeks": green_w, "red_weeks": red_w,
    "avg_win": round(tw/max(len(wins),1),2),
    "avg_loss": round(tl/max(len(losses),1),2),
    "days_100": len([p for p in daily_pnl.values() if p >= 100]),
}
```

def main():
print(”””
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — FULL WEEKLY ANALYZER              ║
║     + 30 verbeteringen testen voor meer winst                ║
╚══════════════════════════════════════════════════════════════╝
“””)
print(“📥 Downloading 60-day gold data…”)
df = yf.download(“GC=F”, period=“60d”, interval=“5m”, progress=True)
if df.empty: df = yf.download(“GC=F”, period=“60d”, interval=“1h”, progress=True)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
if df.index.tz is not None: df.index = df.index.tz_localize(None)
df = df.dropna(subset=[“Open”,“High”,“Low”,“Close”]); df = df[df[“Close”]>0]
print(f”✅ {len(df)} bars: {df.index[0]} → {df.index[-1]}”)
df = calculate_indicators(df)
sh, sl = detect_swings(df, 3)
t0 = time.time()

```
# ═══════════════════════════════════════════════════════════
#  PHASE 1: BASELINE WEEKLY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  📊 PHASE 1: BASELINE WEEKLY BREAKDOWN")
print(f"{'='*70}")

cfg = Config()
closed, bal, dd = run_backtest(df, cfg, sh, sl)
base_stats = calc_stats(closed, cfg)
base_stats["dd"] = round(dd,2)

# Weekly display
weekly = defaultdict(lambda:{"t":0,"w":0,"pnl":0.0})
hourly = defaultdict(lambda:{"t":0,"w":0,"pnl":0.0})
for t in closed:
    try:
        dt = datetime.strptime(t.day,"%Y-%m-%d"); yr,wk,_ = dt.isocalendar()
        mon = dt - timedelta(days=dt.weekday())
        k = f"{yr}-W{wk:02d} ({mon.strftime('%d %b')})"
    except: k = "?"
    weekly[k]["t"] += 1; weekly[k]["pnl"] += t.pnl
    if t.pnl > 0: weekly[k]["w"] += 1
    hourly[t.hour]["t"] += 1; hourly[t.hour]["pnl"] += t.pnl
    if t.pnl > 0: hourly[t.hour]["w"] += 1

print(f"\n  TOTAAL: {base_stats['trades']}t | WR:{base_stats['wr']}% | PF:{base_stats['pf']} | ${base_stats['pnl']:+,.0f} | DD:{dd:.1f}% | ${base_stats['avg_daily']:+,.0f}/dag\n")
print(f"  {'Week':<28} {'Trades':>6} {'WR':>5} {'PnL':>10}")
print(f"  {'-'*52}")
for w in sorted(weekly.keys()):
    d = weekly[w]; wr = d["w"]/max(d["t"],1)*100
    e = "🟢" if d["pnl"] >= 0 else "🔴"
    print(f"  {e} {w:<26} {d['t']:>6} {wr:>4.0f}% ${d['pnl']:>+9,.0f}")

print(f"\n  PER UUR:")
for h in sorted(hourly.keys()):
    d = hourly[h]; wr = d["w"]/max(d["t"],1)*100
    e = "🟢" if d["pnl"] >= 0 else "🔴"
    sess = "London" if 7<=h<12 else "NY Overlap" if 12<=h<15 else "New York" if 15<=h<17 else "Off"
    print(f"  {e} {h:02d}:00 ({sess:>11}): {d['t']:>4}t | WR:{wr:>4.0f}% | ${d['pnl']:>+8,.0f}")

bad_hours = [h for h in hourly if hourly[h]["pnl"] < -50]

# ═══════════════════════════════════════════════════════════
#  PHASE 2: 30+ IMPROVEMENTS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  🔧 PHASE 2: VERBETERINGEN TESTEN")
print(f"{'='*70}")

tests = []

# ─── INDICATORS ───────────────────────────────────────────
print(f"\n  📈 CATEGORIE: Extra Indicatoren")
for name, field, val in [
    ("+ MACD","USE_MACD",True), ("+ Stochastic","USE_STOCH",True),
    ("+ EMA50 filter","USE_EMA50",True), ("+ HTF Trend","USE_HTF_TREND",True),
    ("+ VWAP","USE_VWAP",True), ("+ RSI Filter","USE_RSI_FILTER",True),
    ("+ BB Squeeze","USE_BB_SQUEEZE",True),
]:
    c = Config(); c.LABEL = name; setattr(c, field, val)
    cl,b,d = run_backtest(df, c, sh, sl); s = calc_stats(cl, c)
    if s: s["dd"] = round(d,2); tests.append(s)
    diff = s["pnl"]-base_stats["pnl"] if s else 0
    e = "🟢" if diff > 0 else "🔴"
    print(f"    {e} {name:<20}: {s['trades']:>4}t WR:{s['wr']:>5.1f}% PF:{s['pf']:>5.2f} ${s['pnl']:>+8,.0f} ({diff:>+,.0f})")

# ─── RISK MANAGEMENT ─────────────────────────────────────
print(f"\n  🛡️ CATEGORIE: Risk Management")
for name, setup in [
    ("Regime filter 0.5", {"USE_REGIME_FILTER":True,"REGIME_LOW_VOL":0.5}),
    ("Regime filter 0.7", {"USE_REGIME_FILTER":True,"REGIME_LOW_VOL":0.7}),
    ("Loss pause 30bar", {"USE_LOSS_PAUSE":True,"LOSS_PAUSE_BARS":30}),
    ("Loss pause 60bar", {"USE_LOSS_PAUSE":True,"LOSS_PAUSE_BARS":60}),
    ("Daily target $150", {"USE_DAILY_TARGET":True,"DAILY_TARGET_USD":150}),
    ("Daily target $200", {"USE_DAILY_TARGET":True,"DAILY_TARGET_USD":200}),
    ("Scale high conf", {"USE_SCALING":True,"SCALE_HIGH_CONF":1.5}),
    ("Risk 0.75%", {"RISK_PERCENT":0.75}),
    ("Risk 1.5%", {"RISK_PERCENT":1.5}),
    ("Max 4 concurrent", {"MAX_CONCURRENT_TRADES":4}),
    ("Max 5 consec loss", {"MAX_CONSECUTIVE_LOSSES":5}),
    ("Max 3 consec loss", {"MAX_CONSECUTIVE_LOSSES":3}),
]:
    c = Config(); c.LABEL = name
    for k,v in setup.items(): setattr(c, k, v)
    cl,b,d = run_backtest(df, c, sh, sl); s = calc_stats(cl, c)
    if s: s["dd"] = round(d,2); tests.append(s)
    diff = s["pnl"]-base_stats["pnl"] if s else 0
    e = "🟢" if diff > 0 else "🔴"
    print(f"    {e} {name:<20}: {s['trades']:>4}t WR:{s['wr']:>5.1f}% PF:{s['pf']:>5.2f} ${s['pnl']:>+8,.0f} ({diff:>+,.0f})")

# ─── STRATEGY ─────────────────────────────────────────────
print(f"\n  🎯 CATEGORIE: Strategie")
for name, setup in [
    ("Trend only", {"USE_TREND_ONLY":True}),
    ("TP2 at 1.0R", {"TP2_ENABLED":True,"TP2_RR":1.0}),
    ("TP2 at 1.5R", {"TP2_ENABLED":True,"TP2_RR":1.5}),
    ("RR 1.5", {"RR_RATIO":1.5}),
    ("RR 2.5", {"RR_RATIO":2.5}),
    ("RR 3.0", {"RR_RATIO":3.0}),
    ("TP1 0.3R", {"TP1_RR":0.3}),
    ("TP1 0.5R", {"TP1_RR":0.5}),
    ("Partial 50%", {"PARTIAL_PERCENT":0.50}),
    ("Partial 75%", {"PARTIAL_PERCENT":0.75}),
    ("SL ATR×2.0", {"ATR_SL_MULTIPLIER":2.0}),
    ("SL ATR×3.0", {"ATR_SL_MULTIPLIER":3.0}),
    ("Conf≥4", {"MIN_CONFLUENCE":4}),
    ("Conf≥5", {"MIN_CONFLUENCE":5}),
    ("CD 3/5 (faster)", {"TRADE_COOLDOWN_BARS":3,"LOSS_COOLDOWN_BARS":5}),
    ("CD 8/20 (slower)", {"TRADE_COOLDOWN_BARS":8,"LOSS_COOLDOWN_BARS":20}),
]:
    c = Config(); c.LABEL = name
    for k,v in setup.items(): setattr(c, k, v)
    cl,b,d = run_backtest(df, c, sh, sl); s = calc_stats(cl, c)
    if s: s["dd"] = round(d,2); tests.append(s)
    diff = s["pnl"]-base_stats["pnl"] if s else 0
    e = "🟢" if diff > 0 else "🔴"
    print(f"    {e} {name:<20}: {s['trades']:>4}t WR:{s['wr']:>5.1f}% PF:{s['pf']:>5.2f} ${s['pnl']:>+8,.0f} ({diff:>+,.0f})")

# ─── SESSION ──────────────────────────────────────────────
print(f"\n  🕐 CATEGORIE: Sessie")
for name, setup in [
    ("London only", {"LONDON_ONLY":True}),
    ("NY only", {"NY_ONLY":True}),
]:
    c = Config(); c.LABEL = name
    for k,v in setup.items(): setattr(c, k, v)
    cl,b,d = run_backtest(df, c, sh, sl); s = calc_stats(cl, c)
    if s: s["dd"] = round(d,2); tests.append(s)
    diff = s["pnl"]-base_stats["pnl"] if s else 0
    e = "🟢" if diff > 0 else "🔴"
    print(f"    {e} {name:<20}: {s['trades']:>4}t WR:{s['wr']:>5.1f}% PF:{s['pf']:>5.2f} ${s['pnl']:>+8,.0f} ({diff:>+,.0f})")

if bad_hours:
    c = Config(); c.LABEL = f"Skip uren {bad_hours}"; c.SKIP_HOURS = bad_hours
    cl,b,d = run_backtest(df, c, sh, sl); s = calc_stats(cl, c)
    if s: s["dd"] = round(d,2); tests.append(s)
    diff = s["pnl"]-base_stats["pnl"] if s else 0
    e = "🟢" if diff > 0 else "🔴"
    print(f"    {e} {c.LABEL:<20}: {s['trades']:>4}t WR:{s['wr']:>5.1f}% PF:{s['pf']:>5.2f} ${s['pnl']:>+8,.0f} ({diff:>+,.0f})")

# ═══════════════════════════════════════════════════════════
#  PHASE 3: RANKED RESULTS
# ═══════════════════════════════════════════════════════════
all_r = [base_stats] + tests
better = [r for r in tests if r["pnl"] > base_stats["pnl"]]
worse = [r for r in tests if r["pnl"] <= base_stats["pnl"]]

print(f"\n{'='*70}")
print(f"  🏆 TOP 10 VERBETERINGEN (meer winst dan baseline)")
print(f"{'='*70}")
print(f"\n  {'#':>2} {'Config':<22} {'Trades':>6} {'WR':>6} {'PF':>5} {'$/Dag':>7} {'PnL':>10} {'DD':>5} {'vs Base':>9}")
print(f"  {'-'*78}")
print(f"  {'--':>2} {'v1.5.1 BASELINE':<22} {base_stats['trades']:>6} {base_stats['wr']:>5.1f}% {base_stats['pf']:>4.2f} ${base_stats['avg_daily']:>+5.0f} ${base_stats['pnl']:>+9,.0f} {base_stats['dd']:>4.1f}% {'⬅️':>9}")

for i, r in enumerate(sorted(better, key=lambda x: x["pnl"], reverse=True)[:10]):
    diff = r["pnl"]-base_stats["pnl"]
    print(f"  {i+1:>2} {r['label']:<22} {r['trades']:>6} {r['wr']:>5.1f}% {r['pf']:>4.2f} ${r['avg_daily']:>+5.0f} ${r['pnl']:>+9,.0f} {r['dd']:>4.1f}% ${diff:>+8,.0f}")

print(f"\n  ❌ SLECHTSTE 5 (vermijden!):")
for r in sorted(worse, key=lambda x: x["pnl"])[:5]:
    diff = r["pnl"]-base_stats["pnl"]
    print(f"    🔴 {r['label']:<22} WR:{r['wr']}% PF:{r['pf']} ${r['pnl']:>+9,.0f} ({diff:>+,.0f})")

# ═══════════════════════════════════════════════════════════
#  PHASE 4: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ⭐ AANBEVELINGEN")
print(f"{'='*70}")

if better:
    top = sorted(better, key=lambda x: x["pnl"], reverse=True)[0]
    print(f"""
```

🏆 BESTE VERBETERING: {top[‘label’]}
├── PnL: ${base_stats[‘pnl’]:+,.0f} → ${top[‘pnl’]:+,.0f} (${top[‘pnl’]-base_stats[‘pnl’]:+,.0f} meer)
├── WR: {base_stats[‘wr’]}% → {top[‘wr’]}%
├── PF: {base_stats[‘pf’]} → {top[‘pf’]}
├── DD: {base_stats[‘dd’]}% → {top[‘dd’]}%
├── $/dag: ${base_stats[‘avg_daily’]:+,.0f} → ${top[‘avg_daily’]:+,.0f}
└── $100+ dagen: {base_stats[‘days_100’]} → {top[‘days_100’]}”””)

```
    # Safe improvements (better PnL AND lower DD)
    safe = [r for r in better if r["dd"] <= base_stats["dd"]*1.1 and r["pf"] >= 1.0]
    if safe:
        best_safe = sorted(safe, key=lambda x: x["pnl"], reverse=True)[0]
        print(f"""
```

🛡️ VEILIGSTE VERBETERING: {best_safe[‘label’]}
├── PnL: ${best_safe[‘pnl’]:+,.0f} | DD: {best_safe[‘dd’]}%
└── Meer winst ZONDER meer risico”””)
else:
print(f”\n  ✅ v1.5.1 is al de beste configuratie!”)
print(f”  Geen enkele verbetering scoort beter.”)

```
# Summary advice
print(f"""
```

💡 SAMENVATTING:
├── {len(better)} van {len(tests)} tests zijn beter dan baseline
├── {len(worse)} zijn slechter
├── Baseline verdient ${base_stats[‘avg_daily’]:+,.0f}/dag
└── Beste optie verdient ${sorted(better, key=lambda x: x[‘pnl’], reverse=True)[0][‘avg_daily’]:+,.0f}/dag””” if better else “”)

```
print(f"\n  ⏱️  Klaar in {time.time()-t0:.0f}s")

output = {
    "baseline": {k:v for k,v in base_stats.items()},
    "improvements": sorted([{k:v for k,v in r.items()} for r in better], key=lambda x: x["pnl"], reverse=True)[:10],
    "avoid": sorted([{k:v for k,v in r.items()} for r in worse], key=lambda x: x["pnl"])[:5],
}
with open("weekly_analysis.json","w") as f:
    json.dump(output, f, indent=2)
print(f"  📁 Saved: weekly_analysis.json")
```

if **name** == “**main**”:
main()
