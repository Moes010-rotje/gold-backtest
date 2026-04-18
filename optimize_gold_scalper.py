"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — v1.5.1 vs v1.5.2 BACKTEST        ║
║  v1.5.1: 8 signalen, alle uren                              ║
║  v1.5.2: 6 signalen, skip 07/12/16 UTC                      ║
║  Weekly + daily breakdown + volledige vergelijking            ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum
from collections import defaultdict
from datetime import datetime

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


@dataclass
class Config:
    LABEL: str = "v1.5.1"
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 1.0
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 3
    MAX_DAILY_TRADES: int = 30
    MAX_CONSECUTIVE_LOSSES: int = 5

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

    # v1.5.2 toggles
    USE_SWEEP: bool = True
    USE_DOUBLE: bool = True
    SKIP_BAD_HOURS: bool = False
    BAD_HOURS: tuple = (7, 12, 16)


def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["tr"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()
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
    df["body"] = abs(df["Close"]-df["Open"])
    df["candle_range"] = df["High"]-df["Low"]
    df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df["tr"].rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()
    return df


def detect_swings(df, lookback=3):
    sh, sl = [], []
    for i in range(lookback, len(df)-lookback):
        h, l = df["High"].iloc[i], df["Low"].iloc[i]
        if all(h >= df["High"].iloc[i+j] and h >= df["High"].iloc[i-j] for j in range(1, lookback+1)): sh.append(i)
        if all(l <= df["Low"].iloc[i+j] and l <= df["Low"].iloc[i-j] for j in range(1, lookback+1)): sl.append(i)
    return sh, sl


def detect_signal(df, i, cfg, swing_highs, swing_lows):
    if i < 60: return None
    price = df["Close"].iloc[i]
    atr = df["atr"].iloc[i]
    if pd.isna(atr) or atr <= 0: return None
    ema9, ema21 = df["ema9"].iloc[i], df["ema21"].iloc[i]
    rsi = df["rsi"].iloc[i]
    if pd.isna(ema9) or pd.isna(ema21): return None
    ts = df.index[i]
    if not hasattr(ts, 'hour') or not (7 <= ts.hour < 17): return None

    # Skip bad hours
    if cfg.SKIP_BAD_HOURS and ts.hour in cfg.BAD_HOURS:
        return None

    confluence = 0; reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
    body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

    # 1. EMA
    if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
    else: votes[Direction.SHORT] += 1; reasons.append("ema")

    # 2. Sweep (toggleable)
    if cfg.USE_SWEEP:
        for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
            if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
                wick = min(cc, co) - cl
                if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.LONG] += 2; reasons.append("sweep"); break
        for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
            if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
                wick = ch - max(cc, co)
                if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    # 3. OB
    if i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # 4. FVG
    if i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    # 5. Momentum
    if total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    # 6. Mean Reversion
    bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
    if not (pd.isna(bb_u) or pd.isna(rsi)):
        bbr = bb_u - bb_l
        if bbr > 0:
            pct_b = (price-bb_l)/bbr
            if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
            elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    # 7. RSI div
    rsi14 = df["rsi14"].iloc[i]
    if not pd.isna(rsi14):
        if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
        elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # 8. Double (toggleable)
    if cfg.USE_DOUBLE:
        recent_sl = [s for s in swing_lows if s < i and s > i-30]
        recent_sh = [s for s in swing_highs if s < i and s > i-30]
        tol = atr * cfg.DOUBLE_TOLERANCE_ATR
        if len(recent_sl) >= 2:
            l1, l2 = df["Low"].iloc[recent_sl[-2]], df["Low"].iloc[recent_sl[-1]]
            if abs(l1-l2) < tol and price > max(l1,l2): votes[Direction.LONG] += 1; reasons.append("dbl")
        if len(recent_sh) >= 2:
            h1, h2 = df["High"].iloc[recent_sh[-2]], df["High"].iloc[recent_sh[-1]]
            if abs(h1-h2) < tol and price < min(h1,h2): votes[Direction.SHORT] += 1; reasons.append("dbl")

    # 9. ADX
    adx = df["adx"].iloc[i]
    if not pd.isna(adx) and adx >= cfg.ADX_THRESHOLD:
        confluence += 1; reasons.append("adx")

    ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
    if ls > ss and ls >= 1: direction = Direction.LONG; confluence += ls
    elif ss > ls and ss >= 1: direction = Direction.SHORT; confluence += ss
    else: return None

    if direction == Direction.LONG and ema9 <= ema21 and "MR" not in reasons: return None
    if direction == Direction.SHORT and ema9 >= ema21 and "MR" not in reasons: return None
    if confluence < cfg.MIN_CONFLUENCE: return None
    return direction, confluence, "|".join(reasons)


@dataclass
class Trade:
    direction: Direction; entry: float; sl: float; tp: float; tp1: float
    lots: float; bar: int; day: str = ""; hour: int = 0
    phase: TradePhase = TradePhase.OPEN; pnl: float = 0.0; remaining: float = 0.0
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg, sh, sl, label=""):
    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    daily_pnl = defaultdict(float)

    for i in range(60, len(df)):
        price, high, low = df["Close"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i]
        atr = df["atr"].iloc[i]
        if pd.isna(atr) or price <= 0: continue
        ts = df.index[i]
        today = str(ts.date()) if hasattr(ts,'date') else str(ts)[:10]
        if today != daily_date: daily_date = today; daily_trades = 0

        for t in list(active):
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                daily_pnl[today] += t.pnl
                consec_losses += 1; llb = i; active.remove(t); closed.append(t); continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                daily_pnl[today] += t.pnl
                consec_losses = 0; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
                    cl = round(t.lots*cfg.PARTIAL_PERCENT, 2)
                    if cl >= 0.01:
                        p = ((t.tp1-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp1))*cl*100
                        p -= cfg.COMMISSION_PER_LOT*cl; balance += p; t.pnl += p
                        daily_pnl[today] += p
                        t.remaining = round(t.remaining-cl, 2); t.phase = TradePhase.TP1_HIT
                        if cfg.MOVE_SL_TO_BE: t.sl = t.entry

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
        trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1,
                      lots=lots, bar=i, day=today, hour=ts.hour if hasattr(ts,'hour') else 0)
        active.append(trade)
        daily_trades += 1; ltb = i

    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            daily_pnl[t.day] += t.pnl
            closed.append(t)

    return closed, balance, max_dd, dict(daily_pnl)


def weekly_breakdown(closed, daily_pnl, label):
    """Print weekly + daily breakdown."""
    from datetime import timedelta

    weekly = defaultdict(lambda: {"trades":0,"wins":0,"pnl":0.0,"days":set()})
    for t in closed:
        try:
            dt = datetime.strptime(t.day, "%Y-%m-%d")
            yr, wk, _ = dt.isocalendar()
            monday = dt - timedelta(days=dt.weekday())
            week_key = f"{yr}-W{wk:02d} ({monday.strftime('%d %b')})"
        except: week_key = "?"
        w = weekly[week_key]
        w["trades"] += 1; w["pnl"] += t.pnl; w["days"].add(t.day)
        if t.pnl > 0: w["wins"] += 1

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    wr = len(wins)/max(len(closed),1)*100
    pf = tw/tl if tl > 0 else 99
    net = sum(t.pnl for t in closed)

    green_w = sum(1 for w in weekly.values() if w["pnl"] >= 0)
    red_w = sum(1 for w in weekly.values() if w["pnl"] < 0)

    # Monthly
    monthly = defaultdict(float)
    for t in closed:
        m = t.day[:7] if t.day else "?"
        monthly[m] += t.pnl

    # Daily stats
    trading_days = [d for d, p in daily_pnl.items() if abs(p) > 0.01]
    daily_vals = [daily_pnl[d] for d in trading_days]
    green_d = sum(1 for p in daily_vals if p > 0)
    avg_day = sum(daily_vals)/max(len(daily_vals),1)
    days_100 = sum(1 for p in daily_vals if p >= 100)

    print(f"\n  {'Week':<25} {'Trades':>7} {'WR':>6} {'PnL':>10}")
    print(f"  {'-'*50}")
    for week in sorted(weekly.keys()):
        w = weekly[week]
        wr_w = w["wins"]/max(w["trades"],1)*100
        emoji = "🟢" if w["pnl"] >= 0 else "🔴"
        print(f"  {emoji} {week:<23} {w['trades']:>7} {wr_w:>5.0f}% ${w['pnl']:>+9,.0f}")

    print(f"\n  Groene weken: {green_w} | Rode weken: {red_w} | Week WR: {green_w/max(green_w+red_w,1)*100:.0f}%")

    print(f"\n  ─── MAANDELIJKS ─────────────────")
    for m in sorted(monthly.keys()):
        emoji = "🟢" if monthly[m] >= 0 else "🔴"
        print(f"  {emoji} {m}: ${monthly[m]:>+10,.2f}")

    return {
        "label": label, "trades": len(closed), "wr": round(wr,1), "pf": round(pf,2),
        "pnl": round(net,2), "dd": 0, "green_weeks": green_w, "red_weeks": red_w,
        "green_days": green_d, "trading_days": len(trading_days),
        "avg_daily": round(avg_day,2), "days_100": days_100,
        "monthly": {m: round(v,2) for m,v in monthly.items()},
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — v1.5.1 vs v1.5.2 BACKTEST        ║
║     v1.5.1: 8 signalen, alle uren                           ║
║     v1.5.2: 6 signalen, skip 07/12/16 UTC                   ║
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

    df = calculate_indicators(df)
    sh, sl = detect_swings(df, 3)

    # ═══════════════════════════════════════════════════════
    #  TEST 1: v1.5.1 (current)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  🔵 v1.5.1 — HUIDIG (8 signalen, alle uren)")
    print(f"{'='*60}")

    cfg1 = Config()
    cfg1.LABEL = "v1.5.1"
    cfg1.USE_SWEEP = True; cfg1.USE_DOUBLE = True; cfg1.SKIP_BAD_HOURS = False
    c1, b1, d1, dp1 = run_backtest(df, cfg1, sh, sl)
    r1 = weekly_breakdown(c1, dp1, "v1.5.1")
    r1["dd"] = round(d1, 2)
    r1["balance"] = round(b1, 2)

    w1 = [t for t in c1 if t.pnl > 0]; l1_t = [t for t in c1 if t.pnl <= 0]
    print(f"\n  TOTAAL: {r1['trades']}t | WR:{r1['wr']}% | PF:{r1['pf']} | ${r1['pnl']:+,.0f} | DD:{r1['dd']:.1f}%")
    print(f"  Avg/dag: ${r1['avg_daily']:+.0f} | $100+ dagen: {r1['days_100']} | Groene dagen: {r1['green_days']}/{r1['trading_days']}")

    # ═══════════════════════════════════════════════════════
    #  TEST 2: v1.5.2 (improved)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  🟢 v1.5.2 — NIEUW (6 signalen, skip bad hours)")
    print(f"{'='*60}")

    cfg2 = Config()
    cfg2.LABEL = "v1.5.2"
    cfg2.USE_SWEEP = False; cfg2.USE_DOUBLE = False; cfg2.SKIP_BAD_HOURS = True
    c2, b2, d2, dp2 = run_backtest(df, cfg2, sh, sl)
    r2 = weekly_breakdown(c2, dp2, "v1.5.2")
    r2["dd"] = round(d2, 2)
    r2["balance"] = round(b2, 2)

    w2 = [t for t in c2 if t.pnl > 0]; l2_t = [t for t in c2 if t.pnl <= 0]
    print(f"\n  TOTAAL: {r2['trades']}t | WR:{r2['wr']}% | PF:{r2['pf']} | ${r2['pnl']:+,.0f} | DD:{r2['dd']:.1f}%")
    print(f"  Avg/dag: ${r2['avg_daily']:+.0f} | $100+ dagen: {r2['days_100']} | Groene dagen: {r2['green_days']}/{r2['trading_days']}")

    # ═══════════════════════════════════════════════════════
    #  VERGELIJKING
    # ═══════════════════════════════════════════════════════
    print(f"""
{'='*60}
  ⚖️ VERGELIJKING: v1.5.1 vs v1.5.2
{'='*60}

  {'':>25} {'v1.5.1':>12} {'v1.5.2':>12} {'Verschil':>12}
  {'─'*62}
  {'Trades':>25} {r1['trades']:>12} {r2['trades']:>12} {r2['trades']-r1['trades']:>+12}
  {'Win Rate':>25} {f"{r1['wr']}%":>12} {f"{r2['wr']}%":>12} {f"{r2['wr']-r1['wr']:+.1f}%":>12}
  {'Profit Factor':>25} {r1['pf']:>12} {r2['pf']:>12} {f"{r2['pf']-r1['pf']:+.2f}":>12}
  {'Net PnL':>25} {f"${r1['pnl']:+,.0f}":>12} {f"${r2['pnl']:+,.0f}":>12} {f"${r2['pnl']-r1['pnl']:+,.0f}":>12}
  {'Max DD':>25} {f"{r1['dd']}%":>12} {f"{r2['dd']}%":>12} {f"{r2['dd']-r1['dd']:+.1f}%":>12}
  {'Avg $/dag':>25} {f"${r1['avg_daily']:+,.0f}":>12} {f"${r2['avg_daily']:+,.0f}":>12} {f"${r2['avg_daily']-r1['avg_daily']:+,.0f}":>12}
  {'$100+ dagen':>25} {r1['days_100']:>12} {r2['days_100']:>12} {r2['days_100']-r1['days_100']:>+12}
  {'Groene weken':>25} {f"{r1['green_weeks']}/{r1['green_weeks']+r1['red_weeks']}":>12} {f"{r2['green_weeks']}/{r2['green_weeks']+r2['red_weeks']}":>12} {"":>12}
  {'Rode weken':>25} {r1['red_weeks']:>12} {r2['red_weeks']:>12} {r2['red_weeks']-r1['red_weeks']:>+12}
  {'Groene dagen':>25} {f"{r1['green_days']}/{r1['trading_days']}":>12} {f"{r2['green_days']}/{r2['trading_days']}":>12} {"":>12}""")

    # Monthly comparison
    print(f"\n  {'─'*62}")
    print(f"  {'MAAND':>25} {'v1.5.1':>12} {'v1.5.2':>12} {'Verschil':>12}")
    print(f"  {'─'*62}")
    all_months = sorted(set(list(r1["monthly"].keys()) + list(r2["monthly"].keys())))
    for m in all_months:
        m1 = r1["monthly"].get(m, 0)
        m2 = r2["monthly"].get(m, 0)
        e1 = "🟢" if m1 >= 0 else "🔴"
        e2 = "🟢" if m2 >= 0 else "🔴"
        print(f"  {m:>25} {e1}${m1:>+9,.0f} {e2}${m2:>+9,.0f} ${m2-m1:>+10,.0f}")

    # Verdict
    better_pnl = r2["pnl"] > r1["pnl"]
    better_wr = r2["wr"] > r1["wr"]
    better_pf = r2["pf"] > r1["pf"]
    fewer_red = r2["red_weeks"] < r1["red_weeks"]
    score = sum([better_pnl, better_wr, better_pf, fewer_red])

    if score >= 3:
        verdict = "✅ v1.5.2 IS BETER! Upload naar live bot."
    elif score >= 2:
        verdict = "⚠️ v1.5.2 is iets beter. Overweeg te uploaden."
    else:
        verdict = "❌ v1.5.1 is beter. Houd huidige settings."

    print(f"""
{'='*60}
  🏆 VERDICT: {verdict}
  
  Scores: PnL {'✅' if better_pnl else '❌'} | WR {'✅' if better_wr else '❌'} | PF {'✅' if better_pf else '❌'} | Weken {'✅' if fewer_red else '❌'}
{'='*60}""")

    # Save
    output = {"v1.5.1": {k:v for k,v in r1.items() if k != "monthly"},
              "v1.5.2": {k:v for k,v in r2.items() if k != "monthly"},
              "verdict": verdict}
    with open("v152_backtest.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"  📁 Saved: v152_backtest.json")


if __name__ == "__main__":
    main()
