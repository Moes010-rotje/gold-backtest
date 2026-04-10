"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — BACKTEST v1.6                     ║
║  v1.4 trades (ATR×2.0, short CD) + v1.5 WR (TP1 0.4R, 67%) ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, time
from dataclasses import dataclass
from typing import Optional, List, Tuple
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


# v1.6 settings
@dataclass
class Config:
    START_BALANCE: float = 5000.0
    COMMISSION_PER_LOT: float = 7.0
    SIMULATED_SPREAD: float = 0.30
    RISK_PERCENT: float = 0.5
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PERCENT: float = 10.0
    MAX_CONCURRENT_TRADES: int = 3
    MAX_DAILY_TRADES: int = 30
    MAX_CONSECUTIVE_LOSSES: int = 5

    # v1.6: v1.4 base
    ATR_SL_MULTIPLIER: float = 2.0
    MIN_SL_POINTS: float = 2.0
    MAX_SL_POINTS: float = 10.0
    RR_RATIO: float = 2.0

    # v1.6: v1.5 WR boosters
    PARTIAL_PERCENT: float = 0.67      # from v1.5
    TP1_RR: float = 0.4               # from v1.5
    MOVE_SL_TO_BE: bool = True

    SWING_LOOKBACK: int = 3
    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    MR_CONFLUENCE_SCORE: int = 2

    MIN_CONFLUENCE: int = 3

    # v1.6: v1.4 cooldowns (more trades)
    TRADE_COOLDOWN_BARS: int = 3       # 30s
    LOSS_COOLDOWN_BARS: int = 5        # 50s

    # 8 active signals (5 disabled by v4 audit)
    SIG_EMA: bool = True
    SIG_SWEEP: bool = True
    SIG_OB: bool = True
    SIG_FVG: bool = True
    SIG_MOMENTUM: bool = True
    SIG_MR: bool = True
    SIG_RSI_DIV: bool = True
    SIG_DOUBLE: bool = True
    SIG_ADX: bool = True
    ADX_THRESHOLD: float = 25.0

    SWEEP_WICK_ATR_MIN: float = 0.25
    FVG_MIN_ATR: float = 0.25
    DOUBLE_TOLERANCE_ATR: float = 0.3


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

    confluence = 0; reasons = []
    votes = {Direction.LONG: 0, Direction.SHORT: 0}
    ch, cl, cc, co = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
    body, total = df["body"].iloc[i], df["candle_range"].iloc[i]

    if cfg.SIG_EMA:
        if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
        else: votes[Direction.SHORT] += 1; reasons.append("ema")

    if cfg.SIG_SWEEP:
        for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
            if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
                wick = min(cc, co) - cl
                if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.LONG] += 2; reasons.append("sweep"); break
        for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
            if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
                wick = ch - max(cc, co)
                if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    if cfg.SIG_OB and i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    if cfg.SIG_FVG and i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    if cfg.SIG_MOMENTUM and total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    if cfg.SIG_MR:
        bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
        if not (pd.isna(bb_u) or pd.isna(rsi)):
            bbr = bb_u - bb_l
            if bbr > 0:
                pct_b = (price-bb_l)/bbr
                if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
                elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    if cfg.SIG_RSI_DIV:
        rsi14 = df["rsi14"].iloc[i]
        if not pd.isna(rsi14):
            if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
            elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    if cfg.SIG_DOUBLE:
        recent_sl = [s for s in swing_lows if s < i and s > i-30]
        recent_sh = [s for s in swing_highs if s < i and s > i-30]
        tol = atr * cfg.DOUBLE_TOLERANCE_ATR
        if len(recent_sl) >= 2:
            l1, l2 = df["Low"].iloc[recent_sl[-2]], df["Low"].iloc[recent_sl[-1]]
            if abs(l1-l2) < tol and price > max(l1,l2): votes[Direction.LONG] += 1; reasons.append("dbl")
        if len(recent_sh) >= 2:
            h1, h2 = df["High"].iloc[recent_sh[-2]], df["High"].iloc[recent_sh[-1]]
            if abs(h1-h2) < tol and price < min(h1,h2): votes[Direction.SHORT] += 1; reasons.append("dbl")

    if cfg.SIG_ADX:
        adx = df["adx"].iloc[i]
        if not pd.isna(adx) and adx >= cfg.ADX_THRESHOLD:
            confluence += 1; reasons.append("adx")

    ls, ss = votes[Direction.LONG], votes[Direction.SHORT]
    if ls > ss and ls >= 1: direction = Direction.LONG; confluence += ls
    elif ss > ls and ss >= 1: direction = Direction.SHORT; confluence += ss
    else: return None

    if cfg.SIG_EMA:
        if direction == Direction.LONG and ema9 <= ema21 and "MR" not in reasons: return None
        if direction == Direction.SHORT and ema9 >= ema21 and "MR" not in reasons: return None

    if confluence < cfg.MIN_CONFLUENCE: return None
    return direction, confluence, "|".join(reasons)


@dataclass
class Trade:
    direction: Direction; entry: float; sl: float; tp: float; tp1: float
    lots: float; bar: int; entry_time: str = ""
    phase: TradePhase = TradePhase.OPEN
    pnl: float = 0.0; remaining: float = 0.0
    exit_reason: str = ""; exit_time: str = ""
    def __post_init__(self): self.remaining = self.lots


def run_backtest(df, cfg):
    print(f"\n{'='*60}")
    print(f"  BACKTESTING v1.6 SETTINGS")
    print(f"  ATR×{cfg.ATR_SL_MULTIPLIER} | RR 1:{cfg.RR_RATIO} | TP1 {cfg.TP1_RR}R | Partial {cfg.PARTIAL_PERCENT*100:.0f}%")
    print(f"  CD: {cfg.TRADE_COOLDOWN_BARS}/{cfg.LOSS_COOLDOWN_BARS} | Confluence ≥{cfg.MIN_CONFLUENCE}")
    print(f"  Period: {df.index[0]} → {df.index[-1]} | {len(df)} bars")
    print(f"{'='*60}\n")

    df = calculate_indicators(df)
    sh, sl = detect_swings(df, cfg.SWING_LOOKBACK)

    balance = cfg.START_BALANCE; peak = balance; max_dd = 0.0
    active, closed = [], []; daily_trades = 0; daily_date = ""
    consec_losses = 0; ltb = -999; llb = -999
    monthly_pnl = {}
    signal_stats = {}

    total_bars = len(df)
    report_interval = max(1, total_bars // 10)

    for i in range(60, len(df)):
        if (i-60) % report_interval == 0:
            pct = (i-60)/(total_bars-60)*100
            print(f"  {pct:.0f}% | Bar {i}/{total_bars} | Balance: ${balance:,.2f} | Trades: {len(closed)}")

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
                t.exit_reason = "SL"; t.exit_time = str(ts)
                consec_losses += 1; llb = i; active.remove(t); closed.append(t); continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                t.exit_reason = "TP"; t.exit_time = str(ts)
                consec_losses = 0; active.remove(t); closed.append(t); continue
            if t.phase == TradePhase.OPEN:
                if (t.direction==Direction.LONG and high>=t.tp1) or (t.direction==Direction.SHORT and low<=t.tp1):
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
        trade = Trade(direction=direction, entry=entry, sl=s, tp=t, tp1=t1, lots=lots, bar=i, entry_time=str(ts))
        active.append(trade)
        daily_trades += 1; ltb = i

        # Track signal stats
        for r in reason.split("|"):
            if r not in signal_stats: signal_stats[r] = {"count": 0}
            signal_stats[r]["count"] += 1

    # Close remaining
    if active:
        lp = df["Close"].iloc[-1]
        for t in active:
            t.pnl += ((lp-t.entry) if t.direction==Direction.LONG else (t.entry-lp))*t.remaining*100
            t.exit_reason = "END"; closed.append(t)

    if not closed:
        print("  No trades!"); return

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    tw = sum(t.pnl for t in wins); tl = abs(sum(t.pnl for t in losses))
    wr = len(wins)/len(closed)*100
    pf = tw/tl if tl > 0 else 99
    net = balance - cfg.START_BALANCE
    ret = net/cfg.START_BALANCE*100

    # Monthly PnL
    for t in closed:
        m = t.entry_time[:7] if t.entry_time else "unknown"
        if m not in monthly_pnl: monthly_pnl[m] = 0
        monthly_pnl[m] += t.pnl

    # Session stats
    session_stats = {}
    for t in closed:
        try:
            h = int(t.entry_time[11:13])
            if 12 <= h < 15: s = "overlap"
            elif 7 <= h < 12: s = "london"
            elif 12 <= h < 17: s = "new_york"
            else: s = "off"
        except: s = "unknown"
        if s not in session_stats: session_stats[s] = {"t":0,"w":0,"pnl":0}
        session_stats[s]["t"] += 1
        if t.pnl > 0: session_stats[s]["w"] += 1
        session_stats[s]["pnl"] += t.pnl

    # Exit stats
    exit_stats = {}
    for t in closed:
        r = t.exit_reason
        if r not in exit_stats: exit_stats[r] = {"count":0, "pnl":0}
        exit_stats[r]["count"] += 1
        exit_stats[r]["pnl"] += t.pnl

    # Direction stats
    longs = [t for t in closed if t.direction == Direction.LONG]
    shorts = [t for t in closed if t.direction == Direction.SHORT]
    lw = len([t for t in longs if t.pnl > 0])
    sw = len([t for t in shorts if t.pnl > 0])

    # Streaks
    max_ws, max_ls, cw, cl = 0, 0, 0, 0
    for t in closed:
        if t.pnl > 0: cw += 1; cl = 0; max_ws = max(max_ws, cw)
        else: cl += 1; cw = 0; max_ls = max(max_ls, cl)

    # Print report
    print(f"\n{'='*60}")
    print(f"  📊 v1.6 BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"""
  💰 Start:    ${cfg.START_BALANCE:>10,.2f}
  💰 Final:    ${balance:>10,.2f}
  📈 Net P&L:  ${net:>+10,.2f}
  📈 Return:   {ret:>+9.2f}%
  📉 Max DD:   {max_dd:>9.2f}%

  ─── TRADES ──────────────────────────────────
  Total:       {len(closed):>8}
  Wins:        {len(wins):>8}
  Losses:      {len(losses):>8}
  Win Rate:    {wr:>7.1f}%
  PF:          {pf:>8.2f}

  ─── AVERAGES ────────────────────────────────
  Avg Win:     ${tw/max(len(wins),1):>+9.2f}
  Avg Loss:    ${tl/max(len(losses),1):>9.2f}

  ─── STREAKS ─────────────────────────────────
  Max Win:     {max_ws:>8}
  Max Loss:    {max_ls:>8}

  ─── DIRECTION ───────────────────────────────
  Long:  {len(longs):>4}t | WR: {lw/max(len(longs),1)*100:.1f}%
  Short: {len(shorts):>4}t | WR: {sw/max(len(shorts),1)*100:.1f}%""")

    print(f"\n  ─── EXIT REASONS ────────────────────────────")
    for r, d in sorted(exit_stats.items()):
        print(f"  {r:>6}: {d['count']:>5}t | PnL: ${d['pnl']:>+10,.2f}")

    print(f"\n  ─── SESSION PERFORMANCE ─────────────────────")
    for s, d in sorted(session_stats.items()):
        swr = d['w']/max(d['t'],1)*100
        print(f"  {s:>10}: {d['t']:>4}t | WR: {swr:.0f}% | PnL: ${d['pnl']:>+10,.2f}")

    print(f"\n  ─── MONTHLY P&L ─────────────────────────────")
    for m in sorted(monthly_pnl.keys()):
        e = "🟢" if monthly_pnl[m] >= 0 else "🔴"
        print(f"  {e} {m}: ${monthly_pnl[m]:>+10,.2f}")

    print(f"\n  ─── SIGNAL FREQUENCY ────────────────────────")
    for sig, d in sorted(signal_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"  {sig:>10}: {d['count']:>5} trades")

    # Comparison
    print(f"""
{'='*60}
  ⚖️ VERGELIJKING
{'='*60}
  {'':>15} {'v1.4':>10} {'v1.5':>10} {'v1.6':>10}
  {'WR':>15} {'63.3%':>10} {'73.7%':>10} {f'{wr:.1f}%':>10}
  {'PF':>15} {'1.31':>10} {'1.46':>10} {f'{pf:.2f}':>10}
  {'Return':>15} {'+117%':>10} {'+72%':>10} {f'+{ret:.0f}%':>10}
  {'Trades':>15} {'549':>10} {'334':>10} {f'{len(closed)}':>10}
  {'DD':>15} {'3.3%':>10} {'2.4%':>10} {f'{max_dd:.1f}%':>10}
{'='*60}""")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — BACKTEST v1.6                     ║
║     v1.4 trades + v1.5 win rate = best of both?             ║
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

    cfg = Config()
    start = time.time()
    run_backtest(df, cfg)
    print(f"\n  ⏱️  Klaar in {time.time()-start:.0f}s")

if __name__ == "__main__":
    main()
