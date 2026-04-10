"""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — OPTIMIZER v5.0                    ║
║  Baseline: v1.4 (5 signals removed, 8 active)               ║
║  Target: Push WR above 63.2% while keeping PF > 1.2         ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, json, math, time
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

    # v1.4 baseline
    ATR_SL_MULTIPLIER: float = 2.0
    MIN_SL_POINTS: float = 2.0
    MAX_SL_POINTS: float = 10.0
    RR_RATIO: float = 2.0
    PARTIAL_PERCENT: float = 0.50
    TP1_RR: float = 0.6
    MOVE_SL_TO_BE: bool = True

    SWING_LOOKBACK: int = 3
    ENGULF_BODY_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 0.8
    RSI_OVERSOLD: float = 25.0
    RSI_OVERBOUGHT: float = 75.0
    MR_CONFLUENCE_SCORE: int = 2

    MIN_CONFLUENCE: int = 3
    TRADE_COOLDOWN_BARS: int = 3
    LOSS_COOLDOWN_BARS: int = 5

    # v1.4 active signals (8 active, 5 disabled)
    SIG_EMA: bool = True
    SIG_SWEEP: bool = True
    SIG_OB: bool = True
    SIG_FVG: bool = True
    SIG_MOMENTUM: bool = True
    SIG_MR: bool = True
    SIG_RSI_DIV: bool = True
    SIG_DOUBLE: bool = True
    SIG_ADX: bool = True  # neutral but kept

    # Disabled by v4 audit
    SIG_EXHAUSTION: bool = False
    SIG_ROUND_NUM: bool = False
    SIG_WICK: bool = False
    SIG_VWAP: bool = False
    SIG_SESSION_LVL: bool = False
    SIG_EMA50: bool = False

    ADX_THRESHOLD: float = 25.0
    ADX_BLOCK_BELOW: float = 0.0
    EXHAUSTION_WICK_RATIO: float = 0.65

    # Extra tuning params
    SWEEP_WICK_ATR_MIN: float = 0.25
    FVG_MIN_ATR: float = 0.25
    DOUBLE_TOLERANCE_ATR: float = 0.3


def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
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
    if "Volume" not in df.columns:
        df["Volume"] = 0
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

    # 1. EMA
    if cfg.SIG_EMA:
        if ema9 > ema21: votes[Direction.LONG] += 1; reasons.append("ema")
        else: votes[Direction.SHORT] += 1; reasons.append("ema")

    # 2. Sweep
    if cfg.SIG_SWEEP:
        for si in [s for s in swing_lows if s < i and s > i-25][-3:]:
            if cl < df["Low"].iloc[si] and cc > df["Low"].iloc[si]:
                wick = min(cc, co) - cl
                if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.LONG] += 2; reasons.append("sweep"); break
        for si in [s for s in swing_highs if s < i and s > i-25][-3:]:
            if ch > df["High"].iloc[si] and cc < df["High"].iloc[si]:
                wick = ch - max(cc, co)
                if wick > atr*cfg.SWEEP_WICK_ATR_MIN: votes[Direction.SHORT] += 2; reasons.append("sweep"); break

    # 3. OB
    if cfg.SIG_OB and i >= 2:
        prev, curr = df.iloc[i-1], df.iloc[i]
        pb, cb = abs(prev["Close"]-prev["Open"]), abs(curr["Close"]-curr["Open"])
        if curr["Close"]>curr["Open"] and prev["Close"]<prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]>prev["High"]:
            votes[Direction.LONG] += 1; reasons.append("ob")
        elif curr["Close"]<curr["Open"] and prev["Close"]>prev["Open"] and cb>pb*cfg.ENGULF_BODY_RATIO and curr["Close"]<prev["Low"]:
            votes[Direction.SHORT] += 1; reasons.append("ob")

    # 4. FVG
    if cfg.SIG_FVG and i >= 2:
        c1h, c3l = df["High"].iloc[i-2], df["Low"].iloc[i]
        c1l, c3h = df["Low"].iloc[i-2], df["High"].iloc[i]
        mg = atr*cfg.FVG_MIN_ATR
        if c3l > c1h and (c3l-c1h) >= mg: votes[Direction.LONG] += 1; reasons.append("fvg")
        elif c1l > c3h and (c1l-c3h) >= mg: votes[Direction.SHORT] += 1; reasons.append("fvg")

    # 5. Momentum
    if cfg.SIG_MOMENTUM and total > 0 and body/total >= cfg.ENGULF_BODY_RATIO and body >= atr*cfg.MOMENTUM_CANDLE_ATR:
        if cc > co: votes[Direction.LONG] += 1; reasons.append("mom")
        else: votes[Direction.SHORT] += 1; reasons.append("mom")

    # 6. Mean Reversion
    if cfg.SIG_MR:
        bb_u, bb_l = df["bb_upper"].iloc[i], df["bb_lower"].iloc[i]
        if not (pd.isna(bb_u) or pd.isna(rsi)):
            bbr = bb_u - bb_l
            if bbr > 0:
                pct_b = (price-bb_l)/bbr
                if pct_b <= 0.05 and rsi <= cfg.RSI_OVERSOLD: votes[Direction.LONG] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")
                elif pct_b >= 0.95 and rsi >= cfg.RSI_OVERBOUGHT: votes[Direction.SHORT] += cfg.MR_CONFLUENCE_SCORE; reasons.append("MR")

    # 7. RSI divergence
    if cfg.SIG_RSI_DIV:
        rsi14 = df["rsi14"].iloc[i]
        if not pd.isna(rsi14):
            if rsi14 < 30 and cc > df["Close"].iloc[i-5]: votes[Direction.LONG] += 1; reasons.append("rsi_div")
            elif rsi14 > 70 and cc < df["Close"].iloc[i-5]: votes[Direction.SHORT] += 1; reasons.append("rsi_div")

    # 8. Double bottom/top
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

    # 9. ADX
    if cfg.SIG_ADX:
        adx = df["adx"].iloc[i]
        if not pd.isna(adx):
            if cfg.ADX_BLOCK_BELOW > 0 and adx < cfg.ADX_BLOCK_BELOW: return None
            if adx >= cfg.ADX_THRESHOLD: confluence += 1; reasons.append("adx")

    # Direction
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
            if (t.direction==Direction.LONG and low<=t.sl) or (t.direction==Direction.SHORT and high>=t.sl):
                t.pnl += ((t.sl-t.entry) if t.direction==Direction.LONG else (t.entry-t.sl))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
                consec_losses += 1; llb = i; active.remove(t); closed.append(t); continue
            if (t.direction==Direction.LONG and high>=t.tp) or (t.direction==Direction.SHORT and low<=t.tp):
                t.pnl += ((t.tp-t.entry) if t.direction==Direction.LONG else (t.entry-t.tp))*t.remaining*100
                t.pnl -= cfg.COMMISSION_PER_LOT*t.remaining; balance += t.pnl
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


def optimize(df):
    print("\n" + "="*60)
    print("  🔧 OPTIMIZER v5.0 — PUSH WR HIGHER")
    print("  Baseline: v1.4 (8 active signals)")
    print("  Target: Beat 63.2% WR with PF > 1.2")
    print("="*60)

    df = calculate_indicators(df)
    sh, sl = detect_swings(df, 3)

    # Baseline v1.4
    print("\n  📊 Phase 1: v1.4 Baseline")
    baseline = run_backtest(df, Config(), sh, sl)
    print(f"  ✅ Baseline: {baseline['trades']}t | WR:{baseline['wr']}% | PF:{baseline['pf']} | ${baseline['pnl']:+.0f} | DD:{baseline['dd']:.1f}%")

    # Phase 2: Massive grid search
    print(f"\n  📊 Phase 2: Massive parameter search")

    all_results = []
    count = 0

    sl_mults = [1.5, 2.0, 2.5, 3.0, 3.5]
    confluences = [3, 4, 5, 6]
    rr_ratios = [1.2, 1.5, 1.8, 2.0, 2.5]
    tp1_rrs = [0.4, 0.5, 0.6, 0.8, 1.0]
    partials = [0.33, 0.50, 0.67]
    cd_trades = [2, 3, 5, 8]
    cd_losses = [3, 5, 8, 12]

    total = len(sl_mults)*len(confluences)*len(rr_ratios)*len(tp1_rrs)
    print(f"  Phase 2a: Core params ({total} combos)...")

    for sl_m in sl_mults:
        for conf in confluences:
            for rr in rr_ratios:
                for tp1 in tp1_rrs:
                    count += 1
                    cfg = Config()
                    cfg.ATR_SL_MULTIPLIER = sl_m
                    cfg.MIN_CONFLUENCE = conf
                    cfg.RR_RATIO = rr
                    cfg.TP1_RR = tp1
                    r = run_backtest(df, cfg, sh, sl)
                    r["params"] = {"sl":sl_m,"conf":conf,"rr":rr,"tp1":tp1,"partial":0.50,"cd_t":3,"cd_l":5}
                    all_results.append(r)
                    if count % 100 == 0: print(f"    {count}/{total}...")

    print(f"  ✅ Phase 2a: {count}")

    # Top 10 by WR
    wr_top = sorted([r for r in all_results if r["trades"]>=15], key=lambda x: (x["wr"], x["pf"]), reverse=True)[:10]
    print(f"\n  Top 5 WR from Phase 2a:")
    for i,r in enumerate(wr_top[:5]):
        p = r["params"]
        print(f"    {i+1}. WR:{r['wr']}% PF:{r['pf']} SL×{p['sl']} Conf≥{p['conf']} RR:{p['rr']} TP1:{p['tp1']}R {r['trades']}t ${r['pnl']:+.0f}")

    # Phase 2b: Fine-tune top 5 with partial/cooldown
    print(f"\n  Phase 2b: Fine-tune top 5 with partial/cooldown...")
    count2 = 0
    for base in wr_top[:5]:
        bp = base["params"]
        for partial in partials:
            for cd_t in cd_trades:
                for cd_l in cd_losses:
                    count2 += 1
                    cfg = Config()
                    cfg.ATR_SL_MULTIPLIER = bp["sl"]
                    cfg.MIN_CONFLUENCE = bp["conf"]
                    cfg.RR_RATIO = bp["rr"]
                    cfg.TP1_RR = bp["tp1"]
                    cfg.PARTIAL_PERCENT = partial
                    cfg.TRADE_COOLDOWN_BARS = cd_t
                    cfg.LOSS_COOLDOWN_BARS = cd_l
                    r = run_backtest(df, cfg, sh, sl)
                    r["params"] = {**bp, "partial":partial, "cd_t":cd_t, "cd_l":cd_l}
                    all_results.append(r)
                    if count2 % 50 == 0: print(f"    {count2}...")

    print(f"  ✅ Phase 2b: {count2}")

    # Phase 3: Signal strength tuning on best combos
    print(f"\n  📊 Phase 3: Signal sensitivity tuning")
    best_wr = sorted([r for r in all_results if r["trades"]>=15 and r["pf"]>=1.0], key=lambda x: x["wr"], reverse=True)[:3]
    count3 = 0

    for base in best_wr:
        bp = base["params"]
        for sweep_min in [0.15, 0.20, 0.25, 0.35]:
            for fvg_min in [0.15, 0.20, 0.25, 0.35]:
                for dbl_tol in [0.2, 0.3, 0.4, 0.5]:
                    for adx_th in [20, 25, 30]:
                        for adx_block in [0, 15, 20]:
                            count3 += 1
                            cfg = Config()
                            cfg.ATR_SL_MULTIPLIER = bp["sl"]
                            cfg.MIN_CONFLUENCE = bp["conf"]
                            cfg.RR_RATIO = bp["rr"]
                            cfg.TP1_RR = bp["tp1"]
                            cfg.PARTIAL_PERCENT = bp.get("partial", 0.50)
                            cfg.TRADE_COOLDOWN_BARS = bp.get("cd_t", 3)
                            cfg.LOSS_COOLDOWN_BARS = bp.get("cd_l", 5)
                            cfg.SWEEP_WICK_ATR_MIN = sweep_min
                            cfg.FVG_MIN_ATR = fvg_min
                            cfg.DOUBLE_TOLERANCE_ATR = dbl_tol
                            cfg.ADX_THRESHOLD = adx_th
                            cfg.ADX_BLOCK_BELOW = adx_block
                            r = run_backtest(df, cfg, sh, sl)
                            r["params"] = {**bp, "sweep_min":sweep_min, "fvg_min":fvg_min,
                                           "dbl_tol":dbl_tol, "adx_th":adx_th, "adx_block":adx_block}
                            all_results.append(r)
                            if count3 % 100 == 0: print(f"    {count3}...")

    print(f"  ✅ Phase 3: {count3}")

    # Results
    viable = [r for r in all_results if r["trades"]>=15 and r["pf"]>=1.0]
    for r in viable:
        wr_bonus = max(0, r["wr"]-55)*0.8
        r["score"] = round((r["wr"]/100)*r["pf"]*math.sqrt(r["trades"])*(1-r["dd"]/100)+wr_bonus, 2)

    # Sort by WR first
    by_wr = sorted(viable, key=lambda x: (x["wr"], x["pf"]), reverse=True)
    by_score = sorted(viable, key=lambda x: x["score"], reverse=True)

    above_65 = [r for r in viable if r["wr"] >= 65]
    above_63 = [r for r in viable if r["wr"] >= 63]
    above_60 = [r for r in viable if r["wr"] >= 60]

    print(f"\n{'='*60}")
    print(f"  🏆 RESULTS")
    print(f"  Total tested: {len(all_results)} | Viable: {len(viable)}")
    print(f"  65%+ WR: {len(above_65)} | 63%+ WR: {len(above_63)} | 60%+ WR: {len(above_60)}")
    print(f"{'='*60}")

    # Show highest WR results
    print(f"\n  🎯 TOP 10 HIGHEST WIN RATE:")
    for i, r in enumerate(by_wr[:10]):
        p = r["params"]
        print(f"""
  #{i+1} — WR: {r['wr']}% {'⭐⭐' if r['wr']>=65 else '⭐' if r['wr']>=63 else ''}
  ├── SL: ATR×{p['sl']} | Conf≥{p['conf']} | RR: 1:{p['rr']} | TP1: {p['tp1']}R
  ├── Partial: {p.get('partial',0.50)*100:.0f}% | CD: {p.get('cd_t',3)}/{p.get('cd_l',5)}
  ├── Trades: {r['trades']} | PF: {r['pf']} | PnL: ${r['pnl']:+,.2f}
  └── DD: {r['dd']:.1f}% | Return: {r['return_pct']:+.2f}%""")

    # Show best balanced (score)
    print(f"\n  ⚖️ TOP 5 BEST BALANCED (WR × PF × trades):")
    for i, r in enumerate(by_score[:5]):
        p = r["params"]
        print(f"""
  #{i+1} — Score: {r['score']} | WR: {r['wr']}%
  ├── SL: ATR×{p['sl']} | Conf≥{p['conf']} | RR: 1:{p['rr']} | TP1: {p['tp1']}R
  ├── Trades: {r['trades']} | PF: {r['pf']} | PnL: ${r['pnl']:+,.2f}
  └── DD: {r['dd']:.1f}%""")

    # Recommendation
    best = by_wr[0] if by_wr else by_score[0]
    bp = best["params"]

    improvement = best["wr"] - baseline["wr"]
    print(f"""
{'='*60}
  ⭐ AANBEVOLEN SETTINGS:
{'='*60}
  ATR_SL_MULTIPLIER = {bp['sl']}
  RR_RATIO = {bp['rr']}
  TP1_RR = {bp['tp1']}
  PARTIAL_PERCENT = {bp.get('partial',0.50)}
  MIN_CONFLUENCE = {bp['conf']}
  TRADE_COOLDOWN = {bp.get('cd_t',3)}
  LOSS_COOLDOWN = {bp.get('cd_l',5)}

  VS v1.4 BASELINE:
  WR: {baseline['wr']}% → {best['wr']}% ({improvement:+.1f}%)
  PF: {baseline['pf']} → {best['pf']}
  PnL: ${baseline['pnl']:+.0f} → ${best['pnl']:+.0f}
  Trades: {baseline['trades']} → {best['trades']}
  DD: {baseline['dd']:.1f}% → {best['dd']:.1f}%

  {'✅ BETER DAN v1.4!' if improvement > 0 and best['pf'] >= 1.0 else '⚠️ v1.4 is nog steeds beter — houd huidige settings.'}
{'='*60}""")

    output = {
        "baseline": {"wr":baseline["wr"],"pf":baseline["pf"],"pnl":baseline["pnl"],"trades":baseline["trades"]},
        "top_10_wr": [{"rank":i+1,"params":r["params"],"wr":r["wr"],"pf":r["pf"],"pnl":r["pnl"],"trades":r["trades"],"dd":r["dd"]} for i,r in enumerate(by_wr[:10])],
        "top_5_balanced": [{"rank":i+1,"params":r["params"],"wr":r["wr"],"pf":r["pf"],"pnl":r["pnl"],"trades":r["trades"]} for i,r in enumerate(by_score[:5])],
        "total_tested": len(all_results),
        "above_65wr": len(above_65), "above_63wr": len(above_63), "above_60wr": len(above_60),
    }
    with open("optimization_v5_results.json","w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  📁 Saved: optimization_v5_results.json")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     XAUUSD GOLD SCALPER — OPTIMIZER v5.0                    ║
║     🎯 Push WR above 63.2% (v1.4 baseline)                 ║
║     Testing: SL, RR, TP1, Confluence, Signal sensitivity    ║
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
