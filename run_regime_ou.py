"""
run_regime_ou.py

Regime-gated OU mean-reversion strategy with:
- Forward-looking AR(1) regime model (MR/TF/neutral)
- 3-signal MR gate before trading
- Rolling OU estimation window
- Dynamic OU band based on stationary sigma + |z|-quantile
- Trade log + trade-level stats
- Buy-and-hold baseline for comparison
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

from ou_estimator import OUEstimator


# -----------------------------
# CONFIG
# -----------------------------
TICKER   = "CAD=X"
PERIOD   = "6mo"
INTERVAL = "1d"

# Regime detection config
ROLL_WINDOW_REGIME = 20       # AR(1) window on returns
PHI_THR            = 0.05     # phi > +thr: TF, phi < -thr: MR

# OU estimation / trading config
OU_WINDOW          = 40       # days of price for each OU fit
MR_STREAK_MIN      = 3        # need this many consecutive MR signals before trading
Z_QUANTILE         = 0.8      # band based on 80th percentile of |z| in OU window
ENTRY_MULT         = 1.0      # entry at k * sigma_stat
EXIT_MULT          = 0.5      # exit at (k * EXIT_MULT) * sigma_stat

DT                 = 1.0      # time step (1 trading day)
INITIAL_CAP        = 100_000.0
POSITION_UNITS     = 1.0      # number of "units" (1 = 1 CAD notionally; scaling doesn't change Sharpe)


# -----------------------------
# Regime model (same spirit as your usd_cad_window_experiment.py)
# -----------------------------
def forward_regime(ret: pd.Series, roll_win: int, phi_thr: float) -> pd.Series:
    """
    Forward-looking TF/MR regime series for a given window size.

      - Run AR(1) on rolling window of RETURNS.
      - phi > +phi_thr  -> 1  (TF)
      - phi < -phi_thr  -> -1 (MR)
      - else            -> 0  (neutral)
      - Then shift by 1 so decision at t is used at t+1.
    """
    labels, dates = [], []

    for i in range(roll_win, len(ret)):
        window = ret.iloc[i - roll_win : i]

        y = window.iloc[1:].values
        x = window.iloc[:-1].values.reshape(-1, 1)
        x = sm.add_constant(x)

        phi = sm.OLS(y, x).fit().params[1]

        date_t = window.index[-1]
        dates.append(date_t)

        if phi > phi_thr:
            labels.append(1)      # TF
        elif phi < -phi_thr:
            labels.append(-1)     # MR
        else:
            labels.append(0)      # neutral

    regime_t = pd.Series(labels, index=pd.DatetimeIndex(dates))
    regime_fwd = regime_t.shift(1).dropna()   # decision at t, used at t+1
    return regime_fwd


def build_regime_series(price: pd.Series,
                        roll_win: int = ROLL_WINDOW_REGIME,
                        phi_thr: float = PHI_THR):
    """
    From price levels:
      - compute log returns
      - run forward_regime()
      - reindex to price index
      - return both numeric and string labels
    """
    px = price.dropna()
    log_px = np.log(px)
    ret = log_px.diff().dropna()

    regime_numeric = forward_regime(ret, roll_win=roll_win, phi_thr=phi_thr)
    regime_numeric = regime_numeric.reindex(px.index)

    mapping = {-1: "MR", 0: None, 1: "TF"}
    regime_str = regime_numeric.map(mapping)

    return regime_numeric, regime_str


# -----------------------------
# Buy & Hold baseline
# -----------------------------
def buy_and_hold_stats(price: pd.Series):
    px = price.dropna()
    dates = px.index
    px_vals = px.values
    n = len(px_vals)

    equity = np.zeros(n)
    equity[0] = INITIAL_CAP

    for t in range(1, n):
        ret = POSITION_UNITS * (px_vals[t] / px_vals[t - 1] - 1.0)
        equity[t] = equity[t - 1] * (1.0 + ret)

    eq = pd.Series(equity, index=dates, name="BH_equity")
    daily_ret = eq.pct_change().dropna()

    if daily_ret.std() > 0:
        sharpe = np.sqrt(252.0) * daily_ret.mean() / daily_ret.std()
    else:
        sharpe = 0.0

    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = float(drawdown.min())
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0

    return {
        "equity": eq,
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# -----------------------------
# OU trading logic (improved)
# -----------------------------
def run_signal_ou_strategy(price: pd.Series,
                           regime_numeric: pd.Series,
                           initial_capital: float = INITIAL_CAP):
    """
    Improved 3-signal OU strategy:

    - Wait for MR_STREAK_MIN consecutive MR signals.
    - On MR days with enough history (OU_WINDOW), fit OU to last OU_WINDOW closes.
    - Use stationary sigma (sigma_stat) and |z| quantile to define dynamic band:
        k = quantile(|z_window|, Z_QUANTILE)

        Entry:
          price >= mu + ENTRY_MULT * k * sigma_stat -> short
          price <= mu - ENTRY_MULT * k * sigma_stat -> long

        Exit:
          For short, price <= mu + EXIT_MULT * k * sigma_stat
          For long,  price >= mu - EXIT_MULT * k * sigma_stat

    - Flatten immediately when regime is not MR or MR streak is broken.
    - Track trade log and trade-level stats.
    """
    price = price.dropna()
    regime_numeric = regime_numeric.reindex(price.index)

    dates = price.index
    px = price.values
    n = len(px)

    pos = np.zeros(n)                 # position (units) held over [t, t+1]
    equity = np.zeros(n)
    equity[0] = initial_capital

    mu_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    mr_streak_arr = np.zeros(n, dtype=int)
    k_arr = np.full(n, np.nan)

    est = OUEstimator()

    mr_streak = 0

    # trade tracking
    in_trade = False
    entry_idx = None
    entry_price = None
    entry_side = None  # +1 long, -1 short
    trade_log = []

    for t in range(1, n):
        # --- update equity from yesterday's position ---
        ret = pos[t - 1] * (px[t] / px[t - 1] - 1.0)
        equity[t] = equity[t - 1] * (1.0 + ret)

        # --- update MR streak based on today's regime ---
        reg_val = regime_numeric.iloc[t]
        if reg_val == -1:
            mr_streak += 1
        else:
            mr_streak = 0
        mr_streak_arr[t] = mr_streak

        # default: carry previous position
        pos[t] = pos[t - 1]

        # gating: if not MR / streak too short / not enough data -> flatten & skip
        if (reg_val != -1) or (mr_streak < MR_STREAK_MIN) or (t < OU_WINDOW - 1):
            # if we were in a trade, close it here at today's price
            if in_trade:
                exit_price = px[t]
                pnl = entry_side * POSITION_UNITS * (exit_price / entry_price - 1.0)
                trade_log.append({
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[t],
                    "side": "LONG" if entry_side > 0 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "holding_period": t - entry_idx,
                })
                in_trade = False
                entry_idx = entry_price = entry_side = None

            pos[t] = 0.0
            continue

        # --- fit OU on last OU_WINDOW closings ---
        start = t - OU_WINDOW + 1
        window_prices = px[start : t + 1]

        try:
            theta, mu_t, sigma_t = est.estimate_mle(window_prices, dt=DT)
        except Exception:
            pos[t] = 0.0
            continue

        if sigma_t <= 0 or theta <= 0:
            pos[t] = 0.0
            continue

        # store raw OU params for debugging
        mu_arr[t] = mu_t
        sigma_arr[t] = sigma_t

        # --- stationary sigma & dynamic band from |z| quantile ---
        sigma_stat_t = np.sqrt(est.stationary_variance())
        if not np.isfinite(sigma_stat_t) or sigma_stat_t <= 0:
            pos[t] = 0.0
            continue

        z_window = (window_prices - mu_t) / sigma_stat_t
        k = np.quantile(np.abs(z_window), Z_QUANTILE)
        if k <= 0:
            pos[t] = 0.0
            continue
        k_arr[t] = k

        price_t = px[t]

        upper_entry = mu_t + ENTRY_MULT * k * sigma_stat_t
        lower_entry = mu_t - ENTRY_MULT * k * sigma_stat_t
        upper_exit  = mu_t + EXIT_MULT * k * sigma_stat_t
        lower_exit  = mu_t - EXIT_MULT * k * sigma_stat_t

        # --- trading rules ---

        # If currently flat: look for new entries
        if not in_trade:
            if price_t >= upper_entry:
                pos[t] = -POSITION_UNITS
                in_trade = True
                entry_idx = t
                entry_price = price_t
                entry_side = -1
            elif price_t <= lower_entry:
                pos[t] = +POSITION_UNITS
                in_trade = True
                entry_idx = t
                entry_price = price_t
                entry_side = +1
            else:
                pos[t] = 0.0

        # If currently in a trade: check for exit condition
        else:
            # short
            if entry_side == -1:
                if price_t <= upper_exit:
                    # exit short
                    pos[t] = 0.0
                    exit_price = price_t
                    pnl = entry_side * POSITION_UNITS * (exit_price / entry_price - 1.0)
                    trade_log.append({
                        "entry_date": dates[entry_idx],
                        "exit_date": dates[t],
                        "side": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "holding_period": t - entry_idx,
                    })
                    in_trade = False
                    entry_idx = entry_price = entry_side = None
                else:
                    pos[t] = -POSITION_UNITS

            # long
            elif entry_side == +1:
                if price_t >= lower_exit:
                    # exit long
                    pos[t] = 0.0
                    exit_price = price_t
                    pnl = entry_side * POSITION_UNITS * (exit_price / entry_price - 1.0)
                    trade_log.append({
                        "entry_date": dates[entry_idx],
                        "exit_date": dates[t],
                        "side": "LONG",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "holding_period": t - entry_idx,
                    })
                    in_trade = False
                    entry_idx = entry_price = entry_side = None
                else:
                    pos[t] = +POSITION_UNITS

    # if still in trade at the very end, close at last price
    if in_trade and entry_idx is not None:
        exit_price = px[-1]
        pnl = entry_side * POSITION_UNITS * (exit_price / entry_price - 1.0)
        trade_log.append({
            "entry_date": dates[entry_idx],
            "exit_date": dates[-1],
            "side": "LONG" if entry_side > 0 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "holding_period": n - 1 - entry_idx,
        })

    # ----------------- performance stats -----------------
    equity_series = pd.Series(equity, index=dates, name="equity")
    pos_series = pd.Series(pos, index=dates, name="position")
    mu_series = pd.Series(mu_arr, index=dates, name="mu")
    sigma_series = pd.Series(sigma_arr, index=dates, name="sigma")
    mr_streak_series = pd.Series(mr_streak_arr, index=dates, name="mr_streak")
    k_series = pd.Series(k_arr, index=dates, name="k")

    daily_ret = equity_series.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = np.sqrt(252.0) * daily_ret.mean() / daily_ret.std()
    else:
        sharpe = 0.0

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = float(drawdown.min())
    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0

    # trade-level stats
    trades_df = pd.DataFrame(trade_log)
    if len(trades_df) > 0:
        win_rate = (trades_df["pnl"] > 0).mean()
        avg_pnl = trades_df["pnl"].mean()
        avg_hold = trades_df["holding_period"].mean()
    else:
        win_rate = avg_pnl = avg_hold = 0.0

    stats = {
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": int(len(trades_df)),
        "win_rate": win_rate,
        "avg_pnl_per_trade": avg_pnl,
        "avg_holding_period": avg_hold,
    }

    return {
        "equity": equity_series,
        "positions": pos_series,
        "mu": mu_series,
        "sigma": sigma_series,
        "k": k_series,
        "mr_streak": mr_streak_series,
        "stats": stats,
        "trades": trades_df,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Download data
    df = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        progress=False,
        auto_adjust=False,
    )

    # handle possible MultiIndex from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        price = close.dropna()
    else:
        price = df["Close"].dropna()
    price.name = "price"

    # 2) Regime series
    regime_numeric, regime_str = build_regime_series(price)
    print("Regime counts:")
    print(regime_str.value_counts(dropna=False))

    # 3) Strategy run
    results = run_signal_ou_strategy(price, regime_numeric)

    print("\nStats (3-signal OU MR strategy, improved):")
    for k, v in results["stats"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    # 4) Baseline
    bh = buy_and_hold_stats(price)
    print("\nBuy & Hold baseline on CAD=X:")
    print(f"total_return: {bh['total_return']:.6f}")
    print(f"sharpe:       {bh['sharpe']:.6f}")
    print(f"max_drawdown: {bh['max_drawdown']:.6f}")

    # 5) Trade log (last few trades)
    if len(results["trades"]) > 0:
        print("\nLast trades:")
        print(results["trades"].tail(10).to_string(index=False))
    else:
        print("\nNo trades executed.")

    # 6) Optional: debug table (last 20 rows)
    debug = pd.DataFrame({
        "price": price,
        "regime_num": regime_numeric,
        "regime": regime_str,
        "mu": results["mu"],
        "sigma": results["sigma"],
        "k": results["k"],
        "mr_streak": results["mr_streak"],
        "position": results["positions"],
        "equity": results["equity"],
    })
    debug["equity_pct_change"] = debug["equity"].pct_change()
    print("\nDebug view (last 20 rows):")
    print(debug.tail(20).to_string())

    # 7) Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(price.index, price.values, label="Price", alpha=0.5)
    ax1.set_ylabel("Price")

    mr_mask = regime_str == "MR"
    tf_mask = regime_str == "TF"
    ax1.scatter(price.index[mr_mask], price[mr_mask], s=20, marker="o", label="MR")
    ax1.scatter(price.index[tf_mask], price[tf_mask], s=20, marker="x", label="TF")

    ax2 = ax1.twinx()
    ax2.plot(results["equity"].index, results["equity"].values,
             linestyle="--", label="Equity (3-signal OU)", alpha=0.8)
    ax2.plot(bh["equity"].index, bh["equity"].values,
             linestyle=":", label="Equity (Buy & Hold)", alpha=0.8)
    ax2.set_ylabel("Equity")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(f"{TICKER} – 3-Signal OU Mean Reversion in MR Regimes (Improved)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
