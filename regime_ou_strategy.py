"""
OU mean-reversion strategy gated by your forward-looking TF/MR regime model.

1. Download CAD=X (6M daily).
2. Compute log returns and run the forward_regime() logic from
   usd_cad_window_experiment.py to classify TF / MR / neutral.
3. Map -1 -> "MR", 1 -> "TF", 0 -> None, align to price index.
4. Fit a single OU process on the full price level series.
5. Trade OU (1 unit long/short) ONLY when regime == "MR"; stay flat otherwise.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

from ou_estimator import OUEstimator


# -----------------------------
# Config
# -----------------------------
TICKER   = "CAD=X"
PERIOD   = "6mo"
INTERVAL = "1d"

ROLL_WINDOW = 20      # same as your usd_cad_window_experiment default (can tweak)
PHI_THR     = 0.05    # phi > +thr -> TF, phi < -thr -> MR, else neutral

THRESH_SIGMA = 2.0    # OU entry threshold in stationary sigma units
DT           = 1.0    # 1 trading day per step
INITIAL_CAP  = 100_000.0


# -----------------------------
# Regime model (your logic)
# -----------------------------
def forward_regime(ret: pd.Series, roll_win: int, phi_thr: float) -> pd.Series:
    """
    Forward-looking TF/MR regime series for a given window size.

    Same logic as in usd_cad_window_experiment.py:
      - Run AR(1) on rolling window of RETURNS.
      - phi > phil_thr   -> 1  (TF)
      - phi < -phi_thr   -> -1 (MR)
      - else             -> 0  (neutral)
      - Then shift by 1 so decision at t is used at t+1.
    """
    phis, labels, dates = [], [], []

    for i in range(roll_win, len(ret)):
        window = ret.iloc[i - roll_win : i]

        y = window.iloc[1:].values
        x = window.iloc[:-1].values.reshape(-1, 1)
        x = sm.add_constant(x)

        phi = sm.OLS(y, x).fit().params[1]

        date_t = window.index[-1]
        dates.append(date_t)
        phis.append(phi)

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
                        roll_win: int = ROLL_WINDOW,
                        phi_thr: float = PHI_THR) -> pd.Series:
    """
    Take price levels, compute log returns, run forward_regime(),
    then map -1/0/1 to "MR"/None/"TF" and align with price index.
    """
    px = price.dropna()
    log_px = np.log(px)
    ret = log_px.diff().dropna()

    regime_numeric = forward_regime(ret, roll_win=roll_win, phi_thr=phi_thr)

    # numeric -> string labels
    mapping = {-1: "MR", 0: None, 1: "TF"}
    regime_str = regime_numeric.map(mapping)

    # align to full price index; early dates will be NaN/None
    regime_str = regime_str.reindex(px.index)

    return regime_str


# -----------------------------
# OU strategy (MR-only)
# -----------------------------
def run_ou_in_mr_regimes(price: pd.Series,
                         regimes: pd.Series,
                         dt: float = DT,
                         k_sigma: float = None,
                         initial_capital: float = INITIAL_CAP) -> dict:
    """
    Simple daily bar backtest:

    - Fit OU params (theta, mu, sigma) on the FULL price series.
    - Compute z-scores using stationary sigma.
    - If k_sigma is None, choose it from the distribution of |z| in MR regimes.
    - For each day t > 0:
        * Equity update from position held over [t-1, t].
        * If regime[t] != "MR": set position[t] = 0 (flat).
        * If regime[t] == "MR":
            - if z > +k_sigma  -> short 1 unit
            - if z < -k_sigma  -> long 1 unit
            - else keep previous position (as long as we're still in MR)
    """
    price = price.dropna()
    regimes = regimes.reindex(price.index)

    # --- fit OU parameters on levels ---
    est = OUEstimator()
    theta, mu, sigma = est.estimate_mle(price.values, dt=dt)
    sigma_stat = np.sqrt(est.stationary_variance())

    if sigma_stat is None or sigma_stat <= 0:
        raise ValueError("Stationary variance is non-positive; OU fit failed.")

    dates = price.index
    px = price.values
    n = len(px)

    # --- compute z-scores for diagnostics + threshold selection ---
    z_all = (px - mu) / sigma_stat
    z_series = pd.Series(z_all, index=dates, name="z")
    print("z-score stats (all days):")
    print(z_series.describe())

    # dynamic threshold if not provided
    if k_sigma is None:
        mr_mask = regimes == "MR"
        z_mr = z_series[mr_mask].dropna()
        if len(z_mr) == 0:
            raise ValueError("No MR days to calibrate threshold on.")
        # e.g. 80th percentile of |z| in MR regimes
        k_sigma = np.quantile(np.abs(z_mr), 0.8)
        print(f"Using dynamic k_sigma based on MR |z| 80th percentile: {k_sigma:.4f}")
    else:
        print(f"Using fixed k_sigma = {k_sigma}")

    pos = np.zeros(n)       # position held over [t, t+1)
    equity = np.zeros(n)
    equity[0] = initial_capital

    for t in range(1, n):
        # realized return from previous position
        ret = pos[t - 1] * (px[t] / px[t - 1] - 1.0)
        equity[t] = equity[t - 1] * (1.0 + ret)

        # decide position for next day, based on today's close
        if regimes.iloc[t] != "MR":
            pos[t] = 0.0
            continue

        z = z_all[t]

        if z > k_sigma:
            pos[t] = -1.0     # short
        elif z < -k_sigma:
            pos[t] = +1.0     # long
        else:
            # keep previous position IF we were already in MR
            pos[t] = pos[t - 1]

    equity_series = pd.Series(equity, index=dates, name="equity")
    daily_ret = equity_series.pct_change().dropna()

    if daily_ret.std() > 0:
        sharpe = np.sqrt(252.0) * daily_ret.mean() / daily_ret.std()
    else:
        sharpe = 0.0

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = float(drawdown.min())

    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0

    stats = {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "sigma_stat": sigma_stat,
        "k_sigma": k_sigma,
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }

    return {
        "equity": equity_series,
        "positions": pd.Series(pos, index=dates, name="position"),
        "regimes": regimes,
        "z": z_series,
        "stats": stats,
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

    # Handle possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # first ticker
        price = close.dropna()
    else:
        price = df["Close"].dropna()

    # set the series name explicitly
    price.name = "price"


    # 2) Build regimes using YOUR forward-looking AR(1) model
    regimes = build_regime_series(price,
                                  roll_win=ROLL_WINDOW,
                                  phi_thr=PHI_THR)

    print("Regime counts:")
    print(regimes.value_counts(dropna=False))

    # 3) OU + MR-only strategy
    results = run_ou_in_mr_regimes(price, regimes, k_sigma=None)


    print("Stats (OU in MR regimes only):")
    for k, v in results["stats"].items():
        print(f"{k}: {v}")

        # -----------------------------
    # Debug table: price, regime, z, position, equity, equity change
    # -----------------------------
    debug = pd.DataFrame({
        "price": price,
        "regime": results["regimes"],
        "z": results["z"],
        "position": results["positions"],
        "equity": results["equity"],
    })

    debug["equity_pct_change"] = debug["equity"].pct_change()

    print("\nDebug view (last 20 rows):")
    print(debug.tail(20).to_string())


    # 4) Plot price + equity + MR/TF markers
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(price.index, price.values, label="Price", alpha=0.5)
    ax1.set_ylabel("Price")

    mr_mask = regimes == "MR"
    tf_mask = regimes == "TF"

    ax1.scatter(price.index[mr_mask], price[mr_mask],
                s=20, marker="o", label="MR")
    ax1.scatter(price.index[tf_mask], price[tf_mask],
                s=20, marker="x", label="TF")

    ax2 = ax1.twinx()
    ax2.plot(results["equity"].index, results["equity"].values,
             linestyle="--", label="Equity (MR-only OU)", alpha=0.8)
    ax2.set_ylabel("Equity")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(f"{TICKER} – OU Mean Reversion Only in Forward-Looking MR Regimes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
