# ou_on_mr_regimes.py
# Run OU mean-reversion strategy only on dates classified as Mean-Reverting
# by the forward-looking regime model.

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

from trading_strategy import MeanReversionStrategy

# -----------------------------
# Config
# -----------------------------
TICKER   = "CAD=X"
PERIOD   = "6mo"
INTERVAL = "1d"

WINDOW   = 20      # forward window used to classify regimes
PHI_THR  = 0.05    # AR(1) phi threshold for TF vs MR


# -----------------------------
# Helper: forward-looking regime
# (same logic as in forward.py)
# -----------------------------
def forward_regime(ret, roll_win, phi_thr):
    """Forward-looking TF/MR regime series for a given window size."""
    phis, labels, dates = [], [], []

    for i in range(roll_win, len(ret)):
        window = ret.iloc[i - roll_win : i]

        # AR(1) on returns in that window
        y = window.iloc[1:].values
        x = window.iloc[:-1].values.reshape(-1, 1)
        x = sm.add_constant(x)
        phi = sm.OLS(y, x).fit().params[1]

        date_t = window.index[-1]
        dates.append(date_t)
        phis.append(phi)

        if phi > phi_thr:
            labels.append(1)      # Trend-Following
        elif phi < -phi_thr:
            labels.append(-1)     # Mean-Reverting
        else:
            labels.append(0)      # Neutral

    regime_t = pd.Series(labels, index=pd.DatetimeIndex(dates))
    regime_fwd = regime_t.shift(1).dropna()   # decision at t, used at t+1

    return regime_fwd


# -----------------------------
# 1) Download price data
# -----------------------------
df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    progress=False,
    auto_adjust=False,
)

px = df["Close"].dropna()
log_px = np.log(px)
ret = log_px.diff().dropna()

# -----------------------------
# 2) Get MR / TF regimes
# -----------------------------
regime = forward_regime(ret, roll_win=WINDOW, phi_thr=PHI_THR)

# Align regime series with the full price index
regime_full = regime.reindex(px.index).fillna(0).astype(int)

# Extract only the dates labelled as Mean-Reverting (-1)
mr_dates = regime_full[regime_full == -1].index
px_mr = px.loc[mr_dates].values  # numpy array of MR-only prices

print(f"Total points: {len(px)}")
print(f"MR points used for OU strategy: {len(px_mr)}")

# -----------------------------
# 3) Run OU mean-reversion strategy
#     only on MR dates
# -----------------------------
strategy = MeanReversionStrategy(
    estimator_method="mle",
    threshold_sigma=2.0,  # you can tune this
)

results = strategy.backtest(
    px_mr,
    dt=1.0,
    initial_capital=10000,
    stop_loss_pct=0.20,
    exit_at_mean=True,
    allow_short=True,
)

print("\n=== OU-on-MR Regimes Results ===")
print(f"Total return: {results['total_return']:.2%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Max drawdown: {results['max_drawdown']:.2%}")
print(f"# of trades : {results['num_trades']}")

# -----------------------------
# (Optional) Quick equity curve plot
# -----------------------------
pv = results["portfolio_values"]

plt.figure(figsize=(10, 4))
plt.plot(pv)
plt.title("OU Strategy Equity Curve (MR Regimes Only)")
plt.ylabel("Portfolio Value")
plt.xlabel("MR Time Index (compressed)")
plt.tight_layout()
plt.show()
