# usd_cad_window_experiment.py
# Compare forward-looking TF/MR regimes for different window sizes.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

TICKER   = "CAD=X"
PERIOD   = "6mo"
INTERVAL = "1d"

WINDOWS  = [10, 20, 30]   # window sizes to test
PHI_THR  = 0.05           # same threshold for all for now

# -----------------------------
# Download data
# -----------------------------
df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    progress=False,
    auto_adjust=False
)

px = df["Close"].dropna()
log_px = np.log(px)
ret = log_px.diff().dropna()

def forward_regime(ret, roll_win, phi_thr):
    """Forward-looking TF/MR regime series for a given window size."""
    phis, labels, dates = [], [], []

    for i in range(roll_win, len(ret)):
        window = ret.iloc[i-roll_win:i]

        y = window.iloc[1:].values
        x = window.iloc[:-1].values.reshape(-1, 1)
        x = sm.add_constant(x)
        phi = sm.OLS(y, x).fit().params[1]

        date_t = window.index[-1]
        dates.append(date_t)
        phis.append(phi)

        if phi > phi_thr:
            labels.append(1)     # TF
        elif phi < -phi_thr:
            labels.append(-1)    # MR
        else:
            labels.append(0)     # neutral

    regime_t = pd.Series(labels, index=pd.DatetimeIndex(dates))
    regime_fwd = regime_t.shift(1).dropna()   # decision at t, used at t+1

    return regime_fwd

# -----------------------------
# Plot all windows in one figure
# -----------------------------
fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(12, 4 * len(WINDOWS)), sharex=True)

if len(WINDOWS) == 1:
    axes = [axes]  # make iterable

for ax, W in zip(axes, WINDOWS):
    regime = forward_regime(ret, roll_win=W, phi_thr=PHI_THR)
    px_sub = px.loc[regime.index]

    mr_mask = regime == -1
    tf_mask = regime == 1

    ax.plot(px.index, px.values, alpha=0.3, linewidth=1.0, label="Price")
    ax.scatter(px_sub.index[tf_mask], px_sub[tf_mask], s=20, marker="x",
               label="Trend-Following")
    ax.scatter(px_sub.index[mr_mask], px_sub[mr_mask], s=20, marker="o",
               label="Mean-Reverting")

    ax.set_title(f"Forward-Looking Regimes – {W}-Day Window")
    ax.set_ylabel("Price")
    ax.legend()

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()
