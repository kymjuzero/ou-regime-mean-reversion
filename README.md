Ornstein-Uhlenbeck Process - Mean Reversion Trading Strategy

Implementation of the Ornstein-Uhlenbeck (OU) process for mean reversion trading with parameter estimation and trading signal generation. Based on QuestDB's OU Process Guide
.

Overview

The OU process models mean-reverting behavior:

dX_t = θ(μ - X_t)dt + σdW_t


Where:

θ (theta): Mean reversion speed

μ (mu): Long-term mean level

σ (sigma): Volatility

dW_t: Wiener process increment

Strategy Logic

Trading signals based on deviations from mean using stationary variance:

SELL Signal: price > μ + k × σ_stationary

BUY Signal: price < μ - k × σ_stationary

HOLD Signal: price within ±k × σ_stationary

Where:

σ_stationary = √(σ² / (2θ))

k = threshold multiplier (default: 2.0)

Files

ou_estimator.py: Parameter estimation (MLE, regression, OLS)

ou_process.py: OU simulation

trading_strategy.py: OU trading signal generator

parameter_analysis.py: Full OU workflow

DERIVATION.md: Mathematical derivation

Regime Switching Model – Trend-Following vs Mean-Reverting Classification

The regime-switching model identifies whether the market is currently trend-following (TF) or mean-reverting (MR) using a rolling AR(1) model on log returns. This regime information is later used to restrict OU trading to true mean-reverting environments.

Overview

We estimate a rolling AR(1) process:

r_t = α + ϕ r_{t-1} + ε_t


Interpretation of the AR(1) coefficient:

ϕ > 0 → Trend-Following (TF)

ϕ < 0 → Mean-Reverting (MR)

|ϕ| small → Neutral

Default configuration:

Rolling window: 20 days

Threshold: ±0.05

A forward shift is applied so that the regime detected at time t is used for trading at t+1, preventing look-ahead bias.

Files

forward.py: Rolling AR(1) regime classification

Computes φ for each window

Labels: MR, TF, or Neutral

Returns a regime time series suitable for trading

Usage
python3 forward.py


Produces:

AR(1) coefficient series

Vector of MR/TF/Neutral labels

Regime plot overlayed on price (if enabled)

OU Process Applied to Mean-Reverting Regimes

A more robust trading approach is to apply the OU mean-reversion strategy only during MR regimes, as detected by the AR(1) model. This prevents OU trades in trending environments where mean reversion is unlikely to hold.

Overview

The OU strategy is activated only when:

MR_streak ≥ 3


This ensures the market has shown consistent mean-reversion behavior before entering OU trades.

Methodology

Run regime detection (forward.py)

Track MR streak length

When MR_streak ≥ 3:

Fit a rolling OU model (40-day default)

Compute μ, σ, θ

Compute stationary σ

Compute z-scores

Use dynamic thresholding:

k = 80th percentile of |z| over the window

Generate trades:

Short when price > μ + kσ

Long when price < μ – kσ

Exit trades when:

Price reverts halfway to μ, or

Market exits MR regime

Files

run_regime_ou.py: OU applied only in MR regimes

Outputs returns, Sharpe, drawdown

Shows a full trade log (entry, exit, side, pnl)

Usage
python3 run_regime_ou.py

Hybrid Strategy – OU in MR Regimes + Long in TF Regimes

This model adapts trading behavior to the market structure:

Use OU mean-reversion during MR periods

Hold a long trend-following position during TF periods

Stay flat during neutral periods

This produces a strategy that performs well in both trending and reverting markets.

Overview

Regime → Action:

MR → OU trading

TF → Always long

Neutral → No position

This hybrid approach generally produces smoother equity curves and better Sharpe ratios.

Example Results (USD/CAD – 6 months)

Hybrid Strategy:

Total return: 2.23%

Sharpe: 1.50

Max drawdown: -1.27%

4 OU trades, 75% win rate

Buy & Hold:

Total return: 2.54%

Sharpe: 1.17

Max drawdown: -2.01%

Files

withTF.py: Hybrid OU(MR) + long(TF) strategy

Includes full backtest statistics

Prints trade log and equity curve

Usage
python3 withTF.py

Project Structure
Mean-Reversion-Strats/
│
├── ou_estimator.py
├── ou_process.py
├── trading_strategy.py
├── parameter_analysis.py
│
├── forward.py
├── run_regime_ou.py
├── withTF.py
│
├── DERIVATION.md
├── requirements.txt
└── README.md

Quick Start

Install packages:

pip install -r requirements.txt


Run pure OU demo:

python3 parameter_analysis.py


Run regime classifier:

python3 forward.py


Run OU only in MR:

python3 run_regime_ou.py


Run hybrid OU + TF:

python3 withTF.py

Troubleshooting

ModuleNotFoundError
Install dependencies:

pip install -r requirements.txt


python3 not found
Use:

python parameter_analysis.py


Plots not saving
Check working directory for PNG outputs.
