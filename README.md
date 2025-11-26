# Ornstein-Uhlenbeck Process - Mean Reversion Trading Strategy

Implementation of the Ornstein-Uhlenbeck (OU) process for mean reversion trading with parameter estimation and trading signal generation. Based on [QuestDB's OU Process Guide](https://questdb.com/glossary/ornstein-uhlenbeck-process-for-mean-reversion/).

## Overview

The OU process models mean-reverting behavior:

```
dX_t = θ(μ - X_t)dt + σdW_t
```

Where:
- **θ** (theta): Mean reversion speed
- **μ** (mu): Long-term mean level
- **σ** (sigma): Volatility
- **dW_t**: Wiener process increment

## Strategy Logic

Trading signals based on deviations from mean using stationary variance:

- **SELL Signal**: price > μ + k × σ_stationary
- **BUY Signal**: price < μ - k × σ_stationary
- **HOLD Signal**: price within ±k × σ_stationary

Where:
- σ_stationary = √(σ² / (2θ))
- k = threshold multiplier (default: 2.0)

## Files

- `ou_estimator.py`: Parameter estimation (MLE, regression, OLS methods)
- `ou_process.py`: OU process simulation
- `trading_strategy.py`: Trading strategy with signal generation
- `parameter_analysis.py`: Parameter estimation with step-by-step calculations
- `DERIVATION.md`: Mathematical derivation

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.7+.

## Usage

### Complete Demo

```bash
python3 parameter_analysis.py
```

Generates synthetic OU process data, estimates parameters, compares estimation methods, builds trading strategy, generates signals, runs backtest, and saves visualization to `ou_analysis.png`.

### Parameter Analysis

```bash
python3 parameter_analysis.py
```

Shows step-by-step parameter estimation calculations.

## Parameter Estimation

### Theta (θ)

Formula: θ = -ln(ρ) / Δt

Where ρ is lag-1 autocorrelation: ρ = Corr(X_t, X_{t+1})

### Mu (μ)

Formula: μ = (1/n) Σ X_i

Sample mean of the data.

### Sigma (σ)

Formula: σ² = (2θ / [n(1-e^(-2θΔt))]) Σ [X_{i+1} - X_i - μ(1-e^(-θΔt))]²

Maximum Likelihood Estimator.

## Parameter Estimation Methods

1. **Maximum Likelihood Estimation (MLE)**: Uses autocorrelation and closed-form formulas
2. **Regression Method**: Linear regression on lagged data
3. **Ordinary Least Squares (OLS)**: Direct OLS estimation on discretized SDE

## Trading Signal Generation

### Stationary Variance

```
σ_stationary = √(σ² / (2θ))
z = (X_t - μ) / σ_stationary
```

### Signal Rules

- **SELL**: z > 2.0
- **BUY**: z < -2.0
- **HOLD**: -2.0 ≤ z ≤ 2.0

## Features

- QuestDB methodology implementation
- Multiple parameter estimation methods
- Step-by-step parameter calculations
- Trading signal generation using stationary variance
- Position sizing based on deviation
- Stop-loss calculation using stationary variance
- Backtesting framework
- Visualizations (6-panel analysis)
- Mathematical derivation documentation

## Output

Running `parameter_analysis.py` produces:

1. Parameter estimation analysis with autocorrelation, theta calculation, mu estimation, sigma calculation, and stationary distribution properties
2. Estimation method comparison (MLE, Regression, OLS)
3. Trading signal examples with z-score calculations
4. Visualization saved to `ou_analysis.png` with price series, trading signals, deviation analysis, z-score, backtest performance, and autocorrelation function

## Mathematical Properties

### Half-Life

Expected time to revert halfway to mean:
```
t_{1/2} = ln(2) / θ
```

### Stationary Variance

Long-term variance around mean:
```
Var_stationary = σ² / (2θ)
```

Used for signal generation as it represents the long-term distribution of the process.

## Documentation

See `DERIVATION.md` for complete mathematical derivation.

## Quick Start

1. Install: `pip install -r requirements.txt`
2. Run: `python3 parameter_analysis.py`

## Troubleshooting

**"ModuleNotFoundError"**: Run `pip install -r requirements.txt`

**"python3: command not found"**: Use `python` instead

**Visualization not showing**: Check that `ou_analysis.png` was created in the same directory
