# Ornstein–Uhlenbeck Process & Regime Switching Models  
Combined Mean Reversion, Trend Following, and OU-Based Trading Strategies

This repository contains three major components:

1. **Ornstein–Uhlenbeck (OU) Mean Reversion Model**  
2. **Regime Switching Model (Mean Reversion vs. Trend Following)**  
3. **OU Process executed only inside Mean-Reverting regimes**, with optional **Trend-Following overlay**

---

## 1. Ornstein–Uhlenbeck Process – Mean Reversion Trading Strategy

Implementation of the Ornstein–Uhlenbeck (OU) process for mean reversion trading with parameter estimation and trading signal generation.  
Based on [QuestDB's OU Process Guide](https://questdb.com/glossary/ornstein-uhlenbeck-process-for-mean-reversion/).

### Overview

The OU process models mean-reverting behavior:

```
dX_t = θ(μ - X_t)dt + σdW_t
```

Where:

- **θ (theta)**: Mean reversion speed  
- **μ (mu)**: Long-term mean level  
- **σ (sigma)**: Volatility  
- **dW_t**: Wiener process increment  

### Strategy Logic

Trading signals based on deviations from the mean using stationary variance:

- **SELL Signal**: price > μ + k × σ_stationary  
- **BUY Signal**: price < μ − k × σ_stationary  
- **HOLD Signal**: price within ±k × σ_stationary  

Where:

```
σ_stationary = √(σ² / (2θ))
```

k = threshold multiplier (default: 2.0)

### Files

- `ou_estimator.py`: Parameter estimation (MLE, regression, OLS)  
- `ou_process.py`: OU simulation  
- `trading_strategy.py`: OU trading logic  
- `parameter_analysis.py`: Step-by-step calculations  
- `DERIVATION.md`: Mathematical derivation  

### Parameter Estimation

**Theta (θ)**  
```
θ = -ln(ρ) / Δt
```

**Mu (μ)**  
```
μ = (1/n) Σ X_i
```

**Sigma (σ)** (MLE)  
```
σ² = (2θ / [n(1 - e^(-2θΔt))]) Σ [X_{i+1} - X_i - μ(1 − e^(-θΔt))]²
```

### Stationary Variance

```
σ_stationary = √(σ² / (2θ))
z = (X_t − μ) / σ_stationary
```

### Signal Rules

- **SELL**: z > 2.0  
- **BUY**: z < −2.0  
- **HOLD**: −2.0 ≤ z ≤ 2.0  

### Mathematical Properties

**Half-Life**  
```
t_{1/2} = ln(2) / θ
```

**Stationary Variance**  
```
Var_stationary = σ² / (2θ)
```

---

## 2. Regime Switching Model (AR(1) Mean Reversion vs. Trend Following)

This model classifies market regimes using a rolling AR(1) estimation on log returns.

### Methodology

We estimate a rolling AR(1):

```
r_t = ϕ r_{t-1} + ε_t
```

Interpretation:

- **ϕ < 0 → Mean Reverting (MR)**  
- **ϕ > 0 → Trend Following (TF)**  

Parameters:

- Rolling window: **20 days**  
- Regime label assigned to each date  

Outputs:

- Regime label per day  
- MR and TF visualization on price series  

### File

- `forward.py`: Regime classification + plotting

- <img width="2992" height="1800" alt="image" src="https://github.com/user-attachments/assets/6d9817e4-3c26-41c0-8a44-f98fcfeebe51" />


---

## 3. OU Trading Inside Mean-Reverting Regimes  
(Regime-Filtered OU Trading)

This combines the OU model with regime switching.

**Only run OU trades when at least 3 consecutive MR signals are detected.**

### Entry Logic

**Short Entry (Mean Reversion Sell):**
- MR streak ≥ 3  
- price ≥ μ + σ  

**Long Entry (Mean Reversion Buy):**
- MR streak ≥ 3  
- price ≤ μ − σ  

### Exit Rule

- Short exit: price → μ + 0.5σ  
- Long exit: price → μ − 0.5σ  

### File

- `run_regime_ou.py`: OU model executed only inside MR segments  

---

## 4. Trend-Following Overlay (Hybrid MR + TF Strategy)

When the regime is TF, the strategy can:

- Enter **long**
- Hold until TF regime ends
- Combine with OU MR trading

This produces a **hybrid Mean Reversion + Trend Following system**.

### File

- `withTF.py`: OU in MR regimes + long TF overlay  

---

## Installation

```
pip install -r requirements.txt
```

Requires Python 3.7+.

---

## Usage

### OU Process Analysis

```
python3 parameter_analysis.py
```

### Regime Switching

```
python3 forward.py
```

### OU in MR Regimes

```
python3 run_regime_ou.py
```

### OU in MR + TF Long Strategy

```
python3 withTF.py
```

---

## Troubleshooting

**ModuleNotFoundError**  
```
pip install -r requirements.txt
```

**No trades executing**
- Reduce OU thresholds  
- Reduce MR streak requirement  
- Increase rolling AR window  

**Plots not showing**
- Ensure PNG files save in working directory  

---
