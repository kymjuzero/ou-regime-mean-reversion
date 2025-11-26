# Ornstein-Uhlenbeck Process: Mathematical Derivation

**By Josephina Kim (JK)**

## Overview

The Ornstein-Uhlenbeck (OU) process is a stochastic differential equation that models mean-reverting behavior:

```
dX_t = θ(μ - X_t)dt + σdW_t
```

Where:
- **X_t**: Value of the process at time t
- **θ** (theta): Mean reversion speed (strength)
- **μ** (mu): Long-term mean level
- **σ** (sigma): Volatility
- **dW_t**: Wiener process increment (Brownian motion)

## Parameter Estimation Strategy

### 1. Theta (θ) - Mean Reversion Speed

**Derivation from Autocorrelation:**

The discretized OU process has the form:
```
X_{t+1} = μ + (X_t - μ)e^{-θΔt} + ε_t
```

Where ε_t is normally distributed with variance:
```
Var(ε_t) = σ²(1 - e^{-2θΔt}) / (2θ)
```

The lag-1 autocorrelation ρ is:
```
ρ = Corr(X_t, X_{t+1}) = e^{-θΔt}
```

Solving for θ:
```
θ = -ln(ρ) / Δt
```

**Implementation:**
1. Calculate lag-1 autocorrelation: ρ = Corr(X_t, X_{t+1})
2. Compute theta: θ = -ln(ρ) / Δt
3. Ensure θ > 0 (mean reversion requires positive theta)

### 2. Mu (μ) - Long-term Mean

**Derivation:**

From the OU process, the long-term mean is the expected value:
```
E[X_t] = μ (as t → ∞)
```

The sample mean is an unbiased estimator:
```
μ̂ = (1/n) Σ X_i
```

**Implementation:**
- Simply compute the sample mean of the data

### 3. Sigma (σ) - Volatility

**Derivation from Maximum Likelihood:**

The discretized process gives:
```
X_{t+1} - X_t = μ(1 - e^{-θΔt}) + (X_t - μ)(1 - e^{-θΔt}) + ε_t
```

Simplifying:
```
X_{t+1} - X_t = μ(1 - e^{-θΔt}) + ε_t
```

Where ε_t ~ N(0, σ²(1 - e^{-2θΔt}) / (2θ))

The maximum likelihood estimator for σ² is:
```
σ² = (2θ / [n(1 - e^{-2θΔt})]) Σ [X_{i+1} - X_i - μ(1 - e^{-θΔt})]²
```

**Implementation:**
1. For each observation i, compute: diff_i = X_{i+1} - X_i - μ(1 - e^{-θΔt})
2. Sum squared differences: Σ diff_i²
3. Calculate: σ² = (2θ × Σ diff_i²) / [n(1 - e^{-2θΔt})]
4. Take square root: σ = √σ²

## Trading Signal Generation

### Strategy Logic

When the price deviates from the mean, it's expected to revert:

- **SELL Signal**: Price > μ + k × σ_stationary
  - Price is above mean → expect downward reversion → SELL

- **BUY Signal**: Price < μ - k × σ_stationary
  - Price is below mean → expect upward reversion → BUY

- **HOLD Signal**: Price within ±k × σ_stationary
  - Price is near mean → no strong signal

### Stationary Variance

The OU process has a stationary distribution with variance:
```
Var_stationary = σ² / (2θ)
```

This is used for signal generation because it represents the long-term variance of the process around the mean.

### Z-Score Calculation

```
z = (X_t - μ) / σ_stationary
```

Where σ_stationary = √(σ² / (2θ))

Trading signals:
- SELL if z > threshold (typically 2.0)
- BUY if z < -threshold (typically -2.0)
- HOLD otherwise

## Half-Life

The half-life represents the expected time for the process to revert halfway to the mean:

```
t_{1/2} = ln(2) / θ
```

This is useful for understanding the speed of mean reversion.

## Mathematical Properties

### Expected Value

Given X_0 = x_0:
```
E[X_t] = μ + (x_0 - μ)e^{-θt}
```

### Variance

```
Var(X_t) = (σ² / (2θ)) × (1 - e^{-2θt})
```

As t → ∞:
```
Var(X_t) → σ² / (2θ) = stationary variance
```

### Exact Solution

The exact solution to the OU SDE is:
```
X_t = μ + (X_0 - μ)e^{-θt} + σ ∫₀ᵗ e^{-θ(t-s)} dW_s
```

This is used for exact simulation of the process.

