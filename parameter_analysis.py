"""
Parameter Estimation Analysis
By Josephina Kim (JK)

This shows detailed step-by-step parameter estimation calculations.
Perfect for understanding how we derive θ, μ, and σ from the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from ou_estimator import OUEstimator
from trading_strategy import MeanReversionStrategy
from polygon_data import PolygonDataFetcher
import yfinance as yf

def detailed_parameter_analysis(data, dt=1.0, true_params=None):
    estimator = OUEstimator()
    
    print("\n" + "="*70)
    print("DETAILED PARAMETER ESTIMATION ANALYSIS")
    print("="*70)
    
    print("\n1. AUTOCORRELATION ANALYSIS")
    print("-"*70)
    rho = estimator.estimate_autocorrelation(data, lag=1)
    print(f"Lag-1 Autocorrelation (ρ): {rho:.6f}")
    
    if rho > 0 and rho < 1:
        print(f"Mean reversion indicator: ρ < 1 (✓ Mean reverting)")
    else:
        print(f"Mean reversion indicator: ρ = {rho:.6f} (✗ May not be mean reverting)")
    
    print("\n2. THETA (θ) ESTIMATION - Mean Reversion Speed")
    print("-"*70)
    theta = estimator.estimate_theta_from_autocorr(data, dt)
    print("Formula: θ = -ln(ρ) / Δt")
    print(f"Calculation: θ = -ln({rho:.6f}) / {dt}")
    print(f"Estimated Theta (θ): {theta:.6f}")
    
    if true_params:
        theta_error = abs(theta - true_params['theta']) / true_params['theta'] * 100
        print(f"True Theta: {true_params['theta']:.6f}")
        print(f"Estimation Error: {theta_error:.2f}%")
    
    half_life = np.log(2) / theta if theta > 0 else None
    if half_life:
        print(f"Half-life: {half_life:.4f} time steps")
        print(f"Interpretation: Process reverts halfway to mean in {half_life:.2f} steps")
    
    print("\n3. MU (μ) ESTIMATION - Long-term Mean")
    print("-"*70)
    mu = estimator.estimate_mu(data)
    print("Formula: μ = (1/n) Σ X_i")
    print(f"Sample size: {len(data)}")
    print(f"Estimated Mu (μ): {mu:.6f}")
    
    if true_params:
        mu_error = abs(mu - true_params['mu']) / abs(true_params['mu']) * 100
        print(f"True Mu: {true_params['mu']:.6f}")
        print(f"Estimation Error: {mu_error:.2f}%")
    
    print("\n4. SIGMA (σ) ESTIMATION - Volatility")
    print("-"*70)
    sigma = estimator.estimate_sigma_from_theta(data, theta, mu, dt)
    print("Formula: σ² = (2θ / [n(1-e^(-2θΔt))]) Σ [X_{i+1} - X_i - μ(1-e^(-θΔt))]²")
    
    exp_neg_theta_dt = np.exp(-theta * dt)
    exp_neg_2theta_dt = np.exp(-2 * theta * dt)
    print("Intermediate values:")
    print(f"  e^(-θΔt) = e^(-{theta:.6f} × {dt}) = {exp_neg_theta_dt:.6f}")
    print(f"  e^(-2θΔt) = e^(-{2*theta:.6f} × {dt}) = {exp_neg_2theta_dt:.6f}")
    
    n = len(data) - 1
    sum_squared = 0.0
    for i in range(n):
        diff = data[i+1] - data[i] - mu * (1 - exp_neg_theta_dt)
        sum_squared += diff ** 2
    
    denominator = n * (1 - exp_neg_2theta_dt)
    sigma_squared = (2 * theta * sum_squared) / denominator
    
    print(f"  Σ squared differences: {sum_squared:.6f}")
    print(f"  Denominator: {denominator:.6f}")
    print(f"  σ² = (2 × {theta:.6f} × {sum_squared:.6f}) / {denominator:.6f} = {sigma_squared:.6f}")
    print(f"Estimated Sigma (σ, daily): {sigma:.6f}")

    # >>> annualize here <<<
    trading_days = 252
    sigma_annual = sigma * np.sqrt(trading_days)
    print(f"Annualized Sigma (σ_annual, {trading_days}d/yr): {sigma_annual:.6f} "
          f"({sigma_annual*100:.2f}% per year)")
    
    if true_params:
        sigma_error = abs(sigma - true_params['sigma']) / true_params['sigma'] * 100
        print(f"True Sigma: {true_params['sigma']:.6f}")
        print(f"Estimation Error: {sigma_error:.2f}%")
    
    print("\n5. STATIONARY DISTRIBUTION PROPERTIES")
    print("-"*70)
    stationary_var = (sigma ** 2) / (2 * theta)
    stationary_std = np.sqrt(stationary_var)
    print(f"Stationary Variance: σ² / (2θ) = {sigma**2:.6f} / (2 × {theta:.6f}) = {stationary_var:.6f}")
    print(f"Stationary Standard Deviation: {stationary_std:.6f}")
    print(f"Interpretation: Long-term variance around mean is {stationary_std:.4f}")
    
    print("\n6. PARAMETER ESTIMATION SUMMARY")
    print("-"*70)
    print(f"{'Parameter':<12} {'Estimated':<15} {'True Value':<15} {'Error %':<10}")
    print("-"*70)
    
    if true_params:
        params = [
            ('θ (theta)', theta, true_params.get('theta')),
            ('μ (mu)', mu, true_params.get('mu')),
            ('σ (sigma)', sigma, true_params.get('sigma'))
        ]
        
        for name, est, true in params:
            if true is not None:
                error = abs(est - true) / abs(true) * 100
                print(f"{name:<12} {est:<15.6f} {true:<15.6f} {error:<10.2f}%")
            else:
                print(f"{name:<12} {est:<15.6f} {'N/A':<15} {'N/A':<10}")
    else:
        print(f"{'θ (theta)':<12} {theta:<15.6f}")
        print(f"{'μ (mu)':<12} {mu:<15.6f}")
        print(f"{'σ (sigma)':<12} {sigma:<15.6f}")
        print(f"{'σ_annual':<12} {sigma_annual:<15.6f}")
    
    print("="*70)
    
    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,              # daily
        'sigma_annual': sigma_annual,
        'rho': rho,
        'half_life': half_life,
        'stationary_variance': stationary_var,
        'stationary_std': stationary_std
    }

def compare_estimation_methods_detailed(data, dt=1.0, true_params=None):
    estimator = OUEstimator()
    
    print("\n" + "="*70)
    print("ESTIMATION METHOD COMPARISON")
    print("="*70)
    
    methods = {
        'MLE': estimator.estimate_mle,
        'Regression': estimator.estimate_regression,
        'OLS': estimator.estimate_ols
    }
    
    results = {}
    for method_name, method_func in methods.items():
        print(f"\n{method_name} Method:")
        print("-"*70)
        theta, mu, sigma = method_func(data, dt)
        
        results[method_name] = {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': estimator.half_life(),
            'stationary_variance': estimator.stationary_variance()
        }
        
        print(f"  θ = {theta:.6f}")
        print(f"  μ = {mu:.6f}")
        print(f"  σ = {sigma:.6f}")
        if results[method_name]['half_life']:
            print(f"  Half-life = {results[method_name]['half_life']:.4f}")
        
        if true_params:
            theta_err = abs(theta - true_params['theta']) / true_params['theta'] * 100
            mu_err = abs(mu - true_params['mu']) / abs(true_params['mu']) * 100
            sigma_err = abs(sigma - true_params['sigma']) / true_params['sigma'] * 100
            print(f"  Error: θ={theta_err:.2f}%, μ={mu_err:.2f}%, σ={sigma_err:.2f}%")
    
    print("\n" + "="*70)
    print("METHOD COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<12} {'θ':<12} {'μ':<12} {'σ':<12} {'Half-life':<12}")
    print("-"*70)
    
    for method_name, result in results.items():
        print(f"{method_name:<12} {result['theta']:<12.6f} {result['mu']:<12.6f} "
              f"{result['sigma']:<12.6f} {result['half_life'] or 0:<12.4f}")
    
    if true_params:
        print("\nTrue Values:")
        print(f"  θ = {true_params['theta']:.6f}")
        print(f"  μ = {true_params['mu']:.6f}")
        print(f"  σ = {true_params['sigma']:.6f}")
    
    return results


# Note: This module provides analysis functions for real data
# Use detailed_parameter_analysis() and compare_estimation_methods_detailed()
# with your real market data from polygon_data.py


def create_visualization(data, strategy, params, ticker="STOCK"):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    time_steps = np.arange(len(data))
    
    axes[0, 0].plot(time_steps, data, 'b-', linewidth=1, label='Price')
    axes[0, 0].axhline(y=params['mu'], color='r', linestyle='--', label=f'Mean (μ={params["mu"]:.2f})')
    axes[0, 0].set_title(f'{ticker} Price Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    stationary_std = np.sqrt(params['stationary_variance'])
    upper_threshold = params['mu'] + 2 * stationary_std
    lower_threshold = params['mu'] - 2 * stationary_std
    
    signals = []
    for price in data:
        signal, _ = strategy.generate_signal(price)
        signals.append(signal)
    
    axes[0, 1].plot(time_steps, data, 'b-', linewidth=1, label='Price')
    axes[0, 1].axhline(y=params['mu'], color='r', linestyle='--', label='Mean')
    axes[0, 1].axhline(y=upper_threshold, color='g', linestyle=':', label='SELL threshold')
    axes[0, 1].axhline(y=lower_threshold, color='orange', linestyle=':', label='BUY threshold')
    
    buy_indices = [i for i, s in enumerate(signals) if s == 1]
    sell_indices = [i for i, s in enumerate(signals) if s == -1]
    
    if buy_indices:
        axes[0, 1].scatter(buy_indices, data[buy_indices], color='green', marker='^', s=100, label='BUY', zorder=5)
    if sell_indices:
        axes[0, 1].scatter(sell_indices, data[sell_indices], color='red', marker='v', s=100, label='SELL', zorder=5)
    
    axes[0, 1].set_title('Trading Signals')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    deviation = data - params['mu']
    axes[1, 0].plot(time_steps, deviation, 'purple', linewidth=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Deviation from Mean')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Deviation')
    axes[1, 0].grid(True, alpha=0.3)
    
    z_scores = deviation / stationary_std
    axes[1, 1].plot(time_steps, z_scores, 'cyan', linewidth=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].axhline(y=2, color='g', linestyle=':', label='SELL threshold')
    axes[1, 1].axhline(y=-2, color='orange', linestyle=':', label='BUY threshold')
    axes[1, 1].set_title('Z-Score')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Z-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    backtest_results = strategy.backtest(data, dt=1.0)
    portfolio_values = backtest_results['portfolio_values']
    
    axes[2, 0].plot(range(len(portfolio_values)), portfolio_values, 'g-', linewidth=2)
    axes[2, 0].set_title(f'Backtest Performance (Return: {backtest_results["total_return"]*100:.2f}%)')
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Portfolio Value')
    axes[2, 0].grid(True, alpha=0.3)
    
    estimator = OUEstimator()
    lags = range(1, min(50, len(data)//2))
    autocorrs = [estimator.estimate_autocorrelation(data, lag=l) for l in lags]
    
    axes[2, 1].plot(lags, autocorrs, 'ro-', markersize=4)
    axes[2, 1].axhline(y=0, color='k', linestyle='--')
    axes[2, 1].set_title('Autocorrelation Function')
    axes[2, 1].set_xlabel('Lag')
    axes[2, 1].set_ylabel('Autocorrelation')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ou_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to ou_analysis.png")
    plt.close()


if __name__ == "__main__":
    # 6-month USD/CAD from Yahoo Finance
    # Yahoo ticker for USD/CAD is "CAD=X"
    ticker = "CAD=X"
    period = "6mo"    # last 6 months
    interval = "1d"   # daily data

    print("="*70)
    print("PARAMETER ESTIMATION ANALYSIS - REAL DATA (YAHOO FINANCE)")
    print("="*70)
    print(f"\nFetching {ticker} data ({period}, {interval}) from Yahoo Finance...")

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)

        if df.empty:
            raise ValueError("Yahoo Finance returned no data.")

        # Use whatever price column is available
        if "Adj Close" in df.columns:
            price_series = df["Adj Close"]
        elif "Close" in df.columns:
            price_series = df["Close"]
        else:
            raise ValueError(f"No usable price column found. Columns: {df.columns}")

        data = price_series.dropna().values.astype(float)

        if len(data) < 10:
            raise ValueError(f"Not enough data points: got {len(data)}")

        print(f"✓ Fetched {len(data)} data points")
        print(f"  Price range: {data.min():.4f} - {data.max():.4f}")
        print(f"  Mean price: {data.mean():.4f}")

        dt = 1.0  # 1 day between observations

        print("\n" + "="*70)
        print("DETAILED PARAMETER ANALYSIS")
        print("="*70)
        params = detailed_parameter_analysis(data, dt=dt, true_params=None)

        print("\n" + "="*70)
        print("ESTIMATION METHOD COMPARISON")
        print("="*70)
        method_results = compare_estimation_methods_detailed(data, dt=dt, true_params=None)

        print("\n" + "="*70)
        print("TRADING STRATEGY")
        print("="*70)
        strategy = MeanReversionStrategy(threshold_sigma=2.0)
        strategy.fit(data, dt=dt)

        current_price = data[-1]
        signal, signal_type = strategy.generate_signal(current_price)
        deviation = current_price - params['mu']
        z_score = deviation / params['stationary_std']

        print(f"\nCurrent Price: {current_price:.4f}")
        print(f"Mean (μ): {params['mu']:.4f}")
        print(f"Deviation: {deviation:+.4f}")
        print(f"Z-Score: {z_score:.2f}")
        print(f"Signal: {signal_type}")

        print("\n" + "="*70)
        print("BACKTEST")
        print("="*70)
        backtest_results = strategy.backtest(data, dt=dt, allow_short=True, exit_at_mean=True)
        print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
        print(f"Number of Trades: {backtest_results['num_trades']}")

        print("\n" + "="*70)
        print("CREATING VISUALIZATION")
        print("="*70)
        create_visualization(data, strategy, params, ticker=ticker)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
