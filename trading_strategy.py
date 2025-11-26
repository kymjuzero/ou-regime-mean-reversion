"""
Mean Reversion Trading Strategy
By Josephina Kim (JK)

This is where the trading logic happens - above mean = SELL, below mean = BUY.
Uses stationary variance for signal generation (as recommended by QuestDB).
"""

import numpy as np
from ou_estimator import OUEstimator


class MeanReversionStrategy:
    def __init__(self, estimator_method="mle", threshold_sigma=2.0):
        self.estimator = OUEstimator()
        self.estimator_method = estimator_method
        self.threshold_sigma = threshold_sigma
        self.theta = None
        self.mu = None
        self.sigma = None
        self.dt = 1.0

    def fit(self, data, dt=1.0):
        self.dt = dt

        if self.estimator_method == "mle":
            theta, mu, sigma = self.estimator.estimate_mle(data, dt)
        elif self.estimator_method == "regression":
            theta, mu, sigma = self.estimator.estimate_regression(data, dt)
        elif self.estimator_method == "ols":
            theta, mu, sigma = self.estimator.estimate_ols(data, dt)
        else:
            theta, mu, sigma = self.estimator.estimate_mle(data, dt)

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

        return self

    def generate_signal(self, current_value):
        if self.mu is None or self.sigma is None or self.theta is None:
            return 0, "NO_SIGNAL"

        stationary_std = np.sqrt((self.sigma**2) / (2 * self.theta))
        deviation = current_value - self.mu
        z_score = deviation / stationary_std

        if z_score > self.threshold_sigma:
            return -1, "SELL"
        elif z_score < -self.threshold_sigma:
            return 1, "BUY"
        else:
            return 0, "HOLD"

    def generate_signal_continuous(self, current_value):
        if self.mu is None or self.sigma is None or self.theta is None:
            return 0.0

        stationary_std = np.sqrt((self.sigma**2) / (2 * self.theta))
        deviation = current_value - self.mu
        z_score = deviation / stationary_std

        if abs(z_score) < 0.5:
            return 0.0

        signal_strength = -np.clip(z_score, -3, 3) / 3.0
        return signal_strength

    def get_position_size(self, current_value, base_size=1.0):
        if self.mu is None or self.sigma is None:
            return 0.0

        deviation = current_value - self.mu
        z_score = abs(deviation) / self.sigma

        position_size = base_size * z_score / self.threshold_sigma
        return np.clip(position_size, 0, base_size * 2)

    def get_stop_loss(self, k=3.0):
        if self.mu is None or self.sigma is None or self.theta is None:
            return None, None

        stationary_var = (self.sigma**2) / (2 * self.theta)
        stop_loss_upper = self.mu + k * np.sqrt(stationary_var)
        stop_loss_lower = self.mu - k * np.sqrt(stationary_var)

        return stop_loss_lower, stop_loss_upper

    def get_parameters(self):
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "dt": self.dt,
            "half_life": self.estimator.half_life() if self.theta else None,
            "stationary_variance": self.estimator.stationary_variance()
            if self.theta
            else None,
        }

    def backtest(
        self,
        data,
        dt=1.0,
        initial_capital=10000,
        stop_loss_pct=0.20,
        exit_at_mean=True,
        allow_short=True,
    ):
        self.fit(data, dt)

        positions = []
        signals = []
        portfolio_values = [initial_capital]
        cash = initial_capital
        position = 0
        entry_price = 0
        position_type = None

        trades = 0

        for i in range(len(data)):
            current_price = data[i]
            signal, signal_type = self.generate_signal(current_price)
            signals.append(signal_type)

            deviation = current_price - self.mu
            prev_price = data[i - 1] if i > 0 else current_price
            prev_deviation = prev_price - self.mu

            # Enter long position on BUY signal
            if signal == 1 and position == 0:
                shares = int(cash / current_price)
                if shares > 0:
                    position = shares
                    position_type = "long"
                    entry_price = current_price
                    cash -= shares * current_price
                    trades += 1

            # Enter short position on SELL signal
            elif signal == -1 and position == 0 and allow_short:
                shares = int(cash / current_price)
                if shares > 0:
                    position = shares
                    position_type = "short"
                    entry_price = current_price
                    cash += shares * current_price
                    trades += 1

            # Exit long position
            if position > 0 and position_type == "long":
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct > stop_loss_pct:
                    cash += position * current_price
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif exit_at_mean and prev_deviation < 0 and deviation >= 0:
                    cash += position * current_price
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif signal == -1:
                    cash += position * current_price
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1

            # Exit short position
            elif position > 0 and position_type == "short":
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct > stop_loss_pct:
                    cash += position * (2 * entry_price - current_price)
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif exit_at_mean and prev_deviation > 0 and deviation <= 0:
                    cash += position * (2 * entry_price - current_price)
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif signal == 1:
                    cash += position * (2 * entry_price - current_price)
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1

            # Portfolio value
            if position > 0:
                if position_type == "long":
                    portfolio_value = cash + position * current_price
                else:
                    portfolio_value = cash + position * (2 * entry_price - current_price)
            else:
                portfolio_value = cash

            portfolio_values.append(portfolio_value)
            positions.append(position)

        # Exit any remaining position
        if position > 0:
            if position_type == "long":
                cash += position * data[-1]
            else:
                cash += position * (2 * entry_price - data[-1])
            portfolio_values[-1] = cash

        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital

        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )

        return {
            "signals": signals,
            "positions": positions,
            "portfolio_values": portfolio_values,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": self._calculate_max_drawdown(portfolio_values),
            "num_trades": trades,
        }

    def backtest_with_regime(
        self,
        data,
        regimes,
        dt=1.0,
        initial_capital=10000,
        stop_loss_pct=0.20,
        exit_at_mean=True,
        allow_short=True,
        use_mr_only_for_fit=False,
        min_mr_points_for_fit=30,
    ):
        """
        Same as backtest(), but only trades when regime == 'MR'.
        Outside MR, we just hold existing position and mark-to-market.

        If use_mr_only_for_fit is True, OU parameters are estimated
        using only the MR points (if there are enough).
        """
        assert len(data) == len(regimes), "data and regimes must be same length"

        regimes_arr = np.array(regimes, dtype=object)

        if use_mr_only_for_fit:
            mr_mask = np.array([r == "MR" for r in regimes_arr], dtype=bool)
            mr_data = np.asarray(data)[mr_mask]

            if len(mr_data) >= min_mr_points_for_fit:
                self.fit(mr_data, dt)
            else:
                # Fallback: not enough MR-only data, fit on full series
                self.fit(data, dt)
        else:
            self.fit(data, dt)

        positions = []
        signals = []
        portfolio_values = [initial_capital]
        cash = initial_capital
        position = 0
        entry_price = 0
        position_type = None
        trades = 0

        for i in range(len(data)):
            current_price = data[i]
            prev_price = data[i - 1] if i > 0 else current_price

            # If not in mean-reverting regime, skip trading logic
            # (but still mark-to-market any existing position)
            if regimes_arr[i] is not None and regimes_arr[i] != "MR":
                signals.append("NO_TRADE_REGIME")
                if position > 0:
                    if position_type == "long":
                        portfolio_value = cash + position * current_price
                    else:  # short
                        portfolio_value = cash + position * (
                            2 * entry_price - current_price
                        )
                else:
                    portfolio_value = cash

                portfolio_values.append(portfolio_value)
                positions.append(position)
                continue

            # --- Normal OU trading logic ---
            signal, signal_type = self.generate_signal(current_price)
            signals.append(signal_type)

            deviation = current_price - self.mu
            prev_deviation = prev_price - self.mu

            # Enter long
            if signal == 1 and position == 0:
                shares = int(cash / current_price)
                if shares > 0:
                    position = shares
                    position_type = "long"
                    entry_price = current_price
                    cash -= shares * current_price
                    trades += 1

            # Enter short
            elif signal == -1 and position == 0 and allow_short:
                shares = int(cash / current_price)
                if shares > 0:
                    position = shares
                    position_type = "short"
                    entry_price = current_price
                    cash += shares * current_price
                    trades += 1

            # Exit long
            if position > 0 and position_type == "long":
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct > stop_loss_pct:
                    cash += position * current_price
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif exit_at_mean and prev_deviation < 0 and deviation >= 0:
                    cash += position * current_price
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif signal == -1:
                    cash += position * current_price
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1

            # Exit short
            elif position > 0 and position_type == "short":
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct > stop_loss_pct:
                    cash += position * (2 * entry_price - current_price)
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif exit_at_mean and prev_deviation > 0 and deviation <= 0:
                    cash += position * (2 * entry_price - current_price)
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1
                elif signal == 1:
                    cash += position * (2 * entry_price - current_price)
                    position = 0
                    position_type = None
                    entry_price = 0
                    trades += 1

            # Mark-to-market
            if position > 0:
                if position_type == "long":
                    portfolio_value = cash + position * current_price
                else:
                    portfolio_value = cash + position * (
                        2 * entry_price - current_price
                    )
            else:
                portfolio_value = cash

            portfolio_values.append(portfolio_value)
            positions.append(position)

        # Close any remaining position at the last price
        if position > 0:
            if position_type == "long":
                cash += position * data[-1]
            else:
                cash += position * (2 * entry_price - data[-1])
            portfolio_values[-1] = cash

        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )

        return {
            "signals": signals,
            "positions": positions,
            "portfolio_values": portfolio_values,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": self._calculate_max_drawdown(portfolio_values),
            "num_trades": trades,
        }

    def _calculate_max_drawdown(self, portfolio_values):
        peak = portfolio_values[0]
        max_dd = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd
