"""
OU Parameter Estimator
By Josephina Kim (JK)

This handles all the parameter estimation methods - MLE, Regression, and OLS.
The main focus is on getting θ, μ, and σ from the data.
"""

import numpy as np
from scipy import stats


class OUEstimator:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.dt = None
        self.rho = None
    
    def estimate_autocorrelation(self, data, lag=1):
        if len(data) < lag + 1:
            return 0.0
        return np.corrcoef(data[:-lag], data[lag:])[0, 1]
    
    def estimate_theta_from_autocorr(self, data, dt=1.0):
        rho = self.estimate_autocorrelation(data, lag=1)
        self.rho = rho
        
        if rho <= 0 or rho >= 1:
            return 0.01
        
        theta = -np.log(rho) / dt
        return max(theta, 0.001)
    
    def estimate_mu(self, data):
        return np.mean(data)
    
    def estimate_sigma_from_theta(self, data, theta, mu, dt=1.0):
        n = len(data) - 1
        if n < 1 or theta <= 0:
            return np.std(data)
        
        exp_neg_theta_dt = np.exp(-theta * dt)
        exp_neg_2theta_dt = np.exp(-2 * theta * dt)
        
        sum_squared = 0.0
        for i in range(n):
            diff = data[i+1] - data[i] - mu * (1 - exp_neg_theta_dt)
            sum_squared += diff ** 2
        
        denominator = n * (1 - exp_neg_2theta_dt)
        if denominator <= 0:
            return np.std(data)
        
        sigma_squared = (2 * theta * sum_squared) / denominator
        return np.sqrt(max(sigma_squared, 0.001))
    
    def get_estimation_details(self):
        return {
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma,
            'dt': self.dt,
            'rho': self.rho,
            'half_life': self.half_life(),
            'stationary_variance': self.stationary_variance()
        }
    
    def estimate_mle(self, data, dt=1.0):
        if len(data) < 2:
            return None, None, None
        
        mu = self.estimate_mu(data)
        theta = self.estimate_theta_from_autocorr(data, dt)
        sigma = self.estimate_sigma_from_theta(data, theta, mu, dt)
        
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        
        return theta, mu, sigma
    
    def estimate_regression(self, data, dt=1.0):
        if len(data) < 2:
            return None, None, None
        
        X = data[:-1]
        Y = data[1:]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
        
        mu = intercept / (1 - slope)
        theta = -np.log(max(slope, 0.001)) / dt
        residuals = Y - (slope * X + intercept)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        
        return theta, mu, sigma
    
    def estimate_ols(self, data, dt=1.0):
        if len(data) < 2:
            return None, None, None
        
        n = len(data) - 1
        X = data[:-1]
        Y = np.diff(data)
        
        sum_X = np.sum(X)
        sum_Y = np.sum(Y)
        sum_X2 = np.sum(X ** 2)
        sum_XY = np.sum(X * Y)
        
        denominator = n * sum_X2 - sum_X ** 2
        if abs(denominator) < 1e-10:
            return self.estimate_mle(data, dt)
        
        beta = (n * sum_XY - sum_X * sum_Y) / denominator
        alpha = (sum_Y - beta * sum_X) / n
        
        theta = -beta / dt
        mu = alpha / (theta * dt) if theta > 0 else np.mean(data)
        residuals = Y - (alpha + beta * X)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        theta = max(theta, 0.001)
        sigma = max(sigma, 0.001)
        
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        
        return theta, mu, sigma
    
    def get_parameters(self):
        return {
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma,
            'dt': self.dt
        }
    
    def half_life(self):
        if self.theta is None or self.theta <= 0:
            return None
        return np.log(2) / self.theta
    
    def stationary_variance(self):
        if self.theta is None or self.sigma is None or self.theta <= 0:
            return None
        return (self.sigma ** 2) / (2 * self.theta)

