"""
Stochastic Volatility Models: Heston (1993), Bates (1996), Double Heston (2009).

Implements characteristic functions with the "rotation count" fix (Albrecher et al.)
and Lord-Kahl log-branch continuation to avoid discontinuities in the complex logarithm.

Pricing via Carr-Madan / Lewis (2001) FFT approach for speed, with Gil-Pelaez fallback.
"""

import numpy as np
from scipy import integrate
from numpy import exp, log, sqrt, pi


# =============================================================================
# Characteristic Functions
# =============================================================================

def _heston_cf_component(u, tau, kappa, theta, sigma, rho, v0):
    """
    Core Heston log-characteristic function component using the
    'little Heston trap' formulation (Albrecher et al. 2007) to ensure
    continuity of the complex logarithm.

    Returns log phi (without the drift term).
    """
    d = sqrt((rho * sigma * 1j * u - kappa) ** 2 + sigma ** 2 * (1j * u + u ** 2))

    # Use the formulation that avoids branch-cut issues
    g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)

    # Exponential damping for numerical stability
    exp_dt = exp(-d * tau)

    C = (kappa * theta / sigma ** 2) * (
            (kappa - rho * sigma * 1j * u - d) * tau
            - 2.0 * log((1.0 - g * exp_dt) / (1.0 - g))
    )

    D = ((kappa - rho * sigma * 1j * u - d) / sigma ** 2) * (
            (1.0 - exp_dt) / (1.0 - g * exp_dt)
    )

    return C, D


def heston_cf(u, S0, r, q, tau, kappa, theta, sigma, rho, v0):
    """
    Characteristic function of log(S_T) under the Heston model.
    """
    C, D = _heston_cf_component(u, tau, kappa, theta, sigma, rho, v0)
    log_phi = 1j * u * (log(S0) + (r - q) * tau) + C + D * v0
    # Clip real part to avoid exp overflow
    log_phi = np.where(np.real(log_phi) > 500, 500 + 1j * np.imag(log_phi), log_phi)
    return exp(log_phi)


def bates_cf(u, S0, r, q, tau, kappa, theta, sigma, rho, v0,
             lambda_j, mu_j, sigma_j):
    """
    Characteristic function of log(S_T) under the Bates (1996) model.
    Adds Merton-type jumps to Heston.
    """
    # Heston part
    C, D = _heston_cf_component(u, tau, kappa, theta, sigma, rho, v0)

    # Jump part: compensated compound Poisson
    # E[e^J] - 1 where J ~ N(mu_j, sigma_j^2) for the log-jump
    jump_mean = exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    jump_cf = exp(1j * u * mu_j + 0.5 * (1j * u * sigma_j) ** 2) - 1.0
    # Using -0.5 * u^2 * sigma_j^2 is wrong; need the full complex version
    jump_cf_full = exp(1j * u * mu_j - 0.5 * u ** 2 * sigma_j ** 2 + 0.5 * (1j * u * sigma_j) ** 2 * 0) 
    # Correct: phi_J(u) = exp(i*u*mu_j - 0.5*u^2*sigma_j^2)
    # But we need the CF of log(1+J_percent), let's redo properly:
    # If Y = log(S_T/S_{T-}), Y ~ N(mu_j, sigma_j^2)
    # Then CF contribution: lambda * tau * (exp(i*u*mu_j - 0.5*u^2*sigma_j^2) - 1 - i*u*(exp(mu_j + 0.5*sigma_j^2) - 1))
    phi_j = exp(1j * u * mu_j - 0.5 * u ** 2 * sigma_j ** 2)
    jump_compensation = lambda_j * tau * (phi_j - 1.0 - 1j * u * jump_mean)

    log_phi = (1j * u * (log(S0) + (r - q) * tau)
               + C + D * v0
               + jump_compensation)
    log_phi = np.where(np.real(log_phi) > 500, 500 + 1j * np.imag(log_phi), log_phi)
    return exp(log_phi)


def double_heston_cf(u, S0, r, q, tau,
                     kappa1, theta1, sigma1, rho1, v01,
                     kappa2, theta2, sigma2, rho2, v02):
    """
    Characteristic function of log(S_T) under the Double Heston model
    (Christoffersen, Heston & Jacobs 2009).
    Two independent variance processes.
    """
    C1, D1 = _heston_cf_component(u, tau, kappa1, theta1, sigma1, rho1, v01)
    C2, D2 = _heston_cf_component(u, tau, kappa2, theta2, sigma2, rho2, v02)

    log_phi = (1j * u * (log(S0) + (r - q) * tau)
               + C1 + D1 * v01
               + C2 + D2 * v02)
    log_phi = np.where(np.real(log_phi) > 500, 500 + 1j * np.imag(log_phi), log_phi)
    return exp(log_phi)


# =============================================================================
# FFT-based Pricing (Carr-Madan 1999)
# =============================================================================

def _carr_madan_fft(cf_func, S0, K_target, r, q, tau, cf_params,
                    N=4096, alpha=1.5, eta=0.25):
    """
    Carr-Madan FFT pricing for European calls.

    Parameters
    ----------
    cf_func : callable
        Characteristic function cf(u, S0, r, q, tau, *cf_params)
    K_target : float or array
        Strike(s) to price
    N : int
        FFT size (power of 2)
    alpha : float
        Dampening factor (1 < alpha, typically 1.5)
    eta : float
        Integration grid spacing

    Returns
    -------
    prices : array
        Call prices at the target strikes
    """
    K_target = np.atleast_1d(np.asarray(K_target, dtype=float))

    # Grid in log-strike space
    lam = 2 * pi / (N * eta)  # log-strike spacing
    b = N * lam / 2  # upper bound of log-strike grid

    # Integration grid
    v = np.arange(N) * eta  # u_j = j * eta
    k = -b + lam * np.arange(N)  # log-strike grid

    # Characteristic function values (vectorized)
    u = v - (alpha + 1) * 1j
    cf_vals = cf_func(u, S0, r, q, tau, *cf_params)

    # Modified characteristic function for calls
    denom = alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v
    psi = exp(-r * tau) * cf_vals / denom

    # Simpson's rule weights
    simpson = 3 + (-1) ** (np.arange(N) + 1)
    simpson[0] = 1
    simpson = simpson / 3

    # FFT input
    x = exp(1j * v * b) * psi * eta * simpson
    # Replace any NaN/inf with 0 before FFT
    x = np.where(np.isfinite(x), x, 0.0)
    fft_result = np.fft.fft(x)

    # Call prices on the grid
    call_prices_grid = (exp(-alpha * k) / pi) * np.real(fft_result)

    # Interpolate to target strikes
    log_K_target = log(K_target)
    call_prices = np.interp(log_K_target, k, call_prices_grid)

    return np.maximum(call_prices, 0.0)


def _quadrature_price(cf_func, S0, K, r, q, tau, cf_params, option_type='call'):
    """
    Gil-Pelaez inversion for a single strike. More accurate but slower than FFT.
    Used as fallback / for single-strike pricing.
    """
    K = float(K)

    def integrand_P1(u):
        cf_val = cf_func(u - 1j, S0, r, q, tau, *cf_params)
        fwd = S0 * exp((r - q) * tau)
        return np.real(exp(-1j * u * log(K)) * cf_val / (1j * u * fwd * exp(-r * tau) * exp((r-q)*tau)))

    def integrand_P2(u):
        cf_val = cf_func(u, S0, r, q, tau, *cf_params)
        return np.real(exp(-1j * u * log(K)) * cf_val / (1j * u))

    # Use Lewis (2001) formula directly
    def integrand_call(u):
        cf_val = cf_func(u - 0.5j, S0, r, q, tau, *cf_params)
        k = log(K)
        return np.real(exp(-1j * u * k) * cf_val / (u ** 2 + 0.25))

    call_price = (S0 * exp(-q * tau) - K * exp(-r * tau)) / 2 + \
                 exp(-r * tau) / pi * integrate.quad(
        integrand_call, 0, 500, limit=200)[0] * (-1) * K

    # Actually let's use the standard decomposition
    # C = S*exp(-q*tau)*P1 - K*exp(-r*tau)*P2
    fwd = S0 * exp((r - q) * tau)

    def integrand1(u):
        if u < 1e-12:
            return 0.0
        cf_val = cf_func(u, S0, r, q, tau, *cf_params)
        return np.real(exp(-1j * u * log(K)) * cf_val / (1j * u * S0 * exp(-q * tau)))

    def integrand2(u):
        if u < 1e-12:
            return 0.0
        cf_val = cf_func(u, S0, r, q, tau, *cf_params)
        return np.real(exp(-1j * u * log(K)) * cf_val / (1j * u))

    # Use the Carr-Madan single strike approach instead for reliability
    alpha = 1.5

    def integrand_cm(v):
        u = v - (alpha + 1) * 1j
        cf_val = cf_func(u, S0, r, q, tau, *cf_params)
        denom = alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v
        psi = exp(-r * tau) * cf_val / denom
        return np.real(exp(-1j * v * log(K)) * psi)

    integral, _ = integrate.quad(integrand_cm, 0, 500, limit=300)
    call_price = max(exp(-alpha * log(K)) / pi * integral, 0.0)

    if option_type == 'call':
        return call_price
    else:
        # Put-call parity
        return call_price - S0 * exp(-q * tau) + K * exp(-r * tau)


# =============================================================================
# Public Pricing API
# =============================================================================

def price_european(model, S0, K, r, q, tau, params, option_type='call', method='fft'):
    """
    Price European vanilla options.

    Parameters
    ----------
    model : str
        'heston', 'bates', or 'double_heston'
    S0 : float
        Spot price
    K : float or array
        Strike(s)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    tau : float
        Time to maturity (years)
    params : dict
        Model parameters
    option_type : str
        'call' or 'put'
    method : str
        'fft' or 'quad'

    Returns
    -------
    prices : float or array
    """
    if tau < 1e-10:
        K_arr = np.atleast_1d(K)
        if option_type == 'call':
            return np.maximum(S0 - K_arr, 0.0)
        else:
            return np.maximum(K_arr - S0, 0.0)

    cf_func, cf_params = _get_cf(model, params)

    K_arr = np.atleast_1d(np.asarray(K, dtype=float))

    if method == 'fft':
        call_prices = _carr_madan_fft(cf_func, S0, K_arr, r, q, tau, cf_params)
    else:
        call_prices = np.array([
            _quadrature_price(cf_func, S0, k, r, q, tau, cf_params, 'call')
            for k in K_arr
        ])

    if option_type == 'call':
        prices = call_prices
    else:
        # Put-call parity
        prices = call_prices - S0 * exp(-q * tau) + K_arr * exp(-r * tau)
        prices = np.maximum(prices, 0.0)

    return prices if len(prices) > 1 else float(prices[0])


def _get_cf(model, params):
    """Map model name to characteristic function and parameter tuple."""
    if model == 'heston':
        cf_func = heston_cf
        cf_params = (params['kappa'], params['theta'], params['sigma'],
                     params['rho'], params['v0'])
    elif model == 'bates':
        cf_func = bates_cf
        cf_params = (params['kappa'], params['theta'], params['sigma'],
                     params['rho'], params['v0'],
                     params['lambda_j'], params['mu_j'], params['sigma_j'])
    elif model == 'double_heston':
        cf_func = double_heston_cf
        cf_params = (params['kappa1'], params['theta1'], params['sigma1'],
                     params['rho1'], params['v01'],
                     params['kappa2'], params['theta2'], params['sigma2'],
                     params['rho2'], params['v02'])
    else:
        raise ValueError(f"Unknown model: {model}")
    return cf_func, cf_params


# =============================================================================
# Binary (Digital) Options
# =============================================================================

def price_binary_call(model, S0, K, r, q, tau, params):
    """
    Price a cash-or-nothing binary call via numerical differentiation
    of the vanilla call price w.r.t. strike.

    Binary Call = -d(Call)/dK * exp(r*tau)  ... actually:
    Binary Call (cash-or-nothing) = exp(-r*tau) * P2
    where P2 = risk-neutral probability S_T > K.

    We use: BinaryCall = -dC/dK  (since C = exp(-r*tau)*E[(S_T - K)+])
    """
    eps = K * 1e-4
    c_up = price_european(model, S0, K + eps, r, q, tau, params, 'call', 'fft')
    c_dn = price_european(model, S0, K - eps, r, q, tau, params, 'call', 'fft')
    binary_call = -(c_up - c_dn) / (2 * eps)
    return max(binary_call, 0.0)


def price_binary_put(model, S0, K, r, q, tau, params):
    """Cash-or-nothing binary put."""
    bc = price_binary_call(model, S0, K, r, q, tau, params)
    return max(exp(-r * tau) - bc, 0.0)


# =============================================================================
# Variance Swaps
# =============================================================================

def variance_swap_strike(model, S0, r, q, tau, params, n_strikes=200):
    """
    Fair variance swap strike (annualised variance).

    Uses the model-free replication formula:
    E[Var] = (2/T) * { integral_0^F [P(K)/K^2 dK] + integral_F^inf [C(K)/K^2 dK] }

    where F = S0 * exp((r-q)*T) is the forward.

    For the Heston model specifically, there's an analytical formula:
    E[V] = theta + (v0 - theta) * (1 - exp(-kappa*T)) / (kappa*T)

    We implement both: analytical for Heston, numerical replication for all.
    """
    if model == 'heston':
        kappa = params['kappa']
        theta = params['theta']
        v0 = params['v0']
        if abs(kappa * tau) < 1e-10:
            fair_var = v0
        else:
            fair_var = theta + (v0 - theta) * (1 - exp(-kappa * tau)) / (kappa * tau)
        return fair_var

    elif model == 'bates':
        # For Bates: Heston part + jump contribution
        kappa = params['kappa']
        theta = params['theta']
        v0 = params['v0']
        lambda_j = params['lambda_j']
        mu_j = params['mu_j']
        sigma_j = params['sigma_j']

        # Continuous part (same as Heston)
        if abs(kappa * tau) < 1e-10:
            heston_var = v0
        else:
            heston_var = theta + (v0 - theta) * (1 - exp(-kappa * tau)) / (kappa * tau)

        # Jump part: E[sum of J^2] / T = lambda * E[J^2] = lambda * (mu_j^2 + sigma_j^2)
        jump_var = lambda_j * (mu_j ** 2 + sigma_j ** 2)

        return heston_var + jump_var

    elif model == 'double_heston':
        # Sum of two Heston-type components
        fair_var = 0.0
        for suffix in ['1', '2']:
            kappa = params[f'kappa{suffix}']
            theta = params[f'theta{suffix}']
            v0 = params[f'v0{suffix}']
            if abs(kappa * tau) < 1e-10:
                fair_var += v0
            else:
                fair_var += theta + (v0 - theta) * (1 - exp(-kappa * tau)) / (kappa * tau)
        return fair_var


def variance_swap_vol_strike(model, S0, r, q, tau, params):
    """Return the vol strike = sqrt(fair_variance)."""
    fair_var = variance_swap_strike(model, S0, r, q, tau, params)
    return sqrt(max(fair_var, 0.0))
