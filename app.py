"""
Options Market Making Tool — Heston Family Models
===================================================
Streamlit application for calibrating Heston, Bates, and Double Heston models
to market data, and pricing vanilla, binary options, and variance swaps.

Author: Quantitative Finance Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    price_european, price_binary_call, price_binary_put,
    variance_swap_strike, variance_swap_vol_strike
)
from calibration import (
    calibrate, implied_vol, bs_price, bs_vega,
    PARAM_BOUNDS, PARAM_NAMES, DEFAULT_PARAMS
)

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Heston Options Pricer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Headers */
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; }
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.4rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    /* Info boxes */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar — Market Data Input
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Market Parameters")

    S0 = st.number_input("Spot Price (S₀)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    r = st.number_input("Risk-Free Rate (r)", value=0.02, min_value=-0.05, max_value=0.30, step=0.005, format="%.4f")
    q = st.number_input("Dividend Yield (q)", value=0.01, min_value=0.0, max_value=0.20, step=0.005, format="%.4f")

    st.markdown("---")
    st.markdown("## 🔬 Model Selection")

    model_choice = st.selectbox(
        "Stochastic Volatility Model",
        options=['heston', 'bates', 'double_heston'],
        format_func=lambda x: {
            'heston': 'Heston (1993)',
            'bates': 'Bates (1996) — with jumps',
            'double_heston': 'Double Heston (2009)'
        }[x]
    )

    st.markdown("---")
    st.markdown("## 📈 Calibration Settings")

    loss_type = st.selectbox(
        "Loss Function",
        options=['ivrmse', 'price_abs', 'iv_relative', 'price_rmse'],
        format_func=lambda x: {
            'ivrmse': 'IV RMSE (recommended)',
            'price_abs': 'Vega-weighted Price (fast)',
            'iv_relative': 'IV RMSE relative (slower)',
            'price_rmse': 'Price RMSE relative'
        }[x]
    )

    maxiter = st.slider("Max Iterations", 50, 500, 100, 50)
    popsize = st.slider("Num Restarts", 2, 10, 3, 1)


# =============================================================================
# Main Area
# =============================================================================

st.markdown("# 📊 Options Market Making — Heston Family")
st.markdown("""
<div class="info-box">
Calibrate <b>Heston</b>, <b>Bates</b>, or <b>Double Heston</b> models to your volatility surface,
then price vanilla options, binary options, and variance swaps.
Pricing uses <b>Carr-Madan FFT</b> for speed and the <b>Albrecher et al.</b> formulation for numerical stability.
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Tabs
# =============================================================================

tab_calib, tab_pricer, tab_exotics, tab_surface, tab_params = st.tabs([
    "🎯 Calibration", "💰 Vanilla Pricer", "🔮 Exotics", "🌊 Vol Surface", "📋 Parameters"
])

# =============================================================================
# TAB 1: CALIBRATION
# =============================================================================

with tab_calib:
    st.markdown("### Market Data Input")

    col_upload, col_manual = st.columns([1, 1])

    with col_upload:
        st.markdown("**Upload volatility surface** (.xlsx or .csv)")
        uploaded_file = st.file_uploader(
            "Upload market data",
            type=['xlsx', 'csv', 'xls'],
            help="Expected columns: Strike (or K), Maturity (or tau/T), IV (or implied_vol), Type (call/put, optional)"
        )

    with col_manual:
        st.markdown("**Or use sample data**")
        use_sample = st.checkbox("Load sample vol surface", value=True if uploaded_file is None else False)

    # Parse market data
    market_data = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)

            st.markdown("**Raw uploaded data:**")
            st.dataframe(df_raw.head(10), use_container_width=True)

            # Try to map columns
            col_map = {}
            for col in df_raw.columns:
                cl = col.lower().strip()
                if cl in ['k', 'strike', 'strike_price', 'strikes']:
                    col_map['K'] = col
                elif cl in ['tau', 't', 'maturity', 'expiry', 'tte', 'time_to_maturity']:
                    col_map['tau'] = col
                elif cl in ['iv', 'implied_vol', 'impliedvol', 'implied_volatility',
                            'vol', 'sigma', 'implvol', 'imp_vol', 'ivol']:
                    col_map['iv'] = col
                elif cl in ['type', 'option_type', 'cp', 'call_put']:
                    col_map['type'] = col

            if 'K' in col_map and 'tau' in col_map and 'iv' in col_map:
                # First, detect if IVs are in percentage or decimal
                all_iv_raw = pd.to_numeric(df_raw[col_map['iv']], errors='coerce').dropna()
                is_percentage = all_iv_raw.median() > 1.0  # median > 1 means percentage
                if is_percentage:
                    st.info("📊 Detected IV in percentage format — converting to decimal.")

                market_data = []
                for _, row in df_raw.iterrows():
                    iv_val = float(row[col_map['iv']])
                    if is_percentage:
                        iv_val = iv_val / 100.0
                    opt_type = 'call'
                    if 'type' in col_map:
                        t = str(row[col_map['type']]).lower().strip()
                        if t in ['put', 'p']:
                            opt_type = 'put'
                    market_data.append({
                        'K': float(row[col_map['K']]),
                        'tau': float(row[col_map['tau']]),
                        'market_iv': iv_val,
                        'option_type': opt_type,
                        'weight': 1.0
                    })
                st.success(f"✅ Parsed {len(market_data)} market data points.")
            else:
                st.error(f"Could not identify required columns. Found mappings: {col_map}. "
                         f"Need at least: Strike (K), Maturity (tau), Implied Vol (iv).")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if use_sample and market_data is None:
        # Generate vol surface from a true Heston model
        # This ensures calibration can recover meaningful parameters
        sample_params = {'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'sigma': 0.5, 'rho': -0.7}
        maturities = [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        moneyness_range = np.linspace(0.80, 1.20, 15)

        market_data = []
        for tau in maturities:
            strikes = S0 * moneyness_range
            prices = price_european('heston', S0, strikes, r, q, tau,
                                    sample_params, option_type='call', method='fft')
            for K_val, p_val in zip(strikes, np.atleast_1d(prices)):
                iv_val = implied_vol(p_val, S0, K_val, r, q, tau)
                if not np.isnan(iv_val) and iv_val > 0.01:
                    # Add small noise to simulate real market data
                    iv_noisy = iv_val + np.random.normal(0, 0.001)
                    iv_noisy = max(iv_noisy, 0.01)
                    market_data.append({
                        'K': round(K_val, 2),
                        'tau': tau,
                        'market_iv': round(iv_noisy, 6),
                        'option_type': 'call',
                        'weight': 1.0
                    })

        st.info(f"📊 Using Heston-generated vol surface: {len(market_data)} data points "
                f"({len(maturities)} maturities × {len(moneyness_range)} strikes)")
        st.caption("Sample surface generated from Heston(v₀=0.04, κ=2, θ=0.04, ξ=0.5, ρ=-0.7)")

    if market_data is not None:
        # Display market data summary
        df_mkt = pd.DataFrame(market_data)
        with st.expander("📋 Market Data Preview", expanded=False):
            st.dataframe(df_mkt.head(20), use_container_width=True)

        # Calibration button
        st.markdown("---")
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            run_calib = st.button("🚀 Run Calibration", type="primary", use_container_width=True)

        if run_calib:
            with st.spinner(f"Calibrating {model_choice.replace('_', ' ').title()} model..."):
                result = calibrate(
                    model=model_choice,
                    S0=S0, r=r, q=q,
                    market_data=market_data,
                    loss_type=loss_type,
                    maxiter=maxiter,
                    popsize=popsize,
                )
                st.session_state['calib_result'] = result
                st.session_state['calib_model'] = model_choice
                st.session_state['market_data'] = market_data

        # Display results
        if 'calib_result' in st.session_state:
            result = st.session_state['calib_result']

            st.markdown("### Calibration Results")

            # Metrics
            params = result['params']
            n_params = len(params)

            metric_cols = st.columns(min(n_params + 1, 6))
            metric_cols[0].metric("RMSE", f"{result['rmse']:.6f}")

            param_items = list(params.items())
            col_idx = 1
            for name, val in param_items:
                if col_idx >= len(metric_cols):
                    break
                label_map = {
                    'v0': 'v₀', 'kappa': 'κ', 'theta': 'θ', 'sigma': 'ξ', 'rho': 'ρ',
                    'lambda_j': 'λⱼ', 'mu_j': 'μⱼ', 'sigma_j': 'σⱼ',
                    'v01': 'v₀¹', 'kappa1': 'κ¹', 'theta1': 'θ¹', 'sigma1': 'ξ¹', 'rho1': 'ρ¹',
                    'v02': 'v₀²', 'kappa2': 'κ²', 'theta2': 'θ²', 'sigma2': 'ξ²', 'rho2': 'ρ²',
                }
                metric_cols[col_idx].metric(label_map.get(name, name), f"{val:.4f}")
                col_idx += 1

            if n_params > 5:
                metric_cols2 = st.columns(min(n_params - 4, 6))
                for i, (name, val) in enumerate(param_items[5:]):
                    if i >= len(metric_cols2):
                        break
                    label_map = {
                        'lambda_j': 'λⱼ', 'mu_j': 'μⱼ', 'sigma_j': 'σⱼ',
                        'v02': 'v₀²', 'kappa2': 'κ²', 'theta2': 'θ²', 'sigma2': 'ξ²', 'rho2': 'ρ²',
                    }
                    metric_cols2[i].metric(label_map.get(name, name), f"{val:.4f}")

            # Feller condition check
            if model_choice in ['heston', 'bates']:
                feller = 2 * params['kappa'] * params['theta'] - params['sigma'] ** 2
                if feller > 0:
                    st.success(f"✅ Feller condition satisfied: 2κθ - ξ² = {feller:.4f} > 0")
                else:
                    st.warning(f"⚠️ Feller condition violated: 2κθ - ξ² = {feller:.4f} < 0 (variance can reach zero)")

            # Fit chart
            st.markdown("### Model Fit")

            df_fit = pd.DataFrame({
                'Strike': [md['K'] for md in market_data],
                'Maturity': [md['tau'] for md in market_data],
                'Market IV': result['market_ivs'],
                'Model IV': result['model_ivs'],
            })
            df_fit['Error (bps)'] = (df_fit['Model IV'] - df_fit['Market IV']) * 10000
            df_fit = df_fit.dropna()

            # Choose a few maturities to plot
            unique_taus = sorted(df_fit['Maturity'].unique())
            selected_taus = unique_taus[::max(1, len(unique_taus) // 6)]

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Implied Volatility Fit", "Calibration Error (bps)"),
                                horizontal_spacing=0.08)

            colors = ['#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6']

            for i, tau in enumerate(selected_taus):
                mask = df_fit['Maturity'] == tau
                color = colors[i % len(colors)]
                label = f"τ={tau:.2f}"

                # Market IV
                fig.add_trace(go.Scatter(
                    x=df_fit[mask]['Strike'], y=df_fit[mask]['Market IV'] * 100,
                    mode='markers', name=f'{label} (mkt)',
                    marker=dict(color=color, size=6, symbol='circle-open', line=dict(width=1.5)),
                    legendgroup=label, showlegend=True
                ), row=1, col=1)

                # Model IV
                fig.add_trace(go.Scatter(
                    x=df_fit[mask]['Strike'], y=df_fit[mask]['Model IV'] * 100,
                    mode='lines', name=f'{label} (mdl)',
                    line=dict(color=color, width=2),
                    legendgroup=label, showlegend=True
                ), row=1, col=1)

                # Error
                fig.add_trace(go.Bar(
                    x=df_fit[mask]['Strike'], y=df_fit[mask]['Error (bps)'],
                    name=f'{label} err',
                    marker_color=color, opacity=0.7,
                    legendgroup=label, showlegend=False
                ), row=1, col=2)

            fig.update_layout(
                height=500,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(family='DM Sans', size=12),
                legend=dict(font=dict(size=10)),
                margin=dict(t=40, b=40),
            )
            fig.update_xaxes(title_text="Strike", row=1, col=1, gridcolor='rgba(100,100,100,0.2)')
            fig.update_xaxes(title_text="Strike", row=1, col=2, gridcolor='rgba(100,100,100,0.2)')
            fig.update_yaxes(title_text="Implied Vol (%)", row=1, col=1, gridcolor='rgba(100,100,100,0.2)')
            fig.update_yaxes(title_text="Error (bps)", row=1, col=2, gridcolor='rgba(100,100,100,0.2)')

            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 2: VANILLA PRICER
# =============================================================================

with tab_pricer:
    st.markdown("### European Option Pricer")

    if 'calib_result' not in st.session_state:
        st.warning("⚠️ Please calibrate a model first in the Calibration tab, or set parameters manually below.")

    # Allow manual parameter override
    use_calib = False
    if 'calib_result' in st.session_state:
        use_calib = st.checkbox("Use calibrated parameters", value=True)

    if use_calib and 'calib_result' in st.session_state:
        pricing_params = st.session_state['calib_result']['params']
        pricing_model = st.session_state['calib_model']
    else:
        pricing_model = model_choice
        pricing_params = dict(DEFAULT_PARAMS[model_choice])

        st.markdown("**Manual Parameters:**")
        param_cols = st.columns(min(len(pricing_params), 5))
        names = PARAM_NAMES[pricing_model]
        for i, name in enumerate(names):
            col = param_cols[i % len(param_cols)]
            lo, hi = PARAM_BOUNDS[pricing_model][name]
            pricing_params[name] = col.number_input(
                name, value=pricing_params[name],
                min_value=lo, max_value=hi,
                step=0.01, format="%.4f", key=f"manual_{name}"
            )

    st.markdown("---")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("**Option Contract**")
        opt_type = st.selectbox("Type", ['call', 'put'], key='vanilla_type')
        K_input = st.number_input("Strike (K)", value=S0, min_value=0.01, step=1.0, format="%.2f", key='vanilla_K')
        tau_input = st.number_input("Time to Maturity (years)", value=0.5, min_value=0.01, max_value=10.0,
                                    step=0.05, format="%.4f", key='vanilla_tau')

        price_btn = st.button("💰 Price Option", type="primary", key='price_vanilla')

    with col_result:
        if price_btn:
            try:
                price = price_european(pricing_model, S0, K_input, r, q, tau_input,
                                       pricing_params, option_type=opt_type, method='fft')
                iv = implied_vol(price, S0, K_input, r, q, tau_input, opt_type)

                # Greeks via finite differences (scaled for large S0)
                dS = S0 * 0.005  # 0.5% bump
                dtau = 1.0 / 365
                dr_bump = 0.0001

                p_up = price_european(pricing_model, S0 + dS, K_input, r, q, tau_input, pricing_params, opt_type)
                p_dn = price_european(pricing_model, S0 - dS, K_input, r, q, tau_input, pricing_params, opt_type)
                delta = (p_up - p_dn) / (2 * dS)
                gamma = (p_up - 2 * price + p_dn) / (dS ** 2)

                # Theta: price change per 1 calendar day (negative = time decay)
                if tau_input > dtau:
                    p_t = price_european(pricing_model, S0, K_input, r, q, tau_input - dtau, pricing_params, opt_type)
                    theta_1d = p_t - price  # dollar theta per day
                else:
                    theta_1d = 0.0

                # Rho: per 1% rate move (not per 1bp)
                p_r = price_european(pricing_model, S0, K_input, r + 0.01, q, tau_input, pricing_params, opt_type)
                rho_1pct = p_r - price  # dollar rho per 1% rate

                # Vega: per 1 vol point (bump v0 by ~1 vol point squared)
                params_v = dict(pricing_params)
                dv = 0.01  # ~1 vol point in variance terms
                if 'v0' in params_v:
                    params_v['v0'] += dv
                elif 'v01' in params_v:
                    params_v['v01'] += dv
                p_v = price_european(pricing_model, S0, K_input, r, q, tau_input, params_v, opt_type)
                vega = p_v - price  # dollar vega per 1 vol pt bump in v0

                st.markdown("**Pricing Result**")

                m1, m2 = st.columns(2)
                m1.metric("Price", f"{price:.4f}")
                m2.metric("Implied Vol", f"{iv * 100:.2f}%")

                m3, m4, m5 = st.columns(3)
                m3.metric("Delta", f"{delta:.4f}")
                m4.metric("Gamma", f"{gamma:.2e}")
                m5.metric("Theta /day", f"{theta_1d:.4f}")

                m6, m7 = st.columns(2)
                m6.metric("Rho /1%", f"{rho_1pct:.4f}")
                m7.metric("Vega /1vol", f"{vega:.4f}")

            except Exception as e:
                st.error(f"Pricing error: {e}")

    # Batch pricer
    st.markdown("---")
    st.markdown("### Batch Pricing")

    with st.expander("📋 Price multiple options at once"):
        batch_input = st.text_area(
            "Enter options (one per line): Type, Strike, Maturity",
            value="call, 90, 0.25\ncall, 100, 0.25\ncall, 110, 0.25\nput, 95, 0.5\nput, 100, 0.5\nput, 105, 0.5",
            height=150
        )

        if st.button("Price All", key='batch_price'):
            results = []
            for line in batch_input.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    try:
                        ot, k, t = parts[0].lower(), float(parts[1]), float(parts[2])
                        p = price_european(pricing_model, S0, k, r, q, t,
                                           pricing_params, option_type=ot, method='fft')
                        iv_val = implied_vol(p, S0, k, r, q, t, ot)
                        results.append({
                            'Type': ot.upper(),
                            'Strike': k,
                            'Maturity': t,
                            'Price': round(p, 4),
                            'IV (%)': round(iv_val * 100, 2),
                            'Moneyness': round(k / S0, 4),
                        })
                    except Exception as e:
                        results.append({
                            'Type': parts[0], 'Strike': parts[1], 'Maturity': parts[2],
                            'Price': 'ERROR', 'IV (%)': str(e), 'Moneyness': ''
                        })

            st.dataframe(pd.DataFrame(results), use_container_width=True)


# =============================================================================
# TAB 3: EXOTICS (Binary Options + Variance Swaps)
# =============================================================================

with tab_exotics:
    st.markdown("### Exotic Derivatives Pricer")

    if 'calib_result' in st.session_state:
        ex_params = st.session_state['calib_result']['params']
        ex_model = st.session_state['calib_model']
        st.info(f"Using calibrated {ex_model.replace('_', ' ').title()} parameters")
    else:
        ex_params = DEFAULT_PARAMS[model_choice]
        ex_model = model_choice
        st.warning("Using default parameters. Calibrate first for accurate pricing.")

    col_bin, col_var = st.columns(2)

    with col_bin:
        st.markdown("#### 🎰 Binary (Digital) Options")
        st.markdown("Cash-or-nothing binary options paying 1 unit if ITM at expiry.")

        bin_type = st.selectbox("Binary Type", ['call', 'put'], key='bin_type')
        bin_K = st.number_input("Strike", value=S0, min_value=0.01, step=1.0, key='bin_K')
        bin_tau = st.number_input("Maturity (yrs)", value=0.25, min_value=0.01, step=0.05, key='bin_tau')

        if st.button("Price Binary", type="primary", key='price_bin'):
            try:
                if bin_type == 'call':
                    bp = price_binary_call(ex_model, S0, bin_K, r, q, bin_tau, ex_params)
                else:
                    bp = price_binary_put(ex_model, S0, bin_K, r, q, bin_tau, ex_params)

                disc = np.exp(-r * bin_tau)
                prob = bp / disc  # risk-neutral probability

                c1, c2 = st.columns(2)
                c1.metric("Binary Price", f"{bp:.6f}")
                c2.metric("RN Probability", f"{prob * 100:.2f}%")

                # Binary option strip
                st.markdown("**Binary option strip across strikes:**")
                strikes = np.linspace(S0 * 0.85, S0 * 1.15, 30)
                bin_prices = []
                for k in strikes:
                    if bin_type == 'call':
                        bin_prices.append(price_binary_call(ex_model, S0, k, r, q, bin_tau, ex_params))
                    else:
                        bin_prices.append(price_binary_put(ex_model, S0, k, r, q, bin_tau, ex_params))

                fig_bin = go.Figure()
                fig_bin.add_trace(go.Scatter(
                    x=strikes, y=bin_prices,
                    mode='lines', line=dict(color='#6366f1', width=2.5),
                    fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'
                ))
                fig_bin.update_layout(
                    title=f"Binary {bin_type.title()} Price vs Strike (τ={bin_tau:.2f})",
                    xaxis_title="Strike", yaxis_title="Price",
                    height=350, template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.8)',
                    font=dict(family='DM Sans'),
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_bin, use_container_width=True)
            except Exception as e:
                st.error(f"Binary pricing error: {e}")

    with col_var:
        st.markdown("#### 📐 Variance Swaps")
        st.markdown("Fair variance and volatility strike for variance swaps.")

        vs_tau = st.number_input("Swap Maturity (yrs)", value=1.0, min_value=0.01, step=0.1, key='vs_tau')

        if st.button("Compute Var Swap", type="primary", key='price_vs'):
            try:
                fair_var = variance_swap_strike(ex_model, S0, r, q, vs_tau, ex_params)
                vol_strike = variance_swap_vol_strike(ex_model, S0, r, q, vs_tau, ex_params)

                c1, c2 = st.columns(2)
                c1.metric("Fair Variance", f"{fair_var:.6f}")
                c2.metric("Vol Strike", f"{vol_strike * 100:.2f}%")

                # Term structure
                st.markdown("**Variance swap term structure:**")
                taus = np.linspace(0.05, 5.0, 50)
                vol_strikes = [variance_swap_vol_strike(ex_model, S0, r, q, t, ex_params) * 100 for t in taus]
                fair_vars = [variance_swap_strike(ex_model, S0, r, q, t, ex_params) * 100 for t in taus]

                fig_vs = go.Figure()
                fig_vs.add_trace(go.Scatter(
                    x=taus, y=vol_strikes,
                    mode='lines', name='Vol Strike (%)',
                    line=dict(color='#10b981', width=2.5)
                ))
                fig_vs.update_layout(
                    title="Variance Swap Vol Strike Term Structure",
                    xaxis_title="Maturity (years)", yaxis_title="Vol Strike (%)",
                    height=350, template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.8)',
                    font=dict(family='DM Sans'),
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_vs, use_container_width=True)
            except Exception as e:
                st.error(f"Var swap error: {e}")


# =============================================================================
# TAB 4: VOL SURFACE VISUALIZATION
# =============================================================================

with tab_surface:
    st.markdown("### Model-Implied Volatility Surface")

    if 'calib_result' in st.session_state:
        surf_params = st.session_state['calib_result']['params']
        surf_model = st.session_state['calib_model']
    else:
        surf_params = DEFAULT_PARAMS[model_choice]
        surf_model = model_choice

    col_s1, col_s2, col_s3 = st.columns(3)
    k_min = col_s1.number_input("Min Strike", value=S0 * 0.7, key='surf_kmin')
    k_max = col_s2.number_input("Max Strike", value=S0 * 1.3, key='surf_kmax')
    n_strikes = col_s3.slider("# Strikes", 10, 50, 25, key='surf_nk')

    col_s4, col_s5 = st.columns(2)
    tau_min = col_s4.number_input("Min Maturity", value=0.05, key='surf_tmin')
    tau_max = col_s5.number_input("Max Maturity", value=2.0, key='surf_tmax')
    n_taus = 15

    if st.button("🌊 Generate Surface", type="primary", key='gen_surface'):
        with st.spinner("Computing implied volatility surface..."):
            strikes = np.linspace(k_min, k_max, n_strikes)
            taus = np.linspace(tau_min, tau_max, n_taus)

            iv_surface = np.zeros((n_taus, n_strikes))

            for i, tau in enumerate(taus):
                prices = price_european(surf_model, S0, strikes, r, q, tau,
                                        surf_params, option_type='call', method='fft')
                for j, (k, p) in enumerate(zip(strikes, np.atleast_1d(prices))):
                    try:
                        iv_surface[i, j] = implied_vol(p, S0, k, r, q, tau) * 100
                    except:
                        iv_surface[i, j] = np.nan

            # 3D surface
            fig_3d = go.Figure(data=[go.Surface(
                x=strikes, y=taus, z=iv_surface,
                colorscale='Viridis',
                colorbar=dict(title='IV (%)', titlefont=dict(size=12)),
                opacity=0.9,
            )])

            fig_3d.update_layout(
                title=f"Implied Vol Surface — {surf_model.replace('_', ' ').title()}",
                scene=dict(
                    xaxis_title='Strike',
                    yaxis_title='Maturity (yrs)',
                    zaxis_title='Implied Vol (%)',
                    bgcolor='rgba(15, 23, 42, 0.9)',
                    xaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                    yaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                    zaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                ),
                height=600,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans'),
                margin=dict(t=50, b=20),
            )

            st.plotly_chart(fig_3d, use_container_width=True)

            # Also show the smile at selected maturities
            fig_smile = go.Figure()
            for i, tau in enumerate(taus[::max(1, len(taus) // 5)]):
                idx = np.argmin(np.abs(taus - tau))
                fig_smile.add_trace(go.Scatter(
                    x=strikes / S0, y=iv_surface[idx, :],
                    mode='lines', name=f'τ = {tau:.2f}',
                    line=dict(width=2)
                ))

            fig_smile.update_layout(
                title="Implied Volatility Smile by Maturity",
                xaxis_title="Moneyness (K/S₀)",
                yaxis_title="Implied Vol (%)",
                height=400,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(family='DM Sans'),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_smile, use_container_width=True)


# =============================================================================
# TAB 5: PARAMETER INFO
# =============================================================================

with tab_params:
    st.markdown("### Model Specifications")

    st.markdown("""
    #### Heston (1993)
    The Heston model introduces stochastic variance following a CIR process:

    $$dS_t = (r - q) S_t \, dt + \\sqrt{v_t} \, S_t \, dW_t^S$$

    $$dv_t = \\kappa (\\theta - v_t) \, dt + \\xi \\sqrt{v_t} \, dW_t^v$$

    with $\\text{corr}(dW^S, dW^v) = \\rho$.

    **Parameters:** $v_0$ (initial variance), $\\kappa$ (mean reversion speed), $\\theta$ (long-run variance),
    $\\xi$ (vol of vol), $\\rho$ (correlation).

    **Feller condition:** $2\\kappa\\theta > \\xi^2$ ensures variance stays positive.
    """)

    st.markdown("""
    #### Bates (1996)
    Extends Heston with log-normal jumps in the asset price:

    $$dS_t = (r - q - \\lambda_J \\bar{k}) S_t \, dt + \\sqrt{v_t} \, S_t \, dW_t^S + J_t \, S_t \, dN_t$$

    where $\\ln(1 + J_t) \\sim N(\\mu_J, \\sigma_J^2)$ and $N_t$ is a Poisson process with intensity $\\lambda_J$.

    **Additional parameters:** $\\lambda_J$ (jump intensity), $\\mu_J$ (mean jump size), $\\sigma_J$ (jump vol).
    """)

    st.markdown("""
    #### Double Heston (Christoffersen, Heston & Jacobs, 2009)
    Two independent CIR variance processes, allowing richer term structures:

    $$dv_t^{(i)} = \\kappa_i (\\theta_i - v_t^{(i)}) \, dt + \\xi_i \\sqrt{v_t^{(i)}} \, dW_t^{v,i}, \\quad i = 1, 2$$

    Total instantaneous variance is $v_t = v_t^{(1)} + v_t^{(2)}$.

    **Parameters:** $(v_0^i, \\kappa_i, \\theta_i, \\xi_i, \\rho_i)$ for each factor $i \\in \\{1, 2\\}$.
    """)

    st.markdown("---")
    st.markdown("### Implementation Notes")

    st.markdown("""
    **Characteristic function:** Uses the Albrecher et al. (2007) "little Heston trap" formulation
    to avoid discontinuities in the complex logarithm.

    **FFT pricing:** Carr-Madan (1999) with Simpson's rule weighting, $N = 4096$ grid points,
    dampening parameter $\\alpha = 1.5$. Provides prices for a full strike grid in one FFT call.

    **Calibration:** Two-stage approach — differential evolution (global search) followed by
    L-BFGS-B (local polish). Soft penalty for Feller condition violation.

    **Binary options:** Priced via numerical differentiation of the vanilla call price surface.

    **Variance swaps:** Analytical formulas for all three models (Heston and Double Heston:
    conditional expectation of integrated variance; Bates: adds jump contribution).
    """)

    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    1. Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies, 6(2), 327–343.
    2. Bates, D.S. (1996). *Jumps and Stochastic Volatility.* Review of Financial Studies, 9(1), 69–107.
    3. Christoffersen, P., Heston, S., & Jacobs, K. (2009). *The Shape and Term Structure of the Index Option Smirk.* Management Science, 55(12), 1914–1932.
    4. Carr, P. & Madan, D. (1999). *Option Valuation Using the Fast Fourier Transform.* Journal of Computational Finance, 2(4), 61–73.
    5. Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). *The Little Heston Trap.* Wilmott Magazine.
    6. Gauthier, P. & Possamaï, D. (2011). *Efficient Simulation of the Double Heston Model.* EJOR.
    """)


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
    "Options Market Making Tool — Heston Family Models | "
    "Carr-Madan FFT Pricing | Differential Evolution Calibration"
    "</div>",
    unsafe_allow_html=True
)
