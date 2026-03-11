"""
Heston Family — Options Market Making Tool
============================================
Two views:
  - Trader View: Calibration, pricing, Greeks, vol surface (internal)
  - Customer View: Clean bid-offer quote interface (client-facing)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (price_european, price_binary_call, price_binary_put,
                    variance_swap_strike, variance_swap_vol_strike)
from calibration import (calibrate, implied_vol, bs_price, bs_vega,
                         PARAM_BOUNDS, PARAM_NAMES, DEFAULT_PARAMS)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Heston Market Making", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99,102,241,0.2); border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important; font-family: monospace !important; font-size: 1.4rem !important; }
    .info-box { background: rgba(99,102,241,0.1); border-left: 4px solid #6366f1;
                padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
    .bid-box { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
               border: 1px solid #10b981; border-radius: 14px; padding: 24px;
               text-align: center; }
    .offer-box { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
                 border: 1px solid #f43f5e; border-radius: 14px; padding: 24px;
                 text-align: center; }
    .quote-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 6px; }
    .quote-value { color: #f1f5f9; font-family: monospace; font-size: 1.8rem; font-weight: 700; }
</style>""", unsafe_allow_html=True)


# =============================================================================
# Bid-offer computation (used by both views)
# =============================================================================

def compute_bid_offer(product_type, S0, K, r, q, tau, model_results,
                      option_type='call', spread_bps=50):
    prices, ivs = [], []
    for mdl, res in model_results.items():
        try:
            if product_type == 'vanilla':
                p = price_european(mdl, S0, K, r, q, tau, res['params'], option_type, 'fft')
            elif product_type == 'binary_call':
                p = price_binary_call(mdl, S0, K, r, q, tau, res['params'])
            elif product_type == 'binary_put':
                p = price_binary_put(mdl, S0, K, r, q, tau, res['params'])
            elif product_type == 'varswap':
                p = variance_swap_vol_strike(mdl, S0, r, q, tau, res['params'])
            else:
                continue
            if np.isfinite(p) and p > 0:
                prices.append(p)
                if product_type == 'vanilla':
                    iv = implied_vol(p, S0, K, r, q, tau, option_type)
                    if np.isfinite(iv): ivs.append(iv)
        except Exception:
            continue
    if not prices: return None
    mid = np.mean(prices)
    model_unc = (max(prices) - min(prices)) / 2 if len(prices) > 1 else 0
    if product_type == 'vanilla':
        base = max(bs_vega(S0, K, r, q, tau, np.mean(ivs) if ivs else 0.2) * spread_bps / 10000, mid * 0.005)
    elif product_type in ('binary_call', 'binary_put'):
        base = max(0.005, mid * 0.02)
    else:
        base = max(0.002, mid * 0.01)
    hs = max(base, model_unc)
    return {'mid': mid, 'bid': max(mid - hs, 0), 'offer': mid + hs,
            'spread': mid + hs - max(mid - hs, 0), 'ivs': ivs,
            'n_models': len(prices), 'model_range': max(prices) - min(prices) if len(prices) > 1 else 0}


def calibrate_all_models(market_data, S0, r, q, max_iter=100, pop_size=3):
    results = {}
    for mdl in ['heston', 'bates', 'double_heston']:
        try:
            res = calibrate(model=mdl, S0=S0, r=r, q=q, market_data=market_data,
                            loss_type='ivrmse', maxiter=max_iter, popsize=pop_size)
            if res['success']: results[mdl] = res
        except Exception:
            pass
    return results


# =============================================================================
# View toggle
# =============================================================================

st.markdown("## 📊 Heston Family — Options Market Making")

view = st.radio("", ["⚙️ Trader View", "🏦 Customer View"],
                horizontal=True, label_visibility="collapsed")


# #############################################################################
# CUSTOMER VIEW
# #############################################################################

if view == "🏦 Customer View":

    st.markdown("""<div class="info-box">
    Welcome to our <b>Options Desk</b>. Select a product, enter the contract details,
    and click <b>Get Quote</b> to receive a tradeable bid-offer price.
    </div>""", unsafe_allow_html=True)

    # Show spot price (read-only info for the customer)
    S0 = st.session_state.get('S0_stored', 100.0)
    r = st.session_state.get('r_stored', 0.02)
    q = st.session_state.get('q_stored', 0.01)

    st.markdown(f"**Underlying spot price: {S0:,.2f}**")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("### Product Details")
        product = st.selectbox("Product Type", [
            "European Call", "European Put",
            "Binary Call (cash-or-nothing)", "Binary Put (cash-or-nothing)",
            "Variance Swap"], key='c_prod')
        if product != "Variance Swap":
            c_K = st.number_input("Strike (K)", value=round(S0, 2), min_value=0.01, step=1.0, key='c_K')
        else:
            c_K = 0
        c_tau = st.number_input("Maturity (years)", value=0.5, min_value=0.01, max_value=10.0, step=0.05, key='c_tau')
        quote_btn = st.button("📋 Get Quote", type="primary", use_container_width=True, key='c_btn')

    with col_out:
        if quote_btn:
            mm = st.session_state.get('mm_results')
            mkt = st.session_state.get('market_data_stored')

            # Auto-calibrate if needed
            if mm is None and mkt is not None:
                with st.spinner("Initializing pricing engine..."):
                    # FIX: Fetch the stored slider settings
                    stored_iter = st.session_state.get('maxiter_stored', 100)
                    stored_pop = st.session_state.get('popsize_stored', 3)
                    
                    mm = calibrate_all_models(mkt, S0, r, q, max_iter=int(stored_iter), pop_size=int(stored_pop))
                    st.session_state['mm_results'] = mm

            if mm is None or len(mm) == 0:
                st.error("Pricing engine not ready. Please ask the trading desk to load market data.")
            else:
                pmap = {"European Call": ('vanilla', 'call'), "European Put": ('vanilla', 'put'),
                        "Binary Call (cash-or-nothing)": ('binary_call', 'call'),
                        "Binary Put (cash-or-nothing)": ('binary_put', 'put'),
                        "Variance Swap": ('varswap', 'call')}
                pt, ot = pmap[product]
                res = compute_bid_offer(pt, S0, c_K, r, q, c_tau, mm, ot)

                if res is None:
                    st.error("Unable to price this product.")
                else:
                    st.markdown("### Tradeable Quote")
                    is_vs = (pt == 'varswap')
                    fmt = lambda x: f"{x*100:.2f}%" if is_vs else f"{x:,.4f}"

                    c1, c2 = st.columns(2)
                    c1.markdown(f'<div class="bid-box"><div class="quote-label">BID</div>'
                                f'<div class="quote-value">{fmt(res["bid"])}</div></div>',
                                unsafe_allow_html=True)
                    c2.markdown(f'<div class="offer-box"><div class="quote-label">OFFER</div>'
                                f'<div class="quote-value">{fmt(res["offer"])}</div></div>',
                                unsafe_allow_html=True)

                    mc1, mc2 = st.columns(2)
                    mc1.metric("Mid", fmt(res['mid']))
                    mc2.metric("Spread", fmt(res['spread']))

                    if res['ivs']:
                        st.metric("Indicative Vol", f"{np.mean(res['ivs'])*100:.2f}%")

                    st.caption("Quotes are indicative and subject to market conditions.")

    # Batch
    st.markdown("---")
    with st.expander("📋 Batch Quote Request"):
        batch = st.text_area("One per line: type, strike, maturity",
            value="call, 4200, 0.25\ncall, 4450, 0.50\nput, 4200, 0.50\nbinary_call, 4450, 0.25\nvarswap, 0, 1.0",
            height=120, key='c_batch')
        if st.button("Get All Quotes", type="primary", key='c_batch_btn'):
            mm = st.session_state.get('mm_results')
            if mm is None:
                st.error("Pricing engine not ready.")
            else:
                rows = []
                for line in batch.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        try:
                            pt, k, t = parts[0].lower(), float(parts[1]), float(parts[2])
                            if pt in ('call','put'):
                                r2 = compute_bid_offer('vanilla', S0, k, r, q, t, mm, pt)
                            elif pt == 'binary_call':
                                r2 = compute_bid_offer('binary_call', S0, k, r, q, t, mm)
                            elif pt == 'binary_put':
                                r2 = compute_bid_offer('binary_put', S0, k, r, q, t, mm)
                            elif pt == 'varswap':
                                r2 = compute_bid_offer('varswap', S0, 0, r, q, t, mm)
                            else: r2 = None
                            if r2:
                                is_vs = pt == 'varswap'
                                f = lambda x: f"{x*100:.2f}%" if is_vs else f"{x:.4f}"
                                rows.append({'Product': pt.upper(), 'Strike': '-' if is_vs else k,
                                            'Maturity': t, 'Bid': f(r2['bid']), 'Offer': f(r2['offer']),
                                            'Spread': f(r2['spread'])})
                        except: pass
                if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)


# #############################################################################
# TRADER VIEW
# #############################################################################

if view == "⚙️ Trader View":

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Market Parameters")
        
        # 1. Retrieve stored values (or use defaults)
        saved_S0 = st.session_state.get('S0_stored', 100.0)
        saved_r  = st.session_state.get('r_stored', 0.02)
        saved_q  = st.session_state.get('q_stored', 0.01)

        S0 = st.number_input("Spot Price (S₀)", value=float(saved_S0), min_value=0.01, step=1.0, format="%.2f", key='t_S0')
        r = st.number_input("Risk-Free Rate (r)", value=float(saved_r), min_value=-0.05, max_value=0.30, step=0.005, format="%.4f", key='t_r')
        q = st.number_input("Dividend Yield (q)", value=float(saved_q), min_value=0.0, max_value=0.20, step=0.005, format="%.4f", key='t_q')

        st.markdown("---")
        st.markdown("## 🔬 Model")
        
        # 2. Handle Model Selectbox State
        model_opts = ['heston', 'bates', 'double_heston']
        saved_model = st.session_state.get('model_stored', 'heston')
        model_idx = model_opts.index(saved_model) if saved_model in model_opts else 0
        
        model_choice = st.selectbox("Model", model_opts, index=model_idx,
            format_func=lambda x: {'heston':'Heston (1993)','bates':'Bates (1996) — jumps',
                                    'double_heston':'Double Heston (2009)'}[x], key='t_model')

        st.markdown("---")
        st.markdown("## 📈 Calibration")
        
        # 3. Handle Loss Selectbox State
        loss_opts = ['ivrmse', 'price_abs', 'iv_relative', 'price_rmse']
        saved_loss = st.session_state.get('loss_stored', 'ivrmse')
        loss_idx = loss_opts.index(saved_loss) if saved_loss in loss_opts else 0
        
        loss_type = st.selectbox("Loss", loss_opts, index=loss_idx,
            format_func=lambda x: {'ivrmse':'IV RMSE','price_abs':'Vega-wtd Price (fast)',
                                    'iv_relative':'IV RMSE relative','price_rmse':'Price RMSE'}[x], key='t_loss')
        
        # 4. Handle Slider States
        saved_maxiter = st.session_state.get('maxiter_stored', 100)
        saved_popsize = st.session_state.get('popsize_stored', 3)
        
        maxiter = st.slider("Max Iter", 50, 500, value=int(saved_maxiter), step=50, key='t_iter')
        popsize = st.slider("Restarts", 2, 10, value=int(saved_popsize), step=1, key='t_pop')

    # Store ALL current values back into session_state immediately after the sidebar
    st.session_state['S0_stored'] = S0
    st.session_state['r_stored'] = r
    st.session_state['q_stored'] = q
    st.session_state['model_stored'] = model_choice
    st.session_state['loss_stored'] = loss_type
    st.session_state['maxiter_stored'] = maxiter
    st.session_state['popsize_stored'] = popsize

    # Tabs
    tab_cal, tab_van, tab_exo, tab_surf, tab_doc = st.tabs([
        "🎯 Calibration", "💰 Vanilla Pricer", "🔮 Exotics", "🌊 Vol Surface", "📋 Documentation"])

    # ── CALIBRATION ──────────────────────────────────────────────
    with tab_cal:
        st.markdown("### Market Data")
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader("Upload vol surface (.xlsx/.csv)", type=['xlsx','xls','csv'], key='t_file')
        with col2:
            use_sample = st.checkbox("Use sample Heston surface", value=uploaded is None, key='t_sample')

        market_data = None
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
                col_map = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl in ['k','strike','strike_price','strikes']: col_map['K'] = c
                    elif cl in ['tau','t','maturity','expiry','tte','time_to_maturity']: col_map['tau'] = c
                    elif cl in ['iv','implied_vol','impliedvol','implied_volatility','vol','sigma','implvol','imp_vol','ivol']: col_map['iv'] = c
                    elif cl in ['type','option_type','cp','call_put']: col_map['type'] = c
                if 'K' in col_map and 'tau' in col_map and 'iv' in col_map:
                    all_iv = pd.to_numeric(df[col_map['iv']], errors='coerce').dropna()
                    is_pct = all_iv.median() > 1.0
                    if is_pct: st.info("Detected IV in percentage — converting.")
                    market_data = []
                    for _, row in df.iterrows():
                        iv = float(row[col_map['iv']])
                        if is_pct: iv /= 100
                        ot = 'call'
                        if 'type' in col_map:
                            t = str(row[col_map['type']]).lower().strip()
                            if t in ['put','p']: ot = 'put'
                        market_data.append({'K':float(row[col_map['K']]), 'tau':float(row[col_map['tau']]),
                                           'market_iv':iv, 'option_type':ot, 'weight':1.0})
                    st.success(f"✅ {len(market_data)} data points loaded.")
                else:
                    st.error(f"Cannot map columns. Found: {col_map}")
            except Exception as e:
                st.error(str(e))

        if use_sample and market_data is None:
            sp = {'v0':0.04,'kappa':2.0,'theta':0.04,'sigma':0.5,'rho':-0.7}
            market_data = []
            for tau in [0.08,0.17,0.25,0.5,0.75,1.0,1.5,2.0]:
                strikes = S0 * np.linspace(0.8, 1.2, 15)
                prices = price_european('heston', S0, strikes, r, q, tau, sp, 'call', 'fft')
                for K_v, p_v in zip(strikes, np.atleast_1d(prices)):
                    iv_v = implied_vol(p_v, S0, K_v, r, q, tau)
                    if np.isfinite(iv_v) and iv_v > 0.01:
                        market_data.append({'K':round(K_v,2),'tau':tau,'market_iv':round(iv_v+np.random.normal(0,0.001),6),
                                           'option_type':'call','weight':1.0})
            st.info(f"Sample Heston surface: {len(market_data)} points")

        if market_data:
            st.session_state['market_data_stored'] = market_data
            with st.expander("Preview", expanded=False):
                st.dataframe(pd.DataFrame(market_data).head(20), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Calibrate Selected Model", type="primary", use_container_width=True, key='t_cal1'):
                    with st.spinner(f"Calibrating {model_choice}..."):
                        result = calibrate(model=model_choice, S0=S0, r=r, q=q, market_data=market_data,
                                          loss_type=loss_type, maxiter=maxiter, popsize=popsize)
                        st.session_state['calib_result'] = result
                        st.session_state['calib_model'] = model_choice
                        st.session_state['calib_market_data'] = market_data
            with c2:
                if st.button("🔄 Calibrate All 3 Models (for Quotes)", type="secondary", use_container_width=True, key='t_cal_all'):
                    with st.spinner("Calibrating Heston, Bates & Double Heston..."):
                        # FIX: Pass the maxiter and popsize from the sliders!
                        mm = calibrate_all_models(market_data, S0, r, q, max_iter=maxiter, pop_size=popsize)
                        
                        st.session_state['mm_results'] = mm
                        st.session_state['mm_models_calibrated'] = True
                        
                        if model_choice in mm:
                            st.session_state['calib_result'] = mm[model_choice]
                            st.session_state['calib_model'] = model_choice
                            st.session_state['calib_market_data'] = market_data
                            
                        st.success(f"✅ {len(mm)} models ready: {', '.join(m.replace('_',' ').title() for m in mm)}")

        if 'calib_result' in st.session_state:
            result = st.session_state['calib_result']
            st.markdown("### Calibration Results")
            
            # --- UI UPGRADE: Beautiful Metric Cards ---
            params = result['params']
            n_params = len(params)
            metric_cols = st.columns(min(n_params + 1, 6))
            metric_cols[0].metric("RMSE", f"{result['rmse']:.6f}")
            
            param_items = list(params.items())
            col_idx = 1
            for name, val in param_items:
                if col_idx >= len(metric_cols): break
                label_map = {
                    'v0': 'v₀', 'kappa': 'κ', 'theta': 'θ', 'sigma': 'ξ', 'rho': 'ρ',
                    'lambda_j': 'λⱼ', 'mu_j': 'μⱼ', 'sigma_j': 'σⱼ',
                    'v01': 'v₀¹', 'kappa1': 'κ¹', 'theta1': 'θ¹', 'sigma1': 'ξ¹', 'rho1': 'ρ¹',
                    'v02': 'v₀²', 'kappa2': 'κ²', 'theta2': 'θ²', 'sigma2': 'ξ²', 'rho2': 'ρ²',
                }
                metric_cols[col_idx].metric(label_map.get(name, name), f"{val:.4f}")
                col_idx += 1

            # Handle extra parameters for Bates/Double Heston
            if n_params > 5:
                metric_cols2 = st.columns(min(n_params - 4, 6))
                for i, (name, val) in enumerate(param_items[5:]):
                    if i >= len(metric_cols2): break
                    label_map = {
                        'lambda_j': 'λⱼ', 'mu_j': 'μⱼ', 'sigma_j': 'σⱼ',
                        'v02': 'v₀²', 'kappa2': 'κ²', 'theta2': 'θ²', 'sigma2': 'ξ²', 'rho2': 'ρ²',
                    }
                    metric_cols2[i].metric(label_map.get(name, name), f"{val:.4f}")

            if st.session_state['calib_model'] in ['heston','bates']:
                f = 2*params['kappa']*params['theta'] - params['sigma']**2
                if f > 0: st.success(f"✅ Feller condition satisfied: 2κθ - ξ² = {f:.4f} > 0")
                else: st.warning(f"⚠️ Feller condition violated: 2κθ - ξ² = {f:.4f} < 0")

            # --- UI UPGRADE: Enhanced Plotly Chart ---
            if result['model_ivs'] is not None:
                # FETCH THE SAVED DATA HERE
                plot_data = st.session_state.get('calib_market_data', market_data) 
                
                df_fit = pd.DataFrame({
                    'Strike': [m['K'] for m in plot_data],
                    'Tau': [m['tau'] for m in plot_data],
                    'Market': result['market_ivs'] * 100, 
                    'Model': result['model_ivs'] * 100
                })
                df_fit = df_fit.dropna()
                
                if len(df_fit) > 0:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Implied Volatility Fit", "Calibration Error (bps)"), horizontal_spacing=0.08)
                    taus = sorted(df_fit['Tau'].unique())
                    sel = taus[::max(1, len(taus)//6)]
                    colors = ['#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6']
                    
                    for i, t in enumerate(sel):
                        s = df_fit[df_fit['Tau'] == t]
                        c = colors[i % len(colors)]
                        lb = f"τ={t:.2f}"
                        
                        # Market points as open circles
                        fig.add_trace(go.Scatter(x=s['Strike'], y=s['Market'], mode='markers', name=f'{lb} mkt',
                            marker=dict(color=c, size=6, symbol='circle-open', line=dict(width=1.5)), legendgroup=lb), row=1, col=1)
                        
                        # Model fit as solid lines
                        fig.add_trace(go.Scatter(x=s['Strike'], y=s['Model'], mode='lines', name=f'{lb} mdl',
                            line=dict(color=c, width=2), legendgroup=lb), row=1, col=1)
                        
                        # Error bars
                        fig.add_trace(go.Bar(x=s['Strike'], y=(s['Model'] - s['Market']) * 100, name=f'{lb} err',
                            marker_color=c, opacity=0.7, legendgroup=lb, showlegend=False), row=1, col=2)
                            
                    fig.update_layout(
                        height=500, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15, 23, 42, 0.8)', margin=dict(t=40, b=40),
                        font=dict(family='DM Sans', size=12), legend=dict(font=dict(size=10))
                    )
                    
                    fig.update_xaxes(title_text="Strike", gridcolor='rgba(100,100,100,0.2)')
                    fig.update_yaxes(title_text="Implied Vol (%)", row=1, col=1, gridcolor='rgba(100,100,100,0.2)')
                    fig.update_yaxes(title_text="Error (bps)", row=1, col=2, gridcolor='rgba(100,100,100,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)

    # ── VANILLA PRICER ───────────────────────────────────────────
    with tab_van:
        st.markdown("### European Option Pricer")
        if 'calib_result' not in st.session_state:
            st.warning("Calibrate a model first.")
        else:
            pp = st.session_state['calib_result']['params']
            pm = st.session_state['calib_model']
            ci, co = st.columns([1,1])
            with ci:
                ot = st.selectbox("Type", ['call','put'], key='v_type')
                vK = st.number_input("Strike", value=S0, min_value=0.01, step=1.0, key='v_K')
                vt = st.number_input("Maturity (yrs)", value=0.5, min_value=0.01, max_value=10.0, step=0.05, key='v_tau')
                if st.button("💰 Price", type="primary", use_container_width=True, key='v_btn'):
                    p = price_european(pm, S0, vK, r, q, vt, pp, ot, 'fft')
                    iv = implied_vol(p, S0, vK, r, q, vt, ot)
                    dS = S0*0.005
                    pu = price_european(pm,S0+dS,vK,r,q,vt,pp,ot); pd2 = price_european(pm,S0-dS,vK,r,q,vt,pp,ot)
                    delta = (pu-pd2)/(2*dS); gamma = (pu-2*p+pd2)/(dS**2)
                    dt = 1/365
                    theta = (price_european(pm,S0,vK,r,q,vt-dt,pp,ot)-p) if vt>dt else 0
                    rho_g = price_european(pm,S0,vK,r+0.01,q,vt,pp,ot)-p
                    pp2 = dict(pp); dv=0.01
                    if 'v0' in pp2: pp2['v0']+=dv
                    elif 'v01' in pp2: pp2['v01']+=dv
                    vega = price_european(pm,S0,vK,r,q,vt,pp2,ot)-p
                    with co:
                        st.markdown("### Pricing Result")
                        m1,m2 = st.columns(2); m1.metric("Price",f"{p:.4f}"); m2.metric("Implied Vol",f"{iv*100:.2f}%")
                        m3,m4,m5 = st.columns(3); m3.metric("Delta",f"{delta:.4f}"); m4.metric("Gamma",f"{gamma:.2e}"); m5.metric("Theta/day",f"{theta:.4f}")
                        m6,m7 = st.columns(2); m6.metric("Rho/1%",f"{rho_g:.4f}"); m7.metric("Vega/1vol",f"{vega:.4f}")

            # Batch
            st.markdown("---")
            with st.expander("Batch Pricing"):
                bi = st.text_area("Type, Strike, Maturity",
                    value="call, 90, 0.25\ncall, 100, 0.25\nput, 95, 0.5", height=100, key='v_batch')
                if st.button("Price All", key='v_batch_btn'):
                    rows = []
                    for line in bi.strip().split('\n'):
                        pts = [x.strip() for x in line.split(',')]
                        if len(pts)>=3:
                            try:
                                o,k,t = pts[0].lower(),float(pts[1]),float(pts[2])
                                p = price_european(pm,S0,k,r,q,t,pp,o,'fft')
                                iv = implied_vol(p,S0,k,r,q,t,o)
                                rows.append({'Type':o.upper(),'Strike':k,'Mat':t,'Price':round(p,4),'IV%':round(iv*100,2)})
                            except: pass
                    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── EXOTICS ──────────────────────────────────────────────────
    with tab_exo:
        if 'calib_result' not in st.session_state:
            st.warning("Calibrate first.")
        else:
            ep = st.session_state['calib_result']['params']
            em = st.session_state['calib_model']
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🎰 Binary Options")
                bt = st.selectbox("Type",['call','put'],key='b_type')
                bK = st.number_input("Strike",value=S0,min_value=0.01,step=1.0,key='b_K')
                bT = st.number_input("Maturity",value=0.25,min_value=0.01,step=0.05,key='b_tau')
                if st.button("Price Binary",type="primary",key='b_btn'):
                    bp = price_binary_call(em,S0,bK,r,q,bT,ep) if bt=='call' else price_binary_put(em,S0,bK,r,q,bT,ep)
                    d = np.exp(-r*bT)
                    m1,m2 = st.columns(2); m1.metric("Binary Price",f"{bp:.6f}"); m2.metric("RN Prob",f"{bp/d*100:.2f}%")
                    ks = np.linspace(S0*0.85,S0*1.15,30)
                    bps = [price_binary_call(em,S0,k,r,q,bT,ep) if bt=='call' else price_binary_put(em,S0,k,r,q,bT,ep) for k in ks]
                    fig = go.Figure(go.Scatter(x=ks,y=bps,mode='lines',line=dict(color='#6366f1',width=2.5),fill='tozeroy',fillcolor='rgba(99,102,241,0.1)'))
                    fig.update_layout(title=f"Binary {bt} strip",xaxis_title="Strike",yaxis_title="Price",height=300,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(15,23,42,0.8)')
                    st.plotly_chart(fig,use_container_width=True)
            with c2:
                st.markdown("#### 📐 Variance Swaps")
                vT = st.number_input("Maturity",value=1.0,min_value=0.01,step=0.1,key='vs_tau')
                if st.button("Compute",type="primary",key='vs_btn'):
                    fv = variance_swap_strike(em,S0,r,q,vT,ep)
                    vs = variance_swap_vol_strike(em,S0,r,q,vT,ep)
                    m1,m2 = st.columns(2); m1.metric("Fair Var",f"{fv:.6f}"); m2.metric("Vol Strike",f"{vs*100:.2f}%")
                    ts = np.linspace(0.05,5,50)
                    vss = [variance_swap_vol_strike(em,S0,r,q,t,ep)*100 for t in ts]
                    fig = go.Figure(go.Scatter(x=ts,y=vss,mode='lines',line=dict(color='#10b981',width=2.5)))
                    fig.update_layout(title="Term Structure",xaxis_title="Maturity",yaxis_title="Vol Strike (%)",height=300,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(15,23,42,0.8)')
                    st.plotly_chart(fig,use_container_width=True)

    # ── VOL SURFACE ──────────────────────────────────────────────
    with tab_surf:
        if 'calib_result' not in st.session_state:
            st.warning("Calibrate first.")
        else:
            sp = st.session_state['calib_result']['params']
            sm = st.session_state['calib_model']
            c1,c2,c3,c4 = st.columns(4)
            kmin = c1.number_input("Min K",value=S0*0.7,key='s_kmin')
            kmax = c2.number_input("Max K",value=S0*1.3,key='s_kmax')
            tmin = c3.number_input("Min T",value=0.05,key='s_tmin')
            tmax = c4.number_input("Max T",value=2.0,key='s_tmax')
            if st.button("🌊 Generate",type="primary",key='s_btn'):
                with st.spinner("Computing..."):
                    ks = np.linspace(kmin,kmax,25); ts = np.linspace(tmin,tmax,15)
                    iv_mat = np.zeros((15,25))
                    for i,t in enumerate(ts):
                        ps = price_european(sm,S0,ks,r,q,t,sp,'call','fft')
                        for j,(k,p) in enumerate(zip(ks,np.atleast_1d(ps))):
                            try: iv_mat[i,j] = implied_vol(p,S0,k,r,q,t)*100
                            except: iv_mat[i,j] = np.nan
                    fig = go.Figure(go.Surface(x=ks,y=ts,z=iv_mat,colorscale='Viridis',opacity=0.9))
                    fig.update_layout(title="Implied Vol Surface",scene=dict(xaxis_title='Strike',yaxis_title='Maturity',zaxis_title='IV(%)'),
                                      height=550,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig,use_container_width=True)

    # ── DOCUMENTATION ────────────────────────────────────────────
    with tab_doc:
        st.markdown("""
### Model Specifications

**Heston (1993):** $dS = (r-q)S\\,dt + \\sqrt{v}S\\,dW^S$, $dv = \\kappa(\\theta-v)dt + \\xi\\sqrt{v}dW^v$, corr $= \\rho$

**Bates (1996):** Heston + Merton log-normal jumps ($\\lambda_J, \\mu_J, \\sigma_J$)

**Double Heston (2009):** Two independent CIR variance factors

### Implementation
- **Pricing:** Carr-Madan FFT (N=4096, Simpson weights)
- **Char. function:** Albrecher et al. (2007) little Heston trap
- **Calibration:** Multi-start Nelder-Mead + L-BFGS-B
- **Binary options:** Numerical differentiation of call prices
- **Variance swaps:** Analytical formula
- **Market making:** Bid-offer from 3-model ensemble + vega-weighted spread
        """)

# Footer
st.markdown("---")
st.caption("Heston Family — Options Market Making Tool | Carr-Madan FFT | Multi-Model Calibration")
