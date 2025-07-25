# earnings_dashboard.py (Streamlit webapp)

import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------- Helper Functions -------------------

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break

    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    return result.iloc[-1] if return_last_only else result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days, ivs = days[sort_idx], ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]: return ivs[0]
        if dte > days[-1]: return ivs[-1]
        return float(spline(dte))

    return term_spline

def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

def compute_recommendation(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()
    stock = yf.Ticker(ticker_symbol)
    if not stock.options:
        return "Error: No options found."

    try:
        exp_dates = filter_dates(list(stock.options))
    except:
        return "Error: Not enough valid expiration dates (need ≥ 45 days ahead)."

    options_chains = {date: stock.option_chain(date) for date in exp_dates}
    try:
        underlying_price = get_current_price(stock)
    except:
        return "Error: Couldn't fetch underlying price."

    atm_iv = {}
    straddle = None
    for i, (exp_date, chain) in enumerate(options_chains.items()):
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            continue

        call_strike = (calls['strike'] - underlying_price).abs().idxmin()
        put_strike = (puts['strike'] - underlying_price).abs().idxmin()
        call_iv = calls.loc[call_strike, 'impliedVolatility']
        put_iv = puts.loc[put_strike, 'impliedVolatility']
        atm_iv[exp_date] = (call_iv + put_iv) / 2.0

        if i == 0:
            call_mid = (calls.loc[call_strike, 'bid'] + calls.loc[call_strike, 'ask']) / 2.0
            put_mid = (puts.loc[put_strike, 'bid'] + puts.loc[put_strike, 'ask']) / 2.0
            straddle = call_mid + put_mid

    if not atm_iv:
        return "Error: Couldn't extract ATM IV data."

    today = datetime.today().date()
    dtes, ivs = [], []
    for exp_date, iv in atm_iv.items():
        dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days
        dtes.append(dte)
        ivs.append(iv)

    term_spline = build_term_structure(dtes, ivs)
    ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
    price_history = stock.history(period='3mo')
    iv30_rv30 = term_spline(30) / yang_zhang(price_history)
    avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
    expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else "N/A"

    result = {
        'avg_volume': avg_volume >= 1_500_000,
        'iv30_rv30': iv30_rv30 >= 1.25,
        'ts_slope_0_45': ts_slope_0_45 <= -0.00406,
        'expected_move': expected_move,
        'term_curve': (dtes, ivs),
        'price_history': price_history
    }
    return result

# ------------------- Streamlit App -------------------

st.set_page_config(page_title="Earnings Trade Analyzer", layout="centered")
st.title("📈 Earnings Trade Analyzer")

symbol = st.text_input("Enter Stock Symbol (e.g. AAPL)", value="AAPL")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result = compute_recommendation(symbol)

    if isinstance(result, str):
        st.error(result)
    else:
        st.markdown("### 🔍 Result Summary")
        st.markdown(f"**Expected Move**: {result['expected_move']}")

        st.markdown("**Checks:**")
        st.success("✅ Volume PASS" if result['avg_volume'] else "❌ Volume FAIL")
        st.success("✅ IV30/RV30 PASS" if result['iv30_rv30'] else "❌ IV30/RV30 FAIL")
        st.success("✅ Term Slope PASS" if result['ts_slope_0_45'] else "❌ Term Slope FAIL")

        if result['avg_volume'] and result['iv30_rv30'] and result['ts_slope_0_45']:
            st.markdown("### ✅ Final Recommendation: **RECOMMENDED**")
        elif result['ts_slope_0_45'] and (result['avg_volume'] or result['iv30_rv30']):
            st.markdown("### ⚠️ Final Recommendation: **CONSIDER**")
        else:
            st.markdown("### ❌ Final Recommendation: **AVOID**")

        # Plot IV Term Structure
        dtes, ivs = result['term_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dtes, y=ivs, mode='lines+markers', name='ATM IV'))
        fig.update_layout(title="ATM IV Term Structure", xaxis_title="Days to Expiration", yaxis_title="Implied Volatility")
        st.plotly_chart(fig, use_container_width=True)

        # Plot closing price chart
        price_df = result['price_history']
        price_chart = go.Figure()
        price_chart.add_trace(go.Scatter(x=price_df.index, y=price_df['Close'], name='Close'))
        price_chart.update_layout(title="3-Month Price History", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(price_chart, use_container_width=True)

        # Optional: Download CSV
        csv = price_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Historical Prices as CSV",
            data=csv,
            file_name=f'{symbol}_3mo_history.csv',
            mime='text/csv',
        )
