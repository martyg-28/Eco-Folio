# -*- coding: utf-8 -*-

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Eco-Folio: Financial Analysis Tool")

# Title & Introduction
st.markdown("""
    <h1 style="text-align: center; color: #2E8B57; font-size: 50px; font-weight: bold;">Eco-Folio</h1>
    <h2 style="text-align: center; color: #00008B;">Financial Analysis Tool</h2>
    <hr style="height: 4px; background-color: #2E8B57;">
""", unsafe_allow_html=True)

# Function to fetch stock data
def get_stock_data(stock_symbols, start_date, end_date):
    stock_symbols = [s.strip().upper() for s in stock_symbols if s]
    if not stock_symbols:
        st.error("Please enter at least one valid stock symbol.")
        return pd.DataFrame()

    try:
        all_data = yf.download(stock_symbols, start=start_date, end=end_date, group_by='ticker')

        if isinstance(all_data.columns, pd.MultiIndex):
            price_df = pd.DataFrame({stock: all_data[stock]['Adj Close'] for stock in stock_symbols if 'Adj Close' in all_data[stock]})
        else:
            price_df = all_data.get('Adj Close', all_data.get('Close', pd.DataFrame()))

        if price_df.empty:
            st.error("No price data available. Check stock symbols or date range.")
            return pd.DataFrame()

        return price_df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to analyze stocks
def analyze_stocks(price):
    if price.empty:
        st.error("No price data. Cannot perform analysis.")
        return {}

    log_returns = np.log(price / price.shift(1)).dropna()
    stock_stats = {
        stock: {
            "risk": round(log_returns[stock].std(), 5),
            "expected_return": round(log_returns[stock].mean(), 5),
            "cagr": round(((price[stock].iloc[-1] / price[stock].iloc[0]) ** (1 / ((end_date - start_date).days / 365)) - 1), 5)
        } for stock in price.columns
    }

    stock_stats["correlation"] = price.corr()
    return stock_stats

# Function for portfolio optimization & Efficient Frontier
def portfolio_analysis(price):
    log_returns = np.log(price / price.shift(1)).dropna()
    num_stocks = len(price.columns)

    num_portfolios = 20000
    all_weights = np.random.rand(num_portfolios, num_stocks)
    all_weights /= all_weights.sum(axis=1)[:, np.newaxis]

    ret_arr = np.dot(all_weights, log_returns.mean() * 252)
    vol_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, log_returns.cov() * 252, all_weights))
    sharpe_arr = ret_arr / vol_arr  

    max_sharpe_idx = sharpe_arr.argmax()
    return all_weights[max_sharpe_idx], ret_arr[max_sharpe_idx], vol_arr[max_sharpe_idx], sharpe_arr[max_sharpe_idx]

# User inputs
stock_symbols = [st.sidebar.text_input(f'Enter stock symbol {i+1}', '').strip().upper() for i in range(5)]
date_range = st.sidebar.date_input('Select date range', [pd.to_datetime('2019-01-01'), pd.to_datetime('2023-10-01')])

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid date range.")
    start_date, end_date = None, None

calculate = st.sidebar.button('Analyze Stocks')

if calculate:
    price = get_stock_data(stock_symbols, start_date, end_date)
    if price.empty:
        st.stop()

    stock_stats = analyze_stocks(price)

    # **Stock Metrics**
    for stock, stats in stock_stats.items():
        if stock != "correlation":
            st.markdown(f"### {stock} Metrics")
            st.write(f"Annualized Risk: {stats['risk'] * 100:.2f}%")
            st.write(f"Expected Return: {stats['expected_return'] * 100:.2f}%")
            st.write(f"CAGR: {stats['cagr'] * 100:.2f}%")

    # **Correlation Matrix with Explainer**
    if "correlation" in stock_stats:
        st.markdown("### Correlation Matrix")
        st.dataframe(stock_stats["correlation"].style.background_gradient(cmap='coolwarm').format("{:.2f}"))
        st.markdown("""
        - **1**: Perfect positive correlation (stocks move together).
        - **-1**: Perfect negative correlation (one stock rises, the other falls).
        - **0**: No correlation (movements are unrelated).
        """)

    # **Portfolio Optimization & Visualization**
    optimized_weights, optimized_ret, optimized_vol, optimized_sr = portfolio_analysis(price)

    # **Fix for Pie Chart Issue**
    if optimized_weights is not None and len(optimized_weights) == len(price.columns):
        st.markdown("### Optimized Portfolio Weights")
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, _ = ax.pie(optimized_weights, labels=price.columns, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        st.write("These weights optimize the Sharpe Ratio, balancing return vs. risk.")
    else:
        st.error("Error: Could not generate portfolio weights. Please check your inputs.")

    # **Optimized Portfolio Stats**
    st.markdown("### Optimized Portfolio Return")
    st.write(f"{optimized_ret * 100:.2f}% - Expected annual return of the optimized portfolio.")

    st.markdown("### Optimized Portfolio Volatility")
    st.write(f"{optimized_vol * 100:.2f}% - Standard deviation of returns, representing risk.")

    st.markdown("### Optimized Portfolio Sharpe Ratio")
    st.write(f"{optimized_sr:.4f} - A measure of risk-adjusted return.")

    # **Efficient Frontier with Explainer**
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.6, edgecolors="w")
    ax.scatter(optimized_vol, optimized_ret, c='red', s=100, edgecolors="k", label='Optimized Portfolio')
    ax.set_xlabel("Volatility (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    plt.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig)

    st.markdown("""
    - **Points**: Represent different portfolios with various stock combinations.
    - **Color**: Shows Sharpe Ratio (yellow = better risk-adjusted return).
    - **Red Dot**: Optimized portfolio with the best risk-adjusted return.
    """)

    # **ESG Considerations Section**
    st.markdown("### ESG Considerations")
    for stock in stock_symbols:
        if stock:
            st.markdown(f"[{stock} ESG Data](https://finance.yahoo.com/quote/{stock}/sustainability)", unsafe_allow_html=True)
