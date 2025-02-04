# -*- coding: utf-8 -*-

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Eco-Folio: Financial Analysis Tool")

# Title & Introduction
with st.container():
    st.markdown("""
        <h1 style="text-align: center; color: #2E8B57; font-size: 50px; font-weight: bold;">Eco-Folio</h1>
        <h2 style="text-align: center; color: #00008B;">Financial Analysis Tool</h2>
        <hr style="height: 4px; background-color: #2E8B57;">
    """, unsafe_allow_html=True)

# Function to fetch stock data
def get_stock_data(stock_symbols, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    stock_symbols = [s.strip().upper() for s in stock_symbols if s]
    if not stock_symbols:
        st.error("Please enter at least one valid stock symbol.")
        return pd.DataFrame()

    try:
        all_data = yf.download(stock_symbols, start=start_date, end=end_date, group_by='ticker')

        if isinstance(all_data.columns, pd.MultiIndex):
            st.warning("Extracting 'Adj Close' prices for each stock.")
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
def analyze_stocks(price, stock_symbols, start_date, end_date):
    """Calculate risk, expected return, and CAGR for stocks."""
    if price.empty:
        st.error("No price data. Cannot perform analysis.")
        return {}

    log_returns = np.log(price / price.shift(1)).dropna()
    stock_stats = {
        stock: {
            "risk": round(log_returns[stock].std(), 5),
            "expected_return": round(log_returns[stock].mean(), 5),
            "cagr": round(((price[stock].iloc[-1] / price[stock].iloc[0]) ** (1 / ((end_date - start_date).days / 365)) - 1), 5)
        } for stock in stock_symbols if stock in price.columns
    }

    if len(stock_symbols) > 1:
        stock_stats["correlation"] = price.corr()

    return stock_stats

# Function for portfolio analysis
def portfolio_analysis(price):
    """Perform portfolio optimization to find the max Sharpe ratio."""
    log_returns = np.log(price / price.shift(1)).dropna()
    num_stocks = len(price.columns)

    all_weights = np.random.rand(6000, num_stocks)
    all_weights /= all_weights.sum(axis=1)[:, np.newaxis]

    ret_arr = np.dot(all_weights, log_returns.mean() * 252)
    vol_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, log_returns.cov() * 252, all_weights))
    sharpe_arr = np.where(vol_arr != 0, ret_arr / vol_arr, 0)

    max_sharpe_idx = sharpe_arr.argmax()
    return all_weights[max_sharpe_idx], ret_arr[max_sharpe_idx], vol_arr[max_sharpe_idx], sharpe_arr[max_sharpe_idx], ret_arr, vol_arr, sharpe_arr

# User inputs for stock symbols
stock_symbols = [st.sidebar.text_input(f'Enter stock symbol {i+1}', '').strip().upper() for i in range(5)]

# User inputs for date range
date_range = st.sidebar.date_input(
    'Select date range',
    [pd.to_datetime('2019-01-01'), pd.to_datetime('2023-10-01')]
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid date range.")
    start_date, end_date = None, None

# Analyze button
calculate = st.sidebar.button('Analyze Stocks')

if calculate:
    price = get_stock_data(stock_symbols, start_date, end_date)
    if price.empty:
        st.stop()

    stock_stats = analyze_stocks(price, stock_symbols, start_date, end_date)

    for stock in stock_symbols:
        stats = stock_stats.get(stock, {})
        if stats:
            st.markdown(f"### {stock} Metrics")
            st.write(f"Annualized Risk: {stats['risk'] * 100:.2f}%")
            st.write(f"Expected Return: {stats['expected_return'] * 100:.2f}%")
            st.write(f"CAGR: {stats['cagr'] * 100:.2f}%")

    if len(stock_symbols) > 1 and "correlation" in stock_stats:
        st.write("### Correlation Matrix")
        st.dataframe(stock_stats["correlation"].style.background_gradient(cmap='coolwarm').format("{:.2f}"))

        # Portfolio Optimization
        optimized_weights, optimized_ret, optimized_vol, optimized_sr, ret_arr, vol_arr, sharpe_arr = portfolio_analysis(price)

        st.write("### Optimized Portfolio Metrics")
        st.write(f"Return: {optimized_ret * 100:.2f}%")
        st.write(f"Volatility: {optimized_vol * 100:.2f}%")
        st.write(f"Sharpe Ratio: {optimized_sr:.4f}")

        # Pie chart for weights
        st.write("### Optimized Portfolio Weights")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(optimized_weights, labels=stock_symbols, autopct='%1.1f%%', startangle=90)
        plt.title("Optimized Portfolio Weights")
        st.pyplot(fig)

        # Efficient Frontier
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)
        plt.colorbar(sc, label='Sharpe Ratio')
        ax.scatter(optimized_vol, optimized_ret, c='red', s=100, edgecolors="k", label='Optimized Portfolio')
        ax.legend(fontsize=12)
        plt.xlabel("Volatility (Standard Deviation)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        st.pyplot(fig)

    with st.container():
        st.markdown("### ESG Considerations")
        for stock in stock_symbols:
            if stock:
                esg_link = f"https://finance.yahoo.com/quote/{stock}/sustainability"
                st.markdown(f"[{stock} ESG Data]({esg_link})", unsafe_allow_html=True)
