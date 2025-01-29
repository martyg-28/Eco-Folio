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
        <h1 style="text-align: center; color: #2E8B57; font-size: 50px; font-weight: bold; margin-bottom: 0;">Eco-Folio</h1>
        <h2 style="text-align: center; color: #00008B; font-size: 30px; margin-top: 0;">Financial Analysis Tool</h2>
        <hr style="height: 4px; background-color: #2E8B57; margin: 25px 0;">
    """, unsafe_allow_html=True)

# Function to fetch stock data
def get_stock_data(stock_symbols, start_date, end_date):
    """Fetch stock data based on given symbols and date range."""
    stock_symbols = [s.strip().upper() for s in stock_symbols if s]  # Ensure symbols are clean

    if not stock_symbols:
        st.error("Please enter at least one valid stock symbol.")
        return pd.DataFrame()

    try:
        # Fetch data
        all_data = yf.download(stock_symbols, start=start_date, end=end_date, group_by='ticker')

        # Debugging: Print out the raw data structure
        st.write("Raw Data Returned by yfinance:", all_data)

        # If no data is returned at all
        if all_data.empty:
            st.error("No data returned. Possible reasons: invalid stock symbols or incorrect date range.")
            return pd.DataFrame()

        # Handle MultiIndex (if data is grouped by ticker)
        if isinstance(all_data.columns, pd.MultiIndex):
            st.warning("Data is grouped by ticker. Extracting 'Adj Close' prices for each symbol.")
            price_data = {ticker: all_data[ticker]['Adj Close'] for ticker in stock_symbols if 'Adj Close' in all_data[ticker]}
            price_df = pd.DataFrame(price_data)
        else:
            # If not grouped, try using 'Adj Close' or fallback to 'Close'
            if 'Adj Close' in all_data:
                price_df = all_data['Adj Close']
            elif 'Close' in all_data:
                st.warning("'Adj Close' not available. Using 'Close' instead.")
                price_df = all_data['Close']
            else:
                st.error("No valid price data columns ('Adj Close' or 'Close') found.")
                return pd.DataFrame()

        # Debug: Show the final processed data
        st.write("Processed Price Data:", price_df)

        # Final check
        if price_df.empty:
            st.error("No price data available. Please check your stock symbols or date range.")
            return pd.DataFrame()

        return price_df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()



# Function to analyze stocks
def analyze_stocks(price, stock_symbols, start_date, end_date):
    """Calculate risk, expected return, and CAGR for given stocks."""
    if price.empty:
        st.error("The price data is empty. Cannot perform analysis.")
        return {}

    valid_symbols = price.columns.tolist()
    stock_symbols = [s for s in stock_symbols if s in valid_symbols]

    if not stock_symbols:
        st.error("No valid stock symbols available for analysis.")
        return {}

    log_returns = np.log(price / price.shift(1)).dropna()
    stock_stats = {}
    for stock in stock_symbols:
        stock_stats[stock] = {
            "risk": round(log_returns[stock].std(), 5),
            "expected_return": round(log_returns[stock].mean(), 5),
            "cagr": round(((price[stock].iloc[-1] / price[stock].iloc[0]) ** (1 / ((end_date - start_date).days / 365)) - 1), 5)
        }
    if len(stock_symbols) > 1:
        stock_stats["correlation"] = price[stock_symbols].corr()
    return stock_stats

# Function for portfolio analysis
def portfolio_analysis(price):
    """Perform portfolio analysis on given stock prices."""
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
stock_symbol_1 = st.sidebar.text_input('Enter stock symbol 1', '').strip().upper()
stock_symbol_2 = st.sidebar.text_input('Enter stock symbol 2', '').strip().upper()
stock_symbol_3 = st.sidebar.text_input('Enter stock symbol 3', '').strip().upper()
stock_symbol_4 = st.sidebar.text_input('Enter stock symbol 4', '').strip().upper()
stock_symbol_5 = st.sidebar.text_input('Enter stock symbol 5', '').strip().upper()

stock_symbols = [stock_symbol_1, stock_symbol_2, stock_symbol_3, stock_symbol_4, stock_symbol_5]

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
            st.write(f"{stock} annualized risk: {stats['risk'] * 100:.2f}%")
            st.write(f"{stock} expected return: {stats['expected_return'] * 100:.2f}%")
            st.write(f"{stock} CAGR: {stats['cagr'] * 100:.2f}%")

    if len(stock_symbols) > 1 and "correlation" in stock_stats:
        st.write("### Correlation Matrix")
        st.dataframe(stock_stats["correlation"].style.background_gradient(cmap='coolwarm').format("{:.2f}"))

