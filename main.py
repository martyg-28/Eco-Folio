# -*- coding: utf-8 -*-

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Eco-Folio: Financial Analysis Tool")

# --------------------------
#        PAGE HEADER
# --------------------------
st.markdown(
    """
    <h1 style="text-align: center; color: #2E8B57; font-size: 50px; font-weight: bold; margin-bottom: 0;">Eco-Folio</h1>
    <h2 style="text-align: center; color: #00008B; margin-top: 0;">Financial Analysis Tool</h2>
    <hr style="height: 4px; background-color: #2E8B57; margin: 25px 0;">
    """,
    unsafe_allow_html=True
)

# --------------------------
#     DATA FETCHING
# --------------------------
def get_stock_data(stock_symbols, start_date, end_date):
    """
    Fetch stock data (Adj Close) from Yahoo Finance for the specified symbols and date range.
    """
    # Clean up symbols and remove empties
    stock_symbols = [s.strip().upper() for s in stock_symbols if s]
    if not stock_symbols:
        st.error("Please enter at least one valid stock symbol.")
        return pd.DataFrame()

    try:
        # Download data
        all_data = yf.download(stock_symbols, start=start_date, end=end_date, group_by='ticker')

        # Handle multi-index columns if present
        if isinstance(all_data.columns, pd.MultiIndex):
            price_df = pd.DataFrame({
                stock: all_data[stock]['Adj Close'] 
                for stock in stock_symbols if 'Adj Close' in all_data[stock]
            })
        else:
            # Single-index scenario
            price_df = all_data.get('Adj Close', all_data.get('Close', pd.DataFrame()))

        # Check for empty DataFrame
        if price_df.empty:
            st.error("No price data found. Please check symbols or date range.")
            return pd.DataFrame()

        return price_df
    
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

# --------------------------
#     STOCK ANALYSIS
# --------------------------
def analyze_stocks(price, start_date, end_date):
    """
    Calculate annualized risk, expected return (log-based), and CAGR for each stock in 'price'.
    Also computes correlation matrix for all stocks.
    """
    if price.empty:
        return {}

    # Compute daily log returns
    log_returns = np.log(price / price.shift(1)).dropna()

    # For each stock, compute metrics
    stats = {}
    for stock in price.columns:
        # Annualized standard deviation (risk)
        risk = log_returns[stock].std()  
        # Average daily return * 252 (approx trading days) for expected annual return
        expected_return = log_returns[stock].mean()  
        # CAGR
        days_held = (end_date - start_date).days
        years_held = days_held / 365.0
        cagr = (price[stock].iloc[-1] / price[stock].iloc[0]) ** (1 / years_held) - 1

        stats[stock] = {
            "risk": round(risk, 5),
            "expected_return": round(expected_return, 5),
            "cagr": round(cagr, 5)
        }

    # Correlation matrix across all stocks
    stats["correlation"] = price.corr()

    return stats

# --------------------------
#  PORTFOLIO OPTIMIZATION
# --------------------------
def portfolio_analysis(price):
    """
    Use a Monte Carlo approach to find the portfolio with the highest Sharpe Ratio.
    Return the max Sharpe weights, expected return, volatility, and Sharpe ratio.
    Also compute arrays for plotting if needed (ret_arr, vol_arr, sharpe_arr).
    """
    # Compute daily log returns
    log_returns = np.log(price / price.shift(1)).dropna()
    num_stocks = len(price.columns)

    # Monte Carlo simulation
    num_portfolios = 20000
    all_weights = np.random.rand(num_portfolios, num_stocks)
    all_weights /= all_weights.sum(axis=1)[:, np.newaxis]

    # Annualized returns & volatility
    ret_arr = np.dot(all_weights, log_returns.mean() * 252)
    vol_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, (log_returns.cov() * 252), all_weights))
    
    # Sharpe Ratio (no risk-free rate assumption for simplicity)
    sharpe_arr = np.where(vol_arr == 0, 0, ret_arr / vol_arr)

    # Identify max Sharpe Ratio
    max_sharpe_idx = sharpe_arr.argmax()

    # Return the best weights & stats
    best_weights = all_weights[max_sharpe_idx]
    best_return = ret_arr[max_sharpe_idx]
    best_vol = vol_arr[max_sharpe_idx]
    best_sr = sharpe_arr[max_sharpe_idx]

    return best_weights, best_return, best_vol, best_sr, ret_arr, vol_arr, sharpe_arr

# --------------------------
#      SIDEBAR INPUTS
# --------------------------
st.sidebar.header("Enter Stock Symbols & Date Range")
stock_symbols = [st.sidebar.text_input(f"Stock symbol {i+1}", "").upper() for i in range(5)]

date_range = st.sidebar.date_input(
    "Select date range",
    [pd.to_datetime("2019-01-01"), pd.to_datetime("2023-10-01")]
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid date range.")
    start_date, end_date = None, None

analyze_button = st.sidebar.button("Analyze Stocks")

# --------------------------
#      MAIN LOGIC
# --------------------------
if analyze_button:
    # 1) Fetch Price Data
    price_data = get_stock_data(stock_symbols, start_date, end_date)
    if price_data.empty:
        st.stop()  # Stop execution if no data

    # 2) Analyze Stocks (Risk, Return, CAGR, Correlation)
    stock_stats = analyze_stocks(price_data, start_date, end_date)

    # 3) Display Stock Metrics
    st.markdown("## Stock Metrics")
    for symbol in price_data.columns:
        if symbol in stock_stats:
            st.markdown(f"### {symbol}")
            st.write(f"**Annualized Risk (std):** {stock_stats[symbol]['risk']*100:.2f}%")
            st.write(f"**Expected Return:** {stock_stats[symbol]['expected_return']*100:.2f}%")
            st.write(f"**CAGR:** {stock_stats[symbol]['cagr']*100:.2f}%")

    # 4) Correlation Matrix + Explainer
    if "correlation" in stock_stats:
        st.markdown("## Correlation Matrix")
        corr_matrix = stock_stats["correlation"]
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}")
        )
        st.markdown("""
            **How to read the Correlation Matrix**  
            - **1.0**: Perfect positive correlation (move in the same direction).  
            - **-1.0**: Perfect negative correlation (one goes up, the other goes down).  
            - **0**: No correlation (movements are unrelated).  
            Mixing assets with lower or negative correlations can help reduce overall portfolio risk.
        """)

    # 5) Portfolio Optimization
    if len(price_data.columns) > 1:
        (
            optimized_weights,
            optimized_ret,
            optimized_vol,
            optimized_sr,
            ret_arr,
            vol_arr,
            sharpe_arr
        ) = portfolio_analysis(price_data)

        # 5a) Show Pie Chart for Optimized Weights
        st.markdown("## Optimized Portfolio Weights")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.Set2.colors  # Distinguishable color set

        # In case # of stocks > # of default colors, handle color cycle
        wedges, texts, autotexts = ax.pie(
            optimized_weights,
            labels=price_data.columns,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "black"}
        )
        for text in texts + autotexts:
            text.set_fontsize(9)

        plt.title("Optimized Weights (Max Sharpe)", fontsize=14, fontweight='bold')
        plt.legend(
            wedges,
            [f"{stock} - {weight*100:.1f}%" for stock, weight in zip(price_data.columns, optimized_weights)],
            title="Allocation",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        st.pyplot(fig)

        st.markdown("""
        **Explanation**: These weights represent the proportion of total capital allocated to each stock 
        to maximize the portfolio's Sharpe Ratio. A higher Sharpe Ratio indicates better risk-adjusted returns.
        """)

        # 5b) Display Optimized Portfolio Metrics
        st.markdown("### Optimized Portfolio Return")
        st.write(f"**{optimized_ret*100:.2f}%** expected annual return.")

        st.markdown("### Optimized Portfolio Volatility")
        st.write(f"**{optimized_vol*100:.2f}%** annualized standard deviation (risk).")

        st.markdown("### Optimized Portfolio Sharpe Ratio")
        st.write(f"**{optimized_sr:.4f}** (risk-adjusted return measure).")

        # 5c) Plot the Efficient Frontier
        st.markdown("## Efficient Frontier")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            vol_arr, 
            ret_arr, 
            c=sharpe_arr, 
            cmap="viridis", 
            alpha=0.6, 
            edgecolors="white",
            label="Random Portfolios"
        )
        # Mark the optimized portfolio
        ax.scatter(
            optimized_vol,
            optimized_ret,
            c="red",
            s=100,
            edgecolors="black",
            label="Max Sharpe Portfolio"
        )
        plt.colorbar(scatter, label="Sharpe Ratio")
        ax.set_xlabel("Volatility (Risk)")
        ax.set_ylabel("Expected Return")
        ax.set_title("Efficient Frontier")
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        **Understanding the Graph**  
        - Each dot represents a unique portfolio (random combination of your selected stocks).  
        - The color scale indicates the Sharpe Ratio. Brighter/yellow dots have higher Sharpe Ratios.  
        - The red dot is the 'Optimized Portfolio' with the highest Sharpe Ratio.  
        Ideally, you want a portfolio in the top-left area: **high return, low risk**.
        """)

    else:
        st.warning("You need at least two stocks to perform portfolio optimization and see the Efficient Frontier.")

    # 6) ESG Explainer & Considerations
    st.markdown("""
    <div style="background-color: #F0F8FF; padding: 20px; border-radius: 10px; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 30px;">
        <p style="text-align: center; font-size: 18px; color: #333; font-weight: bold;">
            Investing in companies with strong 
            <span style="color: #2E8B57;">Environmental, Social, and Governance (ESG)</span> 
            practices aligns your portfolio with ethical and sustainable values. 
            A robust ESG score can indicate long-term resilience, societal benefits, 
            and transparent governance. Check out the ESG profiles below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ESG Considerations")
    for stock in price_data.columns:
        esg_link = f"https://finance.yahoo.com/quote/{stock}/sustainability"
        st.markdown(f"- [{stock} ESG Data]({esg_link})", unsafe_allow_html=True)
