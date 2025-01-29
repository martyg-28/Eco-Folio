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
        <div style="background-color: #F0F8FF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <p style="text-align: center; font-size: 24px; color: #333; font-weight: bold;">
                Welcome to <span style="color: #2E8B57;">Eco-Folio</span> - your financial analysis companion!
            </p>
            <p style="text-align: center; font-size: 22px; color: #333; font-weight: bold;">
                Here, you can build and analyze a portfolio that optimizes your investment choices.
            </p>
            <p style="text-align: center; font-size: 22px; color: #333; font-weight: bold;">
                Enter your desired stock symbols, select a date range, and let the tool work its magic!
            </p>
        </div>
        <hr style="height: 2px; background-color: #2E8B57; margin: 25px 0;">
    """, unsafe_allow_html=True)

st.markdown("""
    <hr style="height: 2px; background-color: #2E8B57; margin: 25px 0;">
    <h2 style="text-align: center; color: #00008B; font-size: 30px;">Insights Below:</h2>
    <hr style="height: 2px; background-color: #2E8B57; margin: 25px 0;">
""", unsafe_allow_html=True)



def get_stock_data(stock_symbols, start_date, end_date):
    """Fetch stock data based on given symbols and date range."""
    stock_symbols = [s for s in stock_symbols if s]  # Remove empty strings
    if not stock_symbols:
        st.error("Please enter at least one valid stock symbol.")
        return pd.DataFrame()

    all_data = yf.download(stock_symbols, start=start_date, end=end_date)
    if all_data.empty:
        st.error("No data was returned. Please check your stock symbols or date range.")
        return pd.DataFrame()

    # Flatten columns if it's a multi-index
    if isinstance(all_data.columns, pd.MultiIndex):
        all_data.columns = [' '.join(col).strip() for col in all_data.columns]

    # Use 'Adj Close' if available, otherwise fallback to 'Close'
    if 'Adj Close' in all_data:
        return all_data['Adj Close']
    elif 'Close' in all_data:
        st.warning("'Adj Close' not available. Using 'Close' instead.")
        return all_data['Close']
    else:
        st.error("Neither 'Adj Close' nor 'Close' columns are available in the data.")
        return pd.DataFrame()


def analyze_stocks(price, stock_symbols, start_date, end_date):
    """Calculate risk, expected return, and CAGR for given stocks."""
    log_returns = np.log(price / price.shift(1))
    stock_stats = {}
    for stock in stock_symbols:
        stock_stats[stock] = {
            "risk": round(log_returns[stock].std(), 5),
            "expected_return": round(log_returns[stock].mean(), 5),
            "cagr": round(((price[stock][-1] / price[stock][0]) ** (1 / ((end_date - start_date).days / 365)) - 1), 5)
        }
    if len(stock_symbols) > 1:
        stock_stats["correlation"] = round(price.corr(), 3)
    return stock_stats


def portfolio_analysis(price):
    """Perform portfolio analysis on given stock prices."""
    log_returns = np.log(price / price.shift(1)).dropna()  # Droping NaN values

    num_stocks = len(price.columns)
    all_weights = np.random.rand(6000, num_stocks)
    all_weights /= all_weights.sum(axis=1)[:, np.newaxis]
    
    ret_arr = np.dot(all_weights, log_returns.mean() * 252)
    vol_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, log_returns.cov() * 252, all_weights))
    sharpe_arr = np.where(vol_arr != 0, ret_arr / vol_arr, 0) 

    max_sharpe_idx = sharpe_arr.argmax()
    return all_weights[max_sharpe_idx], ret_arr[max_sharpe_idx], vol_arr[max_sharpe_idx], sharpe_arr[max_sharpe_idx], ret_arr, vol_arr, sharpe_arr



# Lets users input  stock symbols
stock_symbol_1 = st.sidebar.text_input('Enter stock symbol 1', '').strip().upper()
stock_symbol_2 = st.sidebar.text_input('Enter stock symbol 2', '').strip().upper()
stock_symbol_3 = st.sidebar.text_input('Enter stock symbol 3', '').strip().upper()
stock_symbol_4 = st.sidebar.text_input('Enter stock symbol 4', '').strip().upper()
stock_symbol_5 = st.sidebar.text_input('Enter stock symbol 5', '').strip().upper()

stock_symbols = [stock_symbol_1, stock_symbol_2, stock_symbol_3, stock_symbol_4, stock_symbol_5]

date_range = st.sidebar.date_input(
    'Select date range',
    [pd.to_datetime('2019-01-01'), pd.to_datetime('2023-10-01')]
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid date range.")  # Show an error message
    start_date, end_date = None, None 


calculate = st.sidebar.button('Analyze Stocks')

if calculate:
    price = get_stock_data(stock_symbols, start_date, end_date)
    stock_stats = analyze_stocks(price, stock_symbols, start_date, end_date)

    for stock in stock_symbols:
        stats = stock_stats.get(stock, {})
        if stats:
            st.markdown(f"### {stock} Metrics")
            st.write(f"{stock} annualized stock risk: {stats['risk'] * 100:.2f}%")
            st.write(f"{stock} annualized stock expected returns: {stats['expected_return'] * 100:.2f}%")
            st.write(f"{stock} stock CAGR: {stats['cagr'] * 100:.2f}%", help="The Compound Annual Growth Rate (CAGR) represents the mean annual growth rate of an investment over a specified time period longer than one year.")

    if len(stock_symbols) > 1:
        st.write("Correlation matrix:")

        styled_corr = stock_stats["correlation"].style.background_gradient(cmap='coolwarm').format("{:.2f}")
        st.dataframe(styled_corr)

        st.write("""
        ### Understanding the Correlation Matrix
        The correlation matrix measures how changes in one stock's prices are associated with changes in another stock's prices. Values range between -1 and 1:
        
        - **1**: Perfect positive correlation. Both stocks tend to move in the same direction.
        - **-1**: Perfect negative correlation. When one stock goes up, the other tends to go down.
        - **0**: No correlation. The movements of the stocks are not related.
        
        Correlations can help in diversifying a portfolio. For instance, mixing stocks with low or negative correlations can reduce the portfolio's overall risk.
        """)

 


        optimized_weights, optimized_ret, optimized_vol, optimized_sr, ret_arr, vol_arr, sharpe_arr = portfolio_analysis(price)
        optimized_weights_str = ", ".join([f"{stock}: {weight:.2f}" for stock, weight in zip(stock_symbols, optimized_weights)])
        st.write("### Optimized Weights")
        st.write(optimized_weights_str)

        st.write("""
These represent the proportion of the total investment that should be allocated to each stock in order to optimize the portfolio based on the Sharpe Ratio. The values sum up to 1 or 100%.
""")

        fig, ax = plt.subplots(figsize=(2.5, 2.5))  
        fig.subplots_adjust(left=0.1, right=0.75)  
        colors = plt.cm.Dark2.colors 
        wedges, _ = ax.pie(optimized_weights, startangle=90, colors=colors, shadow=False)
        ax.axis('equal') 

        title_text = 'Portfolio Weights'
        plt.title(title_text, fontsize=12, color='black', fontweight='bold', pad=10)

        percentages = [f"{weight*100:.1f}%" for weight in optimized_weights]
        legend_labels = [f"{label} - {percent}" for label, percent in zip(stock_symbols, percentages)]
        plt.legend(wedges, legend_labels, title="Allocation", loc="center left", fontsize=8, title_fontsize=10, bbox_to_anchor=(1, 0, 0.5, 1))

        fig.patch.set_linewidth(1)  
        fig.patch.set_edgecolor('grey')  

        st.pyplot(fig)


        st.write("### Optimized Portfolio Return")
        st.write(f"{optimized_ret * 100:.2f}%")  # Display value with 2 decimal places
        st.write("""
        This is the expected annual return of the optimized portfolio, given the weights. 
        """)
        st.write("### Optimized Portfolio Volatility")
        st.write(f"{optimized_vol * 100:.2f}%")  # Display value as percentage with 2 decimal places
        st.write("""
        This represents the standard deviation of the optimized portfolio's return, which is a measure of its risk. A higher volatility indicates a wider range of potential outcomes for the portfolio's return, implying more risk.
        """)
        st.write("### Optimized Portfolio Sharpe Ratio")
        st.write(f"{optimized_sr:.4f}")
        st.write("""
        The Sharpe Ratio is a measure of risk-adjusted return. It's calculated as the portfolio's excess return (over the risk-free rate) divided by its volatility.In simpler terms, a higher Sharpe Ratio indicates that the portfolio is delivering better returns for the level of risk taken. Generally, a ratio above 1 is favorable, with higher values being increasingly desirable. The 'optimized' portfolio showcased here has the highest Sharpe Ratio.
        """)
        ...

        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)
        plt.colorbar(sc, label='Sharpe Ratio', ax=ax)
        ax.set_title('Efficient Frontier', fontsize=18, )
        ax.set_xlabel('Volatility (Standard Deviation)', fontsize=14)
        ax.set_ylabel('Expected Return', fontsize=14)

        if np.isfinite(vol_arr.min()) and np.isfinite(vol_arr.max()):
            ax.set_xlim([vol_arr.min() - 0.05, vol_arr.max() + 0.05])
        if np.isfinite(ret_arr.min()) and np.isfinite(ret_arr.max()):
            ax.set_ylim([ret_arr.min() - 0.05, ret_arr.max() + 0.05])

        ax.scatter(optimized_vol, optimized_ret, c='red', s=100, edgecolors="k", label='Optimized Portfolio')

        ax.legend(fontsize=12)

        fig.patch.set_linewidth(2)  
        fig.patch.set_edgecolor('grey') 

        st.pyplot(fig)


    st.write("""
        ## Understanding the Graph
        The scatter plot above, known as the 'Efficient Frontier', represents various portfolio combinations and their expected returns against their volatilities.
        
        - **Points**: Each point signifies a portfolio with a specific combination of the stocks you've entered.
        - **Color**: The color of the points indicates the Sharpe Ratio. Yellow points have a higher Sharpe ratio, implying better risk-adjusted returns.
        - **Red Dot**: Represents the 'Optimized Portfolio', the best combination of stocks that offers the maximum expected return for a given level of risk.
        
        In general, you'd want a portfolio that's towards the top left of the graph, where you get the highest returns for the lowest risk.
    """)
    

            

with st.container():
    st.markdown("""
        <hr style="height: 2px; background-color: #2E8B57; margin: 25px 0;">
        <h2 style="text-align: center; color: #00008B; font-size: 30px;">ESG Considerations</h2>
        <div style="background-color: #F0F8FF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <p style="text-align: center; font-size: 22px; color: #333; font-weight: bold;">
                Selecting stocks with strong <span style="color: #2E8B57;">Environmental, Social, and Governance (ESG)</span> credentials can align your investments with your ethical standards. A robust ESG score signifies a company's commitment to environmental sustainability, positive societal impact, and sound, transparent governance. Such companies often exhibit long-term resilience. Explore the ESG profiles of companies in your portfolio using the links below.
            </p>
        </div>
        <hr style="height: 2px; background-color: #2E8B57; margin: 25px 0;">
    """, unsafe_allow_html=True)

    # Generating ESG data links for each stock
    for stock in stock_symbols:
        if stock:
            esg_link = f"https://finance.yahoo.com/quote/{stock}/sustainability"
            st.markdown(f"<h5 style='text-align: center;'><a href='{esg_link}' target='_blank'>{stock} ESG Data</a></h5>", unsafe_allow_html=True)



