import streamlit as st
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid GUI backend errors on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Load price data for chart
@st.cache_data
def load_price_data():
    try:
        df = pd.read_csv('price_data.csv')
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
    except Exception:
        df = pd.DataFrame()
    return df

# Load trade data for statistics/details
@st.cache_data(ttl=60)
def load_trade_data():
    try:
        df = pd.read_csv('result.csv')
        st.write("Debug - Loaded data:", df.head())  # Debug output
        if 'BuyDate' in df.columns:
            df['BuyDate'] = pd.to_datetime(df['BuyDate'], utc=True)
        if 'SellDate' in df.columns:
            df['SellDate'] = pd.to_datetime(df['SellDate'], utc=True)
    except Exception as e:
        st.error(f"加载交易数据失败: {e}")
        df = pd.DataFrame()
    return df

def main():

    st.set_page_config(page_title="Trading Backtest Results", layout="wide")
    st.title("Trading Backtest Results")


    menu = ["Price Chart", "Trade Details", "Portfolio Analyzer", "Smart Portfolio Builder"]
    st.sidebar.markdown("---")
    choice = st.sidebar.radio("Select Page", menu)

    if choice == "Portfolio Analyzer":
        st.subheader("Portfolio Analyzer")
        tickers_input = st.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, BAC, WMT, PG")
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01")).strftime("%Y-%m-%d")
        end_date = st.date_input("End Date", pd.to_datetime("2024-12-31")).strftime("%Y-%m-%d")
        enable_ma200_filter = st.checkbox("Enable 200MA Filter (Only stocks above 200-day moving average)", value=True)
        if st.button("Analyze Portfolio"):
            with st.spinner("Analyzing portfolio, please wait..."):
                import sys
                sys.path.append('.')
                from portfolio_analyzer import analyze_portfolio
                tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
                result = analyze_portfolio(tickers, start_date, end_date)
                if result is None or (isinstance(result, dict) and 'error' in result):
                    st.error(result['error'] if result and 'error' in result else "No result returned. Please check the input parameters and try again.")
                else:
                    st.subheader("Portfolio Cumulative Return")
                    st.line_chart(result["portfolio_cumulative_return"])
                    st.subheader("Portfolio Statistics")
                    st.write(f"Total Return: {result['total_return']:.2%}")
                    st.write(f"Annualized Return: {result['annualized_return']:.2%}")
                    st.write(f"Max Drawdown: {result['max_drawdown']:.2%}")
                    st.write(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                    # 详细交易明细（每只股票）
                    st.info("Detailed trades for each stock are not shown here, but can be added if simulate_golden_cross_trading returns trades.")

    elif choice == "Price Chart":
        df = load_price_data()
        if df.empty:
            st.warning("No price data found. Please run the backend and export price_data.csv first!")
            return
        st.subheader("Price Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        # 画累计收益曲线
        if 'Close' in df.columns:
            cum_return = df['Close'] / df['Close'].iloc[0]
            ax.plot(df['Date'], cum_return, label='Stock Cumulative Return', color='blue')
      
        if 'BuySignal' in df.columns:
            buy_points = df[df['BuySignal'] == True]
            ax.scatter(buy_points['Date'], (buy_points['Close'] / df['Close'].iloc[0]), color='red', label='Buy Signal', marker='^', s=100)
        if 'SellSignal' in df.columns:
            sell_points = df[df['SellSignal'] == True]
            ax.scatter(sell_points['Date'], (sell_points['Close'] / df['Close'].iloc[0]), color='purple', label='Sell Signal', marker='v', s=100)
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Return')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif choice == "Trade Details":
        df = load_trade_data()
        st.subheader("Trade Details")
        if df.empty:
            st.error("未找到交易数据，请确保 result.csv 文件格式正确。")
        else:
            st.dataframe(df)

    elif choice == "Smart Portfolio Builder":
        st.subheader("Smart Portfolio Builder")
        tickers_input = st.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, BAC, WMT, PG")
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01")).strftime("%Y-%m-%d")
        end_date = st.date_input("End Date", pd.to_datetime("2024-12-31")).strftime("%Y-%m-%d")
        if st.button("Build Portfolio"):
            with st.spinner("Building smart portfolio, please wait..."):
                import sys
                sys.path.append('.')
                from smart_portfolio import build_smart_portfolio
                tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
                result = build_smart_portfolio(tickers, start_date, end_date)
                if result is None or (isinstance(result, dict) and 'error' in result):
                    st.error(result['error'] if result and 'error' in result else "No result returned. Please check the input parameters and try again.")
                else:
                    st.subheader("Selected Stocks")
                    st.dataframe(pd.DataFrame({'Ticker': result['selected_stocks']}))
                    st.subheader("Momentum Weights")
                    st.bar_chart(pd.Series(result['weights']))
                    st.subheader("Sector Concentration")
                    sector_df = pd.DataFrame(list(result['sector_concentration'].items()), columns=['Sector', 'Weight'])
                    st.pyplot(sector_df.set_index('Sector').plot.pie(y='Weight', autopct='%1.1f%%', legend=False, figsize=(6,6)).figure)
                    st.write(f"HHI: {result.get('hhi', 0.5):.3f}")
                    if result.get('is_concentrated', False):
                        st.warning("Risk Warning: Portfolio is highly concentrated (HHI > 0.6)")

if __name__ == "__main__":
    main()
