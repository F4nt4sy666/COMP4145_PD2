import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

# Download 5 years of MSFT data
def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Calculate KDJ and MACD indicators
def calculate_indicators(data):
    # KDJ Calculation
    low_min = data['Low'].rolling(window=9).min()
    high_max = data['High'].rolling(window=9).max()
    rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['K'] = rsv.ewm(com=2).mean()
    data['D'] = data['K'].ewm(com=2).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']

    # MACD Calculation
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Identify buy/sell signals based on KDJ and MACD
def identify_signals(data):
    data['Signal'] = 0  # Initialize signal column with 0
    # Buy signal: K crosses above D and MACD crosses above Signal
    data['BuySignal'] = (data['K'] > data['D']) & (data['MACD'] > data['Signal'])
    return data

# Implement trading strategy
def implement_strategy(data, symbol="AAPL"):
    positions = []

    # Need at least 26 days to calculate MACD
    data = data.iloc[26:].copy()

    buy_dates = data[data['BuySignal'] == True].index.tolist()

    for buy_date in buy_dates:
        # Get buy price
        buy_price = data.loc[buy_date, 'Close']

        # Calculate target sell price (15% profit)
        target_price = buy_price * 1.15

        # Set maximum holding period
        max_sell_date = buy_date + pd.Timedelta(days=60)

        # Get data slice for potential sell period
        sell_period = data.loc[buy_date:max_sell_date].copy()

        # Check if target price is reached during the period
        target_reached = sell_period[sell_period['Close'] >= target_price]

        if not target_reached.empty:
            # Sell at first date target is reached
            sell_date = target_reached.index[0]
            sell_price = target_reached.loc[sell_date, 'Close']
            sell_reason = "Target reached"
        else:
            # Sell at end of maximum holding period
            sell_date_candidates = sell_period.index.tolist()
            if sell_date_candidates:
                sell_date = sell_date_candidates[-1]
                sell_price = data.loc[sell_date, 'Close']
                sell_reason = "Max holding period"
            else:
                # Skip if no valid sell date (should not happen in practice)
                continue

        # Calculate holding period in calendar days
        holding_days = (sell_date - buy_date).days

        # Calculate profit
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)

# Analyze the results
def analyze_results(positions):
    if positions.empty:
        return "No trading signals detected"

    # Summary statistics
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0

    avg_holding = positions['HoldingDays'].mean()

    target_reached = len(positions[positions['SellReason'] == 'Target reached'])
    max_period = len(positions[positions['SellReason'] == 'Max holding period'])

    print("\n===== Trading Strategy Results (KDJ + MACD)=====")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {win_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {loss_trades}")
    print(f"Average Profit: {avg_profit:.2f}%")

    return positions

def get_benchmark_return(ticker="^GSPC", start_date=None, end_date=None):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    cum_return = df['Close'] / df['Close'].iloc[0]
    cum_return.index = pd.to_datetime(df.index)
    return cum_return

def backtest(ticker, start_date, end_date, compare_with_benchmark=False, benchmark_ticker="^GSPC"):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        return None
    df = calculate_indicators(df)
    df = identify_signals(df)
    df = df.reset_index()
    df['BuySignal'] = df['BuySignal']
    df['SellSignal'] = False
    # 计算策略累计收益
    df['DailyReturn'] = df['Close'].pct_change().fillna(0)
    df['StrategyReturn'] = df['DailyReturn'] * df['BuySignal'].shift(1).fillna(0)
    strategy_cum_return = (1 + df['StrategyReturn']).cumprod()
    result = {"strategy_cumulative_return": strategy_cum_return, "trades": implement_strategy(df.set_index('Date'), symbol="AAPL")}
    if compare_with_benchmark:
        benchmark = get_benchmark_return(benchmark_ticker, start_date, end_date)
        result["benchmark_cumulative_return"] = benchmark
    return result

# Main function
def main():

    # Get stock data
    data = get_stock_data("MSFT")

    # Calculate indicators
    data = calculate_indicators(data)

    # Identify signals
    data = identify_signals(data)

    # 标记买入/卖出信号（用于前端画点）
    data = data.reset_index()
    data['BuySignal'] = data['BuySignal']
    # 卖出点可根据策略进一步标记，这里暂留空
    data['SellSignal'] = False

    # 导出每日价格和指标数据
    data_out = data[['Date', 'Close', 'K', 'D', 'J', 'MACD', 'Signal', 'BuySignal', 'SellSignal']].copy()
    data_out.to_csv('price_data.csv', index=False)

    # Implement strategy
    positions = implement_strategy(data.set_index('Date'), symbol="AAPL")

    # Analyze results
    analyze_results(positions)

    # 确保 positions 包含 Symbol 列
    if "Symbol" not in positions.columns:
        positions["Symbol"] = "AAPL"

    # 导出为 CSV 和 JSON，便于前端读取
    positions.to_csv('result.csv', index=False)
    positions.to_json('result.json', orient='records', date_format='iso')

    # Return the detailed positions dataframe
    return positions

positions = main()
print("\nDetailed Trades:")
print(positions.to_string())