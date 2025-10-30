import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

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

def identify_signals(data):
    data['BuySignal'] = (data['K'] > data['D']) & (data['MACD'] > data['Signal'])
    return data

def simulate_trading(data):
    data = data.copy()
    data = calculate_indicators(data)
    data = identify_signals(data)
    data['Position'] = 0
    in_position = False
    for i in range(len(data)):
        # 兼容 Series 或单一值
        buy_signal_val = data.iloc[i]['BuySignal']
        if isinstance(buy_signal_val, (pd.Series, np.ndarray)):
            buy_signal = bool(buy_signal_val.item())
        else:
            buy_signal = bool(buy_signal_val)
        macd_cross_down_val = data.iloc[i]['MACD'] < data.iloc[i]['Signal']
        if isinstance(macd_cross_down_val, (pd.Series, np.ndarray)):
            macd_cross_down = bool(macd_cross_down_val.item())
        else:
            macd_cross_down = bool(macd_cross_down_val)
        if (not in_position) and buy_signal:
            data.at[data.index[i], 'Position'] = 1
            in_position = True
        elif in_position and macd_cross_down:
            data.at[data.index[i], 'Position'] = 0
            in_position = False
        else:
            data.at[data.index[i], 'Position'] = data.iloc[i-1]['Position'] if i > 0 else 0
    data['DailyReturn'] = data['Close'].pct_change().fillna(0)
    data['StrategyReturn'] = data['DailyReturn'] * data['Position'].shift(1).fillna(0)
    return data

def analyze_portfolio(tickers, start_date, end_date):
    price_dfs = []
    strat_dfs = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            continue
        df = simulate_trading(df)
        df['Ticker'] = ticker
        price_dfs.append(df[['Close']].rename(columns={'Close': ticker}))
        strat_dfs.append(df[['StrategyReturn']].rename(columns={'StrategyReturn': ticker}))
    if not price_dfs or not strat_dfs:
        return {
            "error": "No valid data downloaded for the given tickers and date range. Please check ticker symbols, date range, or your network connection."
        }
    # 合并所有股票的策略收益
    strat_returns = pd.concat(strat_dfs, axis=1)
    strat_returns = strat_returns.fillna(0)
    # 动态权重分配（基于 MACD 动量）
    weights = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                print(f"No data for {ticker}")
                continue
            df = calculate_indicators(df)
            # 检查200日均线
            df['MA200'] = df['Close'].rolling(window=200).mean()
            close_now = df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else float(df['Close'].iloc[-1])
            ma200_now = df['MA200'].iloc[-1].item() if hasattr(df['MA200'].iloc[-1], 'item') else float(df['MA200'].iloc[-1])
            is_above_ma200 = close_now > ma200_now
            if not is_above_ma200:
                print(f"{ticker} is below 200MA")
                continue
            macd_now = df['MACD'].iloc[-1].item() if hasattr(df['MACD'].iloc[-1], 'item') else float(df['MACD'].iloc[-1])
            weights[ticker] = max(0.1, macd_now)  # 确保最小权重为 0.1
            print(f"{ticker}: MACD = {macd_now}, Weight = {weights[ticker]}")
        except Exception as e:
            print(f"Error calculating momentum for {ticker}: {e}")
            weights[ticker] = 1.0 / len(tickers)

    # 归一化权重
    total_weight = sum(weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
    print(f"Final weights: {weights}")

    # 确保 weights 和 strat_returns 的索引对齐
    valid_tickers = [ticker for ticker in weights.keys() if ticker in strat_returns.columns]
    if not valid_tickers:
        return {"error": "No valid tickers found in the downloaded data."}
    weights = {ticker: weights[ticker] for ticker in valid_tickers}
    strat_returns = strat_returns[valid_tickers]

    # 确保 weights 的索引和 strat_returns 的列名完全一致（包括大小写）
    weights_series = pd.Series(weights)
    weights_series.index = strat_returns.columns

    # 计算加权组合收益
    portfolio_daily_return = strat_returns.multiply(weights_series, axis=1).sum(axis=1)
    portfolio_cum_return = (1 + portfolio_daily_return).cumprod() - 1
    total_return = portfolio_cum_return.iloc[-1]
    n_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    running_max = portfolio_cum_return.cummax()
    drawdown = (portfolio_cum_return - running_max) / running_max
    max_drawdown = drawdown.min()
    sharpe_ratio = portfolio_daily_return.mean() / (portfolio_daily_return.std() + 1e-9) * np.sqrt(252)
    print(f"Portfolio daily returns: {portfolio_daily_return.tail()}")
    print(f"Portfolio cumulative return: {portfolio_cum_return.tail()}")
    return {
        "portfolio_cumulative_return": portfolio_cum_return,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio)
    }

if __name__ == "__main__":
    # 示例调用
    result = analyze_portfolio(["AAPL", "MSFT"], "2020-01-01", "2024-12-31")
    print(result)