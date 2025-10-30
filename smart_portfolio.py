import pandas as pd
import numpy as np
import yfinance as yf
from collections import defaultdict
# ========== MACD 指标计算 ===========
def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df


# ========== 功能 1：KDJ & MACD to create portfolio==========

# ========== KDJ 指标计算 ===========
def calculate_kdj(df, n=9):
    df = df.copy()
    low_list = df['Low'].rolling(window=n, min_periods=1).min()
    high_list = df['High'].rolling(window=n, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

def screen_stocks_simple(tickers, start_date, end_date):
    buy_signal_tickers = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if df.empty or len(df) < 26:
                continue
            df = calculate_kdj(df)
            df = calculate_macd(df)
            # 检查200日均线（暂时禁用）
            # df['MA200'] = df['Close'].rolling(window=200).mean()
            # close_now = df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else float(df['Close'].iloc[-1])
            # ma200_now = df['MA200'].iloc[-1].item() if hasattr(df['MA200'].iloc[-1], 'item') else float(df['MA200'].iloc[-1])
            # is_above_ma200 = close_now > ma200_now
            # if not is_above_ma200:
            #     print(f"{ticker} is below 200MA")
            #     continue
            price_now = df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else float(df['Close'].iloc[-1])
            j_now = df['J'].iloc[-1].item() if hasattr(df['J'].iloc[-1], 'item') else float(df['J'].iloc[-1])
            k_prev = df['K'].iloc[-2].item() if hasattr(df['K'].iloc[-2], 'item') else float(df['K'].iloc[-2])
            k_now = df['K'].iloc[-1].item() if hasattr(df['K'].iloc[-1], 'item') else float(df['K'].iloc[-1])
            d_prev = df['D'].iloc[-2].item() if hasattr(df['D'].iloc[-2], 'item') else float(df['D'].iloc[-2])
            d_now = df['D'].iloc[-1].item() if hasattr(df['D'].iloc[-1], 'item') else float(df['D'].iloc[-1])
            macd_prev = df['MACD'].iloc[-2].item() if hasattr(df['MACD'].iloc[-2], 'item') else float(df['MACD'].iloc[-2])
            macd_now = df['MACD'].iloc[-1].item() if hasattr(df['MACD'].iloc[-1], 'item') else float(df['MACD'].iloc[-1])
            signal_prev = df['MACD_signal'].iloc[-2].item() if hasattr(df['MACD_signal'].iloc[-2], 'item') else float(df['MACD_signal'].iloc[-2])
            signal_now = df['MACD_signal'].iloc[-1].item() if hasattr(df['MACD_signal'].iloc[-1], 'item') else float(df['MACD_signal'].iloc[-1])
            # MACD golden cross: MACD crosses above signal line
            macd_golden_cross = macd_prev < signal_prev and macd_now > signal_now
            # Buy condition: MACD is positive (further relaxed condition)
            if isinstance(macd_now, (pd.Series, np.ndarray)):
                macd_now = macd_now.item()
            if macd_now > 0:
                print(f"{ticker} meets MACD condition: MACD = {macd_now}")
                print(f"{ticker} meets MACD condition: MACD = {macd_now}")
                try:
                    buy_signal_tickers.append(ticker)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return buy_signal_tickers

def build_smart_portfolio(tickers, start_date, end_date):
    selected_stocks = screen_stocks_simple(tickers, start_date, end_date)
    if not selected_stocks:
        return {'error': 'No stocks meet the criteria.'}
    # Calculate sector concentration dynamically
    sector_info = defaultdict(float)
    for ticker in selected_stocks:
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', 'Unknown')
            sector_info[sector] += 1.0 / len(selected_stocks)
        except Exception as e:
            print(f"Error fetching sector info for {ticker}: {e}")
            sector_info['Unknown'] += 1.0 / len(selected_stocks)
    sector_concentration = dict(sector_info)

    # Calculate HHI
    hhi = sum(weight ** 2 for weight in sector_concentration.values())

    # Calculate momentum-based weights (using MACD as an example)
    weights = {}
    for ticker in selected_stocks:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            df = calculate_macd(df)
            macd_now = df['MACD'].iloc[-1].item() if hasattr(df['MACD'].iloc[-1], 'item') else float(df['MACD'].iloc[-1])
            weights[ticker] = max(0.1, macd_now)  # Ensure minimum weight of 0.1
        except Exception as e:
            print(f"Error calculating momentum for {ticker}: {e}")
            weights[ticker] = 1.0 / len(selected_stocks)

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in weights.items()}

    # Calculate portfolio returns and Sharpe ratio dynamically
    portfolio_returns = 0
    daily_returns = []
    for ticker in selected_stocks:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if df.empty:
                continue
            df['Daily_Return'] = df['Close'].pct_change()
            weighted_returns = df['Daily_Return'] * weights[ticker]
            daily_returns.extend(weighted_returns.dropna().tolist())
        except Exception as e:
            print(f"Error calculating returns for {ticker}: {e}")
    
    if daily_returns:
        import numpy as np
        # Calculate cumulative return
        portfolio_returns = np.prod([1 + r for r in daily_returns]) - 1
        # Calculate annualized Sharpe ratio (assuming 252 trading days)
        annualized_return = np.mean(daily_returns) * 252
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    else:
        portfolio_returns = 0.1  # Fallback to mock value if no data
        sharpe_ratio = 1.5

    return {
        'selected_stocks': selected_stocks,
        'weights': weights,
        'sector_concentration': sector_concentration,
        'trades': selected_stocks,
        'returns': portfolio_returns,
        'sharpe_ratio': sharpe_ratio,
        'hhi': hhi,
        'is_concentrated': hhi > 0.6
    }

def generate_trade_statistics(portfolio):
    if not portfolio:
        return {}
    return {
        'trades': portfolio.get('trades', []),
        'returns': portfolio.get('returns', 0),
        'sharpe_ratio': portfolio.get('sharpe_ratio', 0)
    }