import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file (update the filename as needed)
df = pd.read_csv('Transform_data/raw_data/2023-2018_BTC-USD_Data_1h.csv')

# Convert the 'date' column to datetime. The format is day.month.year hour:minute.
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

# Define moving average windows
short_window = 24
long_window = 96

# Calculate short and long moving averages on the 'close' price.
df['short_ma'] = df['close'].rolling(window=short_window).mean()
df['long_ma'] = df['close'].rolling(window=long_window).mean()

# Generate trading signals:
# Signal = 1 when short_ma > long_ma (long position)
# Signal = -1 when short_ma <= long_ma (short position)
df['signal'] = 0  # Default signal value

# Only assign signals once there's enough data for the long moving average.
df.loc[df.index[long_window:], 'signal'] = np.where(
    df['short_ma'].loc[df.index[long_window:]] > df['long_ma'].loc[df.index[long_window:]], 1, -1
)

# Identify trading events: A change in signal indicates a trade.
df['position_change'] = df['signal'].diff()

# Backtesting: Compute percentage returns of the close price.
df['returns'] = df['close'].pct_change()

# Compute strategy returns using the previous period's signal.
df['strategy_returns'] = df['returns'] * df['signal'].shift(1)

# Calculate cumulative returns of the strategy.
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

# Set the initial capital and compute portfolio value over time.
initial_capital = 10000
df['portfolio_value'] = initial_capital * df['cumulative_returns']

# Get final portfolio value and profit.
final_portfolio_value = df['portfolio_value'].iloc[-1]
profit = final_portfolio_value - initial_capital

print(f"Final portfolio value: {final_portfolio_value:.2f} USD")
print(f"Profit: {profit:.2f} USD")

# ------------------ Performance Metrics ------------------

# 1. Annualized Return
# Calculate the total time period in years.
time_period = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
annualized_return = (final_portfolio_value / initial_capital) ** (1 / time_period) - 1

# 2. Sharpe Ratio
# Here we assume a risk-free rate of 0.
# Since the data is hourly, annualization factor = 8760 (365.25 * 24).
mean_strategy_return = df['strategy_returns'].mean()
std_strategy_return = df['strategy_returns'].std()
annual_factor = 365.25 * 24  # hourly data
sharpe_ratio = (mean_strategy_return / std_strategy_return) * np.sqrt(annual_factor)

# 3. Maximum Drawdown
df['running_max'] = df['portfolio_value'].cummax()
df['drawdown'] = df['portfolio_value'] / df['running_max'] - 1
max_drawdown = df['drawdown'].min()

# 4 & 5. Win Rate and Loss Rate from individual trades
# We define a trade as the period between two consecutive position changes.
trade_signals = df[df['position_change'] != 0][['close', 'signal']]
trade_signals = trade_signals.sort_index()
trades = []
for i in range(len(trade_signals) - 1):
    entry = trade_signals.iloc[i]
    exit = trade_signals.iloc[i+1]
    # For a long trade, profit = (exit_price/entry_price - 1)
    # For a short trade, profit = (entry_price/exit_price - 1)
    if entry['signal'] == 1:
        trade_return = exit['close'] / entry['close'] - 1
    elif entry['signal'] == -1:
        trade_return = entry['close'] / exit['close'] - 1
    else:
        continue 
    trades.append(trade_return)

trades = np.array(trades)
win_rate = np.sum(trades > 0) / len(trades) if len(trades) > 0 else np.nan
loss_rate = np.sum(trades < 0) / len(trades) if len(trades) > 0 else np.nan

print(f"Annualized Return: {annualized_return*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Loss Rate: {loss_rate*100:.2f}%")

# ------------------ Plots ------------------

# Plot 1: Portfolio Value Evolution
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.title('Portfolio Value Evolution with Initial Capital $10,000')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Bitcoin Price Chart with Moving Averages and Trading Signals
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='BTC Close Price', alpha=0.7)
plt.plot(df.index, df['short_ma'], label=f'Short MA ({short_window})', alpha=0.7)
plt.plot(df.index, df['long_ma'], label=f'Long MA ({long_window})', alpha=0.7)

# Mark Buy signals (transition from -1 to 1)
buy_signals = df[(df['signal'] == 1) & (df['signal'].shift(1) == -1)]
# Mark Sell signals (transition from 1 to -1)
sell_signals = df[(df['signal'] == -1) & (df['signal'].shift(1) == 1)]

plt.plot(buy_signals.index, buy_signals['close'], '^', markersize=10, color='green', label='Buy Signal')
plt.plot(sell_signals.index, sell_signals['close'], 'v', markersize=10, color='red', label='Sell Signal')

plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Price Chart with Moving Averages and Trading Signals')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Save the backtest results and metrics to a CSV file.
df.to_csv('backtest_results.csv')
