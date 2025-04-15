import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Konfiguration ===
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "Transform_data", "raw_data", "2025-2024_BTC-USD_Data_1h.csv")

SHORT_WINDOW = 24   # 24 Stunden MA
LONG_WINDOW = 96   # 96 Stunden MA
INITIAL_CAPITAL = 1
DATE_FORMAT = "%d.%m.%Y %H:%M"

# === CSV laden ===
df = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8-sig')
df.columns = df.columns.str.strip()  # Entfernt etwaige Leerzeichen

# === Datum umwandeln und sortieren ===
df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT, errors='coerce')
df = df.dropna(subset=['date'])  # Entferne fehlerhafte Datumszeilen
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

# === MA berechnen ===
df['short_ma'] = df['close'].rolling(window=SHORT_WINDOW).mean()
df['long_ma'] = df['close'].rolling(window=LONG_WINDOW).mean()

# === Signale generieren ===
df['signal'] = 0
df.loc[df.index[LONG_WINDOW:], 'signal'] = np.where(
    df['short_ma'][LONG_WINDOW:] > df['long_ma'][LONG_WINDOW:], 1, -1
)

df['position_change'] = df['signal'].diff()

# === Strategie-Performance berechnen ===
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
df['portfolio_value'] = INITIAL_CAPITAL * df['cumulative_returns']
df['portfolio_value'] = df['portfolio_value'].bfill()


# === Performance-Metriken ===
final_value = df['portfolio_value'].iloc[-1]
profit = final_value - INITIAL_CAPITAL
time_period_years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
annualized_return = (final_value / INITIAL_CAPITAL) ** (1 / time_period_years) - 1

sharpe_ratio = (
    df['strategy_returns'].mean() / df['strategy_returns'].std()
) * np.sqrt(365.25 * 24) if df['strategy_returns'].std() != 0 else np.nan

df['running_max'] = df['portfolio_value'].cummax()
df['drawdown'] = df['portfolio_value'] / df['running_max'] - 1
max_drawdown = df['drawdown'].min()

# === Trades analysieren ===
trades = []
signals = df[df['position_change'] != 0]
for i in range(len(signals) - 1):
    entry = signals.iloc[i]
    exit = signals.iloc[i + 1]
    if entry['signal'] == 1:
        trade_return = exit['close'] / entry['close'] - 1
    elif entry['signal'] == -1:
        trade_return = entry['close'] / exit['close'] - 1
    else:
        continue
    trades.append(trade_return)

trades = np.array(trades)
win_rate = np.mean(trades > 0) if len(trades) > 0 else np.nan
loss_rate = np.mean(trades < 0) if len(trades) > 0 else np.nan

# === Ergebnisse anzeigen ===
print(f"Final portfolio value: {final_value:.2f} USD")
print(f"Profit: {profit:.2f} USD")
print(f"Annualized Return: {annualized_return*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Loss Rate: {loss_rate*100:.2f}%")

# === Plot 1: Portfolio Value ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Portfolio Value Evolution (MA Crossover Strategy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: MA-Signale im BTC-Chart ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Close Price', alpha=0.7)
plt.plot(df.index, df['short_ma'], label=f'Short MA ({SHORT_WINDOW})', alpha=0.7)
plt.plot(df.index, df['long_ma'], label=f'Long MA ({LONG_WINDOW})', alpha=0.7)

buy_signals = df[(df['signal'] == 1) & (df['signal'].shift(1) == -1)]
sell_signals = df[(df['signal'] == -1) & (df['signal'].shift(1) == 1)]

plt.plot(buy_signals.index, buy_signals['close'], '^', markersize=10, color='green', label='Buy Signal')
plt.plot(sell_signals.index, sell_signals['close'], 'v', markersize=10, color='red', label='Sell Signal')

plt.xlabel('Date')
plt.ylabel('BTC Price (USD)')
plt.title('BTC Price with MA Signals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def run_moving_average_strategy():
    return {
        "portfolio": df['portfolio_value']
    }
