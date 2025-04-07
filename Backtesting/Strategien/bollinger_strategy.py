import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Konfiguration ===
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "Transform_data", "raw_data", "2025-2024_BTC-USD_Data_1h.csv")
INITIAL_CAPITAL = 10_000
DATE_FORMAT = "%d.%m.%Y %H:%M"

# === Daten laden ===
df = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8-sig')
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT, errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').set_index('date')

# === Bollinger Bands Berechnung ===
WINDOW = 20
df['sma'] = df['close'].rolling(window=WINDOW).mean()
df['std'] = df['close'].rolling(window=WINDOW).std()
df['upper_band'] = df['sma'] + 2 * df['std']
df['lower_band'] = df['sma'] - 2 * df['std']

# === Signal generieren ===
df['signal'] = 0
df.loc[df['close'] < df['lower_band'], 'signal'] = 1   # Buy
df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell
df['position'] = df['signal'].replace(to_replace=0, method='ffill')

# === Strategieertrag berechnen ===
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['returns'] * df['position'].shift(1)
df['portfolio'] = INITIAL_CAPITAL * (1 + df['strategy_returns']).cumprod()

# === Performance-Metriken ===
final_value = df['portfolio'].iloc[-1]
profit = final_value - INITIAL_CAPITAL
percentage_return = (final_value / INITIAL_CAPITAL - 1) * 100

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['portfolio'], label='Bollinger Bands Strategie')
plt.title(f"Bollinger Bands Strategie – Endwert: {final_value:.2f} € ({percentage_return:.2f}%)")
plt.xlabel("Datum")
plt.ylabel("Portfoliowert (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def run_bollinger_strategy():
    return {
        "portfolio": df['portfolio']
    }
