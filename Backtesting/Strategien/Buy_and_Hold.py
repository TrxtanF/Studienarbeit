import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Konfiguration ===
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "..", "Transform_data", "raw_data", "2025-2024_BTC-USD_Data_1h.csv")

INITIAL_CAPITAL = 1
DATE_FORMAT = "%d.%m.%Y %H:%M"

def run_buy_and_hold_strategy():
    # === Daten laden ===
    df = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'], format=DATE_FORMAT, errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').set_index('date')

    # === Returns berechnen ===
    df['returns'] = df['close'].pct_change()

    # === Buy & Hold Position (immer 1) ===
    df['position'] = 1

    # === Strategieertrag berechnen ===
    df['strategy_returns'] = df['returns'] * df['position'].shift(1)
    df['portfolio'] = INITIAL_CAPITAL * (1 + df['strategy_returns'].fillna(0)).cumprod()
    df['portfolio'] = df['portfolio'].bfill()

    # === Performance-Metriken (optional) ===
    final_value = df['portfolio'].iloc[-1]
    percentage_return = (final_value / INITIAL_CAPITAL - 1) * 100

    # === Plot ===
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['portfolio'], label='Buy & Hold Strategie')
    plt.title(f"Buy & Hold – Endwert: {final_value:.2f} € ({percentage_return:.2f}%)")
    plt.xlabel("Datum")
    plt.ylabel("Portfoliowert (€)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "portfolio": df['portfolio']
    }
