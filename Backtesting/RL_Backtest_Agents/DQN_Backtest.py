#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

# Move up to the correct project root
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

print("Updated Python path:", sys.path)  # Debugging check




# In[ ]:


from stable_baselines3 import DQN
import torch
import random
from Environment.environment import TradingEnv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def run_dqn_backtest():
    # === Vorbereitung ===
    SEED = 42

    # Python
    random.seed(SEED)
    # Numpy
    np.random.seed(SEED)
    # Torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    # Optional: CUDA (falls GPU verwendet wird)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # === Testdaten laden ===
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(BASE_DIR, '..', '..', 'Transform_data', 'stand_data', '2025-2024_stand_data.csv')
    scaler_path = os.path.join(BASE_DIR, '..', '..', 'Transform_data', 'scaler.pkl')


    test_data = pd.read_csv(test_data_path)
    test_data.drop('datetime', axis=1, inplace=True)

    # === Environment erstellen ===
    test_env = TradingEnv(
        data=test_data,
        initial_cash=10_000,
        window_size=336,
        scaler_path=scaler_path,
        default_seed=SEED
    )

    # === Modell laden ===
    model_path = os.path.join(BASE_DIR, '../../Agents/DQN/model_without_buffer')
    model = DQN.load(model_path)

    # === Episode ausführen ===
    reset_result = test_env.reset(seed=SEED)
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    done = False

    action_list = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)  # Wichtig!
        step_result = test_env.step(action)
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
        action_list.append(action)

    # === Ergebnisse anzeigen ===
    test_env.render(mode='human')
    print("Aktionen des Agenten:", action_list)

    from collections import Counter
    action_counts = Counter(action_list)
    actions = list(range(9))
    counts = [action_counts.get(action, 0) for action in actions]

    plt.figure(figsize=(8, 5))
    plt.bar(actions, counts, tick_label=actions)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Agent Action Distribution")
    plt.grid(axis='y')
    plt.show()

    return {
        "portfolio": pd.Series(test_env.portfolio_value_history)
    }



# In[4]:


import numpy as np

def compute_sharpe_ratio(portfolio_values, risk_free_rate=0.0, periods_per_year=8760):
    """
    Compute the Sharpe Ratio using the portfolio returns.

    Parameters:
    - portfolio_values: List or array of portfolio values over time.
    - risk_free_rate: Annual risk-free rate (default: 0).
    - periods_per_year: Number of periods in one year (default: 8760 for hourly data).

    Returns:
    - Sharpe ratio (annualized).
    """
    portfolio_values = np.array(portfolio_values)
    # Calculate period-to-period returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    # Calculate excess returns over the period risk-free rate
    excess_returns = returns - risk_free_rate / periods_per_year
    # Annualize the Sharpe Ratio
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return sharpe_ratio

def compute_max_drawdown(portfolio_values):
    """
    Compute the Maximum Drawdown from the portfolio value history.

    Parameters:
    - portfolio_values: List or array of portfolio values over time.

    Returns:
    - Maximum drawdown as a negative number (e.g., -0.2 means a 20% drawdown).
    """
    portfolio_values = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns)
    return max_drawdown

def compute_annualized_return(portfolio_values, periods_per_year=8760):
    """
    Compute the annualized return (CAGR) based on the portfolio value history.

    Parameters:
    - portfolio_values: List or array of portfolio values over time.
    - periods_per_year: Number of periods in one year.

    Returns:
    - Annualized return as a decimal (e.g., 0.12 for 12% per year).
    """
    portfolio_values = np.array(portfolio_values)
    total_periods = len(portfolio_values)
    total_return = portfolio_values[-1] / portfolio_values[0]
    annualized_return = total_return**(periods_per_year / total_periods) - 1
    return annualized_return

def compute_win_loss_rate(portfolio_values):
    """
    Compute the win-loss rate based on the period-to-period returns.

    Parameters:
    - portfolio_values: List or array of portfolio values over time.

    Returns:
    - A tuple (win_rate, loss_rate) where each value is between 0 and 1.
    """
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    wins = np.sum(returns > 0)
    losses = np.sum(returns <= 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    loss_rate = 1 - win_rate
    return win_rate, loss_rate

def compute_backtest_metrics(portfolio_values, risk_free_rate=0.0, periods_per_year=8760):
    portfolio = pd.Series(portfolio_values)
    returns = portfolio.pct_change().dropna()

    final_portfolio_value = portfolio.iloc[-1]
    profit = final_portfolio_value - portfolio.iloc[0]

    annualized_return = (final_portfolio_value / portfolio.iloc[0]) ** (periods_per_year / len(portfolio)) - 1
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() != 0 else np.nan
    max_drawdown = (portfolio / portfolio.cummax() - 1).min()

    win_rate = (returns > 0).mean()
    loss_rate = (returns < 0).mean()

    return {
        "final_portfolio_value": final_portfolio_value,
        "profit": profit,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "loss_rate": loss_rate
    }

# Example usage with your environment's portfolio history:
# Assuming you have a TradingEnv instance named 'test_env' that has completed an episode:
result = run_dqn_backtest()
portfolio = result["portfolio"]
metrics = compute_backtest_metrics(portfolio)

# Anzeigen
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

