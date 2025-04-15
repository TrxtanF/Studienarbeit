#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
import sys

# Move up to the correct project root
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(project_root)

print("Updated Python path:", sys.path)  # Debugging check


# In[ ]:


VERSION = 3
OUTPUT_NAME = f"PPO_Backtest_v{VERSION}"

if VERSION == 1:
    MODEL_PATH = 'Without_1_EUR_200K'
elif VERSION == 2:
    MODEL_PATH = 'Without_Optuna_1_EUR_200K'
elif VERSION == 3:
    MODEL_PATH = 'Without_Custom_Small_1_EUR_200K'
elif VERSION == 4:
    MODEL_PATH = 'Without_Custom_Deep_1_EUR_200K'
else:
    raise Exception("Fehlerhafte Version")


# In[32]:


#get_ipython().system('jupyter nbconvert --to script "PPO_Backtest.ipynb" --output "{OUTPUT_NAME}"')


# In[ ]:


from stable_baselines3 import DQN, PPO, A2C
import torch
import random
from Environment.environment_withPortfolio import TradingEnv_withPortfolio
from Environment.environment_withoutPortfolio import TradingEnv_withoutPortfolio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter


#TradingEnv = TradingEnv_withPortfolio
TradingEnv = TradingEnv_withoutPortfolio


#def run_ppo_backtest_v1():
#def run_ppo_backtest_v2():
def run_ppo_backtest_v3():
#def run_ppo_backtest_v4():

    
    # === Setup ===
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # === Pfade dynamisch bestimmen ===
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    test_data_path = os.path.join(BASE_DIR, '..', '..', 'Transform_data', 'stand_data', '2025-2024_stand_data.csv')
    scaler_path = os.path.join(BASE_DIR, '..', '..', 'Transform_data', 'scaler.pkl')


    ##### Hier muss der Pfad zu der korrekt trainierten Datei führen ###################
    model_path = os.path.join(BASE_DIR, '..', '..', 'Agents', 'PPO', MODEL_PATH) 

    # === Daten laden ===
    test_data = pd.read_csv(test_data_path)

    if 'date' in test_data.columns:
        test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
        test_data.set_index('date', inplace=True)
    elif 'datetime' in test_data.columns:
        test_data['datetime'] = pd.to_datetime(test_data['datetime'], errors='coerce')  
        test_data.set_index('datetime', inplace=True)
    else:
        raise ValueError("Keine gültige Zeitspalte ('date' oder 'datetime') in test_data gefunden.")

    test_data.dropna(inplace=True)


    # Speichere den Index separat
    full_index = test_data.index

    # === Environment vorbereiten ===
    test_env = TradingEnv(
        data=test_data,
        initial_cash=1,
        window_size=336,
        scaler_path=scaler_path,
        default_seed=SEED
    )

    # === Modell laden ===
    model = PPO.load(model_path)

    # === Episode ausführen ===
    reset_result = test_env.reset(seed=SEED)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    done = False
    action_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        step_result = test_env.step(action)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        done = step_result[2] if isinstance(step_result, tuple) and len(step_result) >= 3 else False
        action_list.append(action)

    # === Portfolio mit Zeitindex erstellen ===
    portfolio_values = test_env.portfolio_value_history
    portfolio_index = full_index[-len(portfolio_values):]
    portfolio_series = pd.Series(portfolio_values, index=portfolio_index)

    # === Action-Verteilung plotten ===
    action_counts = Counter(action_list)
    actions = list(range(max(action_list) + 1))
    counts = [action_counts.get(a, 0) for a in actions]

    plt.figure(figsize=(8, 5))
    plt.bar(actions, counts, tick_label=actions)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title(f"PPO_v{VERSION} Agent Action Distribution")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    

    return {
        "portfolio": portfolio_series,
        "actions": action_list
    }

