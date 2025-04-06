#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().run_line_magic('run', '../../Environment/environment.ipynb')


# In[11]:


import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import Counter


# In[12]:


seed = 42
SEED  = seed % (2**32 - 1)
print(f"SEED: {SEED}")


# In[13]:


# -------------------------------
# CSV Daten einlesen
# -------------------------------
train_data = pd.read_csv("../../Transform_data/stand_data/2023-2018_stand_data.csv")
train_data.drop('datetime', axis=1, inplace=True)

test_data = pd.read_csv("../../Transform_data/stand_data/2025-2024_stand_data.csv")
test_data.drop('datetime', axis=1, inplace=True)

if train_data is not None and test_data is not None:
    print("Daten erfolgreich eingelesen")


# In[14]:


# Für Hyperparameter-Tuning nutzen wir test_data als Validierungsdatensatz.
valid_data = test_data.copy()

# -------------------------------
# Evaluation Helper Function
# -------------------------------
def evaluate_agent(model, env, n_eval_episodes=5):
    """
    Evaluate the model over a number of episodes.
    Returns the average cumulative reward.
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        # Handle reset return for compatibility with gym vs. gymnasium APIs
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            # Check length to handle different API outputs
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
            total_reward += reward
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards)


# In[15]:


# -------------------------------
# Hyperparameter Tuning with Optuna
# -------------------------------
def objective(trial):
    # Sample hyperparameters using updated Optuna functions
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Create the training environment using train_data
    env_train = DummyVecEnv([lambda: TradingEnv(
        data=train_data,
        initial_cash=10_000,
        window_size=336,
        scaler_path="../../Transform_data/scaler.pkl",
        default_seed=SEED
    )])

    # Create the validation environment using valid_data
    env_valid = DummyVecEnv([lambda: TradingEnv(
        data=valid_data,
        initial_cash=10_000,
        window_size=336,
        scaler_path="../../Transform_data/scaler.pkl",
        default_seed=SEED
    )])

    # Initialize the DQN model with sampled hyperparameters
    model = DQN(
        "MlpPolicy",
        env_train,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        verbose=0,
        seed=SEED,
        buffer_size=100_000
    )

    # Train the model for a fixed number of timesteps
    model.learn(total_timesteps=5000, log_interval=1)

    # Evaluate the trained model on the validation environment
    mean_reward = evaluate_agent(model, env_valid, n_eval_episodes=5)
    return mean_reward


# In[ ]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best hyperparameters:", study.best_trial.params)


# In[18]:


# -------------------------------
# TradingEnv erstellen (Final Environment using train_data)
# -------------------------------
env = TradingEnv(
    data=train_data,
    initial_cash=10_000,
    window_size=336,
    scaler_path="../../Transform_data/scaler.pkl",
    default_seed=SEED
)

if env is not None:
    print("Environment created successfully")


# In[20]:


# -------------------------------
# DQN-Agenten initialisieren using Best Hyperparameters
# -------------------------------
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    seed=SEED,
    buffer_size=500_000,
    learning_rate=0.0008676988081645593,
    gamma=0.9567540210286709,
    batch_size=256
)


# In[21]:


# -------------------------------
# Training
# -------------------------------
model.learn(
    total_timesteps=10000, 
    log_interval=1
)

# Speichere das trainierte Modell
model.save("dqn_trading_model")


# In[22]:


# -------------------------------
# Testlauf: Den trainierten Agenten in einer Episode ausführen (Training Environment)
# -------------------------------
training_env = env

reset_result = training_env.reset(seed=SEED)
if isinstance(reset_result, tuple):
    obs, info = reset_result
else:
    obs = reset_result
done = False

# Liste der Aktionen
action_list = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)  # oder: action = action.item()
    step_result = training_env.step(action)
    if len(step_result) == 5:
        obs, reward, done, truncated, info = step_result
    else:
        obs, reward, done, info = step_result
    action_list.append(action)

# Rendern des aktuellen Zustands (z.B. als Plot)
training_env.render(mode='human')
print(action_list)


# In[ ]:


# -------------------------------
# Plot Action Distribution
# -------------------------------
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


# In[ ]:


# -------------------------------
# Testlauf: Den trainierten Agenten in einer Episode ausführen
# -------------------------------
test_env = TradingEnv(
    data=test_data,
    initial_cash=10_000,
    window_size=336,
    scaler_path="../../Transform_data/scaler.pkl",
    default_seed=SEED
)

obs, info = test_env.reset(seed=SEED)
done = False

# Liste der actionen
action_list = []

while not done:
    # Bestimme die Aktion (deterministisch)
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)  # oder: action = action.item()
    obs, reward, done, truncated, info = test_env.step(action)
    action_list.append(action)

# Hier wird der Zustand gerendert (z.B. als Plot). Du kannst den Render-Modus anpassen.
test_env.render(mode='human')
print(action_list)


# In[ ]:


# -------------------------------
# Plot Action Distribution
# -------------------------------
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

