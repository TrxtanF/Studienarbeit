#!/usr/bin/env python3
import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# --------------------------
# Environment-Definition
# --------------------------
class BitcoinTradingEnv(gym.Env):
    """
    Bitcoin Trading Environment (ohne Transaktionskosten).
    Portfolio startet als Multiplikator bei 1.0.
    
    Aktionen:
      0: Halten
      1: Kaufen (nur, wenn nicht investiert)
      2: Verkaufen (nur, wenn investiert)
    
    Der Datenbereich (date_range) wird zur Filterung der historischen Daten genutzt.
    Beim Rendern und Plotten werden die tatsächlichen Datumswerte verwendet.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, window_size=30, initial_capital=1.0,
                 overall_start_date='2014-09-17', overall_end_date='2025-12-31',
                 date_range=None):
        super(BitcoinTradingEnv, self).__init__()
        self.window_size = window_size
        self.initial_capital = initial_capital
        
        # Lade historische Bitcoin-Daten
        self.df = yf.download('BTC-USD', start=overall_start_date, end=overall_end_date)
        self.df.columns = self.df.columns.get_level_values(0)
        # Filtere Daten anhand des angegebenen Datumsbereichs (z. B. Training: 2015-2020)
        if date_range is not None:
            self.df = self.df.loc[date_range[0]:date_range[1]]
            
        self.data = self.df[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
        self.num_features = self.data.shape[1]
        
        # Skaliere die Features in den Bereich [0, 1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_features = self.scaler.fit_transform(self.data.values)
        self.close_prices = self.df['Close'].values
        
        # Aktionsraum: 0 = Halten, 1 = Kaufen, 2 = Verkaufen
        self.action_space = spaces.Discrete(3)
        # Beobachtungsraum: Fenster (aktueller Ausschnitt der skalierten Daten),
        # Positionsstatus (0 oder 1) und Portfolio (Multiplikator)
        self.observation_space = spaces.Dict({
            'window': spaces.Box(low=0, high=1, shape=(self.window_size, self.num_features), dtype=np.float32),
            'position': spaces.Discrete(2),
            'portfolio': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        self.reset()
    
    def reset(self):
        self.current_step = self.window_size  # Beginne, wenn ein komplettes Fenster vorliegt
        self.portfolio = self.initial_capital    # Startwert = 1.0
        self.position = 0  # 0 = nicht investiert, 1 = investiert
        self.buy_price = None
        self.portfolio_history = [self.portfolio]
        return self._get_observation()
    
    def _get_observation(self):
        return {
            'window': self.scaled_features[self.current_step - self.window_size : self.current_step].astype(np.float32),
            'position': self.position,
            'portfolio': np.array([self.portfolio], dtype=np.float32)
        }
    
    def step(self, action):
        done = False
        reward = 0
        penalty = 10
        
        # Wenn das Ende der Daten erreicht ist
        if self.current_step >= len(self.scaled_features) - 1:
            done = True
            return self._get_observation(), reward, done, {}
        
        # Berechne Preisvergleich: Vortag vs. aktueller Tag
        prev_price = self.close_prices[self.current_step - 1]
        current_price = self.close_prices[self.current_step]
        prev_portfolio = self.portfolio
        
        # Falls investiert, aktualisiere Portfolio als Multiplikator
        if self.position == 1:
            self.portfolio *= (current_price / prev_price)
            
        # Verarbeite die Aktion
        if action == 1:  # Kaufen
            if self.position == 0:
                self.position = 1
                self.buy_price = current_price
            else:
                reward -= penalty
        elif action == 2:  # Verkaufen
            if self.position == 1:
                self.position = 0
                self.buy_price = None
            else:
                reward -= penalty
        elif action == 0:  # Halten
            pass
        else:
            raise ValueError("Ungültige Aktion")
        
        # Reward entspricht der Änderung des Portfolio-Multiplikators
        reward += self.portfolio - prev_portfolio
        
        self.current_step += 1
        self.portfolio_history.append(self.portfolio)
        if self.current_step >= len(self.scaled_features) - 1:
            done = True
        
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        # Hole das aktuelle Datum aus dem DataFrame-Index
        date = self.df.index[self.current_step].strftime("%Y-%m-%d")
        status = "Invested" if self.position == 1 else "Not invested"
        current_price = self.close_prices[self.current_step]
        print(f"Date: {date} | Portfolio: {self.portfolio:.4f} | Position: {status} | Price: {current_price:.2f}")
    
    def plot_comparison_log(self):
        # Berechne Buy-and-Hold-Multiplikator (angenommen, ab Beginn der Episode investiert)
        initial_index = self.window_size
        initial_price = self.close_prices[initial_index]
        buy_and_hold = [
            self.initial_capital * (self.close_prices[initial_index + i] / initial_price)
            for i in range(len(self.portfolio_history))
        ]
        # Erzeuge X-Achse als Datum: Entspricht den Zeitpunkten, an denen das Portfolio gemessen wurde
        dates = self.df.index[self.window_size:self.window_size + len(self.portfolio_history)]
        plt.figure(figsize=(10, 5))
        plt.plot(dates, self.portfolio_history, label="Trading Portfolio")
        plt.plot(dates, buy_and_hold, label="Buy and Hold Portfolio", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Multiplier")
        plt.title("Trading vs. Buy and Hold (Log Scale)")
        plt.yscale("log")
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()


# --------------------------
# Hilfsfunktion zum Flatten des Zustands für DQN
# --------------------------
def flatten_state(state):
    window = state['window'].flatten()
    position = np.array([state['position']], dtype=np.float32)
    portfolio = state['portfolio']
    return np.concatenate([window, position, portfolio])


# --------------------------
# DQN-Agent-Definition
# --------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


# --------------------------
# Training und Testen
# --------------------------
if __name__ == '__main__':
    # Hyperparameter
    num_episodes = 50
    batch_size = 64
    gamma = 0.99
    learning_rate = 1e-3
    buffer_capacity = 10000
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    target_update_freq = 5  # Aktualisiere das Target-Netzwerk alle 5 Episoden
    
    window_size = 30
    # Training: 2015-01-01 bis 2020-12-31
    train_env = BitcoinTradingEnv(window_size=window_size, initial_capital=1.0,
                                  overall_start_date='2014-09-17', overall_end_date='2025-12-31',
                                  date_range=("2015-01-01", "2020-12-31"))
    
    sample_obs = train_env.reset()
    flat_sample = flatten_state(sample_obs)
    input_dim = flat_sample.shape[0]
    output_dim = train_env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(input_dim, output_dim).to(device)
    target_net = QNetwork(input_dim, output_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon_start
    
    print("Training started...")
    for episode in range(num_episodes):
        state = train_env.reset()
        state_flat = flatten_state(state)
        total_reward = 0
        done = False
        while not done:
            # Epsilon-greedy Strategie
            if random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            next_state, reward, done, _ = train_env.step(action)
            next_state_flat = flatten_state(next_state)
            replay_buffer.push(state_flat, action, reward, next_state_flat, done)
            state_flat = next_state_flat
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                q_values = q_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.4f} - Epsilon: {epsilon:.4f}")
        if (episode+1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
    print("Training completed.")
    
    # --------------------------
    # Testphase: 2021-01-01 bis 2025-12-31
    # --------------------------
    test_env = BitcoinTradingEnv(window_size=window_size, initial_capital=1.0,
                                 overall_start_date='2014-09-17', overall_end_date='2025-12-31',
                                 date_range=("2021-01-01", "2025-12-31"))
    
    state = test_env.reset()
    state_flat = flatten_state(state)
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        next_state, reward, done, _ = test_env.step(action)
        state_flat = flatten_state(next_state)
    test_env.plot_comparison_log()
