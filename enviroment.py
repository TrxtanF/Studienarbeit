"""
Bitcoin Trading Environment mit Vergleich: Trading vs. Buy and Hold (logarithmische Darstellung)

Dieses Skript beinhaltet:
- Definition eines Bitcoin-Trading-Environments basierend auf historischen Daten von yfinance.
- Simulation einer Episode mit zufälligen Aktionen.
- Berechnung des Buy-and-Hold-Verlaufs (d.h. wenn man ab Episodenstart investiert und bis zum Ende hält).
- Darstellung eines logarithmischen Charts, in dem der Portfolio-Verlauf der Trading-Strategie
  und der Buy-and-Hold-Strategie verglichen werden.
"""

# Imports und Setup
import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class BitcoinTradingEnv(gym.Env):
    """
    Ein einfaches Reinforcement-Learning-Environment für den Bitcoin-Handel (ohne Transaktionskosten).

    Aktionen:
      - 0: Halten
      - 1: Kaufen (nur möglich, wenn nicht bereits investiert)
      - 2: Verkaufen (nur möglich, wenn investiert)

    Beobachtungen (als Dictionary):
      - 'window': Fenster der letzten 'window_size' Tage der skalierten Daten (Close, High, Low, Open, Volume)
      - 'position': 0 (nicht investiert) oder 1 (investiert)
      - 'portfolio': Normalisierter Portfolio-Wert (bezogen auf das initiale Kapital)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, window_size=30, initial_capital=10000, 
                 start_date='2014-09-17', end_date='2025-01-28'):
        super(BitcoinTradingEnv, self).__init__()
        
        self.window_size = window_size
        self.initial_capital = initial_capital
        
        # Bitcoin-Daten laden
        self.df = yf.download('BTC-USD', start=start_date, end=end_date)
        self.df.columns = self.df.columns.get_level_values(0)  # Entferne eventuelle MultiIndex-Ebenen
        
        # Relevante Features auswählen
        self.data = self.df[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
        self.num_features = self.data.shape[1]
        
        # Skaliere die Features in den Bereich [0, 1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_features = self.scaler.fit_transform(self.data.values)
        
        # Speichere die rohen Schlusskurse zur Berechnung des Portfolio-Werts
        self.close_prices = self.df['Close'].values
        
        # Definiere den Aktionsraum: 0 (Halten), 1 (Kaufen), 2 (Verkaufen)
        self.action_space = spaces.Discrete(3)
        
        # Definiere den Beobachtungsraum als Dictionary
        self.observation_space = spaces.Dict({
            'window': spaces.Box(low=0, high=1, shape=(self.window_size, self.num_features), dtype=np.float32),
            'position': spaces.Discrete(2),
            'portfolio': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        self.reset()
        
    def reset(self):
        """
        Setzt das Environment zurück und startet eine neue Episode.
        """
        self.current_step = self.window_size  # Start, wenn ein vollständiges Fenster verfügbar ist
        self.portfolio = self.initial_capital
        self.position = 0  # 0: nicht investiert, 1: investiert
        self.buy_price = None  # Speichert den Kaufpreis, falls investiert
        self.portfolio_history = [self.portfolio]  # Aufzeichnung des Portfolio-Verlaufs
        return self._get_observation()
    
    def _get_observation(self):
        """
        Erzeugt die aktuelle Beobachtung als Dictionary.
        """
        obs = {
            'window': self.scaled_features[self.current_step - self.window_size : self.current_step].astype(np.float32),
            'position': self.position,
            'portfolio': np.array([self.portfolio / self.initial_capital], dtype=np.float32)
        }
        return obs
    
    def step(self, action):
        """
        Führt einen Zeitschritt im Environment aus.

        Parameter:
          - action: 0 (Halten), 1 (Kaufen) oder 2 (Verkaufen)

        Rückgabe:
          - observation: aktuelle Beobachtung (Dictionary)
          - reward: Belohnung (Veränderung des Portfolio-Werts plus Strafpunkte bei ungültigen Aktionen)
          - done: Bool, ob die Episode beendet ist
          - info: zusätzliches Info-Dictionary (hier leer)
        """
        done = False
        reward = 0
        penalty = 10  # Strafwert für ungültige Aktionen
        
        # Wenn das Ende der Daten erreicht ist, beende die Episode
        if self.current_step >= len(self.scaled_features) - 1:
            done = True
            return self._get_observation(), reward, done, {}
        
        # Preisvergleich: Vortag vs. aktueller Tag
        prev_price = self.close_prices[self.current_step - 1]
        current_price = self.close_prices[self.current_step]
        
        # Aktualisiere das Portfolio, falls investiert (all-in-Ansatz)
        prev_portfolio = self.portfolio
        if self.position == 1:
            self.portfolio = self.portfolio * (current_price / prev_price)
        
        # Verarbeite die Aktion
        if action == 1:  # Kaufen
            if self.position == 0:
                self.position = 1
                self.buy_price = current_price
            else:
                reward -= penalty  # Ungültige Aktion: Bereits investiert
        elif action == 2:  # Verkaufen
            if self.position == 1:
                self.position = 0
                self.buy_price = None
            else:
                reward -= penalty  # Ungültige Aktion: Nicht investiert
        elif action == 0:  # Halten
            pass
        else:
            raise ValueError("Ungültige Aktion. Erlaubt sind 0 (Halten), 1 (Kaufen) oder 2 (Verkaufen).")
        
        # Belohnung entspricht der Änderung des Portfolio-Werts
        reward += self.portfolio - prev_portfolio
        
        # Aktualisiere den Zeitschritt und speichere den Portfolio-Wert
        self.current_step += 1
        self.portfolio_history.append(self.portfolio)
        
        if self.current_step >= len(self.scaled_features) - 1:
            done = True
        
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        """
        Gibt den aktuellen Status des Environments aus.
        """
        status = "Investiert" if self.position == 1 else "Nicht investiert"
        current_price = self.close_prices[self.current_step]
        print(f"Schritt: {self.current_step} | Portfolio: {self.portfolio:.2f} | Position: {status} | Preis: {current_price:.2f}")
    
    def plot_comparison_log(self):
        """
        Plottet den Verlauf des simulierten Trading-Portfolios und vergleicht diesen
        mit dem Buy-and-Hold-Portfolio (d.h. wenn man ab Episodenstart investiert und bis zum Ende hält)
        in einem logarithmischen Chart.
        """
        # Zeitpunkt des Episodenstarts (nach dem initialen Fenster)
        initial_index = self.window_size  
        initial_price = self.close_prices[initial_index]
        
        # Berechne für jeden Zeitschritt den Buy-and-Hold-Wert:
        buy_and_hold = [
            self.initial_capital * (self.close_prices[initial_index + i] / initial_price)
            for i in range(len(self.portfolio_history))
        ]
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio_history, label="Simuliertes Trading Portfolio")
        plt.plot(buy_and_hold, label="Buy and Hold Portfolio", linestyle="--")
        plt.xlabel("Schritte")
        plt.ylabel("Portfolio-Wert")
        plt.title("Vergleich: Trading vs. Buy and Hold (logarithmische Skala)")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
    
    def close(self):
        pass


# Simulation einer Episode und Vergleichsplot (logarithmische Darstellung)
if __name__ == '__main__':
    env = BitcoinTradingEnv()
    obs = env.reset()
    
    # Simuliere eine Episode (zufällige Aktionen)
    done = False
    while not done:
        action = env.action_space.sample()  # Zufällige Aktion (0, 1 oder 2)
        obs, reward, done, info = env.step(action)
    
    # Plot zum Vergleich: Trading vs. Buy and Hold (logarithmische Skala)
    env.plot_comparison_log()
