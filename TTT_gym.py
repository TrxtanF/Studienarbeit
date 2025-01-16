import gym
from gym import spaces
import numpy as np
import random

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

        # Actions: Positionen 0..8 (3x3)
        self.action_space = spaces.Discrete(9)
        # Observation: 9 Felder, -1, 0, oder 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=int)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            raise ValueError("Invalid action: Position already occupied")

        self.board[row, col] = self.current_player

        winner = self.check_winner()
        done = (winner is not None) or np.all(self.board != 0)

        # *** Hier haben wir den Reward an den aktuellen Spieler angepasst ***
        if winner == self.current_player:  
            reward = 1.0
        elif winner is not None:  
            reward = -1.0
        elif done:
            reward = 0.5
        else:
            reward = 0.0

        # Spieler wechseln
        self.current_player = -self.current_player

        return self.board.flatten(), reward, done, {}

    def render(self, mode='human'):
        for row in self.board:
            print('|'.join(['x' if x == 1 else 'o' if x == -1 else ' ' for x in row]))
            print('-----')

    def check_winner(self):
        for i in range(3):
            # Zeilen
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
            # Spalten
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]

        # Diagonalen
        diag1 = self.board[0,0] + self.board[1,1] + self.board[2,2]
        diag2 = self.board[0,2] + self.board[1,1] + self.board[2,0]
        if abs(diag1) == 3:
            return self.board[1,1]
        if abs(diag2) == 3:
            return self.board[1,1]

        return None

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, gamma=0.9, epsilon=0.2):
        self.action_space = action_space
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        state_key = tuple(state)
        valid_actions = [i for i, x in enumerate(state) if x == 0]

        # Epsilon-greedy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)

        # Wähle beste Aktion
        q_vals = self.q_table[state_key]
        # Nur gültige Aktionen betrachten
        valid_q_vals = [(act, q_vals[act]) for act in valid_actions]
        best_action = max(valid_q_vals, key=lambda x: x[1])[0]
        return best_action

    def update(self, state, action, reward, next_state):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)

        old_value = self.q_table[state_key][action]
        future_max = np.max(self.q_table[next_state_key])
        new_value = old_value + self.lr * (reward + self.gamma * future_max - old_value)
        self.q_table[state_key][action] = new_value

    # Speichere das Q-Table in einer .npy Datei
    def save(self, filename):
        np.save(filename, self.q_table)

    # Lade das Q-Table aus einer .npy Datei
    def load(self, filename):
        loaded = np.load(filename, allow_pickle=True)
        self.q_table = loaded.item()  # da wir ein Dictionary gespeichert haben

def train_two_agents_selfplay(env, agentX, agentO, episodes=50000):
    """
    Trainiert zwei Q-Learning-Agenten gegeneinander.
    agentX ist immer Spieler +1,
    agentO ist immer Spieler -1.
    """
    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            current_player = env.current_player

            if current_player == 1:
                action = agentX.choose_action(state)
            else:
                action = agentO.choose_action(state)

            next_state, reward, done, _ = env.step(action)

            # Die Umwelt hat den Reward für DEN Spieler zurückgegeben, der gerade gezogen hat
            # => Also update den Agent, der gerade am Zug war
            if current_player == 1:
                agentX.update(state, action, reward, next_state)
            else:
                agentO.update(state, action, reward, next_state)

            state = next_state

def play_against_agent(env, agent, agent_symbol):
    """
    Mensch vs. Agent.
    - agent_symbol = 1  => Der Agent spielt X, Mensch spielt O
    - agent_symbol = -1 => Der Agent spielt O, Mensch spielt X
    """
    state = env.reset()
    done = False

    if agent_symbol == 1:
        print("Du bist Spieler 'o' (Symbol -1), der Agent ist 'x' (Symbol 1).")
    else:
        print("Du bist Spieler 'x' (Symbol 1), der Agent ist 'o' (Symbol -1).")

    env.render()

    while not done:
        if env.current_player == agent_symbol:
            # Agent
            print("Agent ist am Zug...")
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                if reward == 1.0:
                    print("Der Agent hat gewonnen. 😢")
                elif reward == 0.5:
                    print("Unentschieden!")
                else:
                    print("Du hast gewonnen! 🎉")
        else:
            # Mensch
            valid_actions = [i for i, x in enumerate(state) if x == 0]
            print(f"Verfügbare Aktionen: {valid_actions}")
            while True:
                try:
                    human_action = int(input("Wähle deine Aktion (0-8): "))
                    if human_action in valid_actions:
                        break
                    else:
                        print("Ungültige Aktion. Versuche es erneut.")
                except ValueError:
                    print("Bitte gib eine Zahl zwischen 0 und 8 ein.")

            state, reward, done, _ = env.step(human_action)
            env.render()
            if done:
                if reward == 1.0:
                    print("Du hast gewonnen! 🎉")
                elif reward == 0.5:
                    print("Unentschieden!")
                else:
                    print("Der Agent hat gewonnen. 😢")

def main():
    env = TicTacToeEnv()
    agentX = QLearningAgent(env.action_space, epsilon=0.5)  # Start: eher große Epsilon
    agentO = QLearningAgent(env.action_space, epsilon=0.5)

    print("Starte Training im Self-Play...")
    train_two_agents_selfplay(env, agentX, agentO, episodes=50000)

    # Nach Training Epsilon runtersetzen
    agentX.epsilon = 0.0
    agentO.epsilon = 0.0

    # Q-Table speichern
    agentX.save("agentX.npy")
    agentO.save("agentO.npy")
    print("Training fertig. Q-Tables gespeichert.")

    # Beispiel: Gegen AgentX spielen (als O)
    print("\nTeste: Du spielst O, AgentX ist X...")
    play_against_agent(env, agentX, agent_symbol=1)

    # Beispiel: Gegen AgentO spielen (als X)
    print("\nTeste: Du spielst X, AgentO ist O...")
    play_against_agent(env, agentO, agent_symbol=-1)

if __name__ == "__main__":
    main()
