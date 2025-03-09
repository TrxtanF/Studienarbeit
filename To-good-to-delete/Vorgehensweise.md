### **Phase 1: Grundlagen schaffen und grobe Planung (01.12.2024 – 15.01.2025)**

#### **Bis Weihnachten: Fokus auf Klausuren (01.12. – 24.12.):**
1. **Minimaler Aufwand für die Studienarbeit:**
   - **Individuelle Vorbereitung (1–2 Stunden/Woche):**
     - Jeder aus dem Team liest einen kurzen Artikel oder schaut ein Einführungsvideo:
       - Reinforcement Learning: OpenAI-Artikel oder Andrew Ng’s Einführung in ML auf YouTube.
       - Trading: Grundlagenartikel auf Investopedia.
     - Ziel: Verständnis der Grundbegriffe und Schlüsselkonzepte.
   - **Aufgabe:** Jeder erstellt eine 1-seitige Zusammenfassung seiner Erkenntnisse und teilt diese mit dem Team.

#### **Zwischen den Feiertagen: Erste Abstimmung (27.12. – 30.12.):**
- **Teamtreffen (2–3 Stunden, online oder vor Ort):**
  - Ziele:
    1. Aufgabenverteilung für die nächste Phase.
    2. Auswahl eines RL-Algorithmus (z. B. PPO oder DQN) basierend auf Tutorials.
    3. Auswahl eines Trading-Marktes (Aktien oder Kryptowährungen).

---

### **Phase 2: Datenrecherche und technische Grundlagen (15.01.2025 – 15.02.2025)**

#### **Schritt 1: Datenbeschaffung (15.01. – 25.01.):**
- **Datenquellen:** 
  - Yahoo Finance (kostenlos für Aktien) oder Binance API (für Kryptowährungen).
- **Vorgehen:**
  - Registriert euch bei einer Plattform (z. B. Binance).
  - Nutzt Python-Bibliotheken wie `yfinance` oder `ccxt`, um historische Daten herunterzuladen.
  - **Code-Beispiel:**
    ```python
    import yfinance as yf
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    print(data.head())
    ```
  - Speichert die Daten als CSV, damit ihr sie später nutzen könnt.

#### **Schritt 2: Einfache Datenanalyse (26.01. – 05.02.):**
- **Werkzeug:** Nutzt Python mit `pandas` und `matplotlib`, um erste Grafiken zu erstellen:
  - Preisentwicklung über die Zeit.
  - Berechnung einfacher Kennzahlen wie Durchschnitt oder Volatilität.
- **Code-Beispiel:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    data['Close'].plot()
    plt.title("Preisentwicklung")
    plt.show()
    ```

#### **Schritt 3: Technisches Setup (06.02. – 15.02.):**
- **Ziel:** RL-Umgebung aufsetzen und einen simplen Trading-Agenten mit einer Dummy-Strategie erstellen.
- **Konkret:**
  - Installiert Python-Bibliotheken:
    ```bash
    pip install stable-baselines3 gym
    ```
  - Erstellt ein einfaches RL-Szenario mit `gym`:
    ```python
    from stable_baselines3 import PPO
    import gym
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    ```

---

### **Phase 3: Prototyp entwickeln (15.02.2025 – 15.03.2025)**

#### **Schritt 1: RL-Agent für Trading (15.02. – 05.03.):**
- **Vorgehen:**
  - Nutzt Stable-Baselines3 und erstellt ein eigenes Environment basierend auf euren Finanzdaten.
  - **Tutorial nutzen:** Stable-Baselines3-Dokumentation.
  - Modifiziert die Rewards:
    - +1 für Gewinne, -1 für Verluste.
- **Beispiel:**
    ```python
    class TradingEnv(gym.Env):
        def __init__(self, data):
            self.data = data
        def step(self, action):
            # Logik für Gewinne/Verluste hier
            pass
    ```

#### **Schritt 2: Backtesting einbauen (06.03. – 15.03.):**
- **Werkzeug:** `Backtrader` oder einfache eigene Logik.
- **Ziel:** Performance eures RL-Agenten auf ungesehenen Daten testen.

---

### **Phase 4: Dokumentation und Optimierung (15.03.2025 – 24.03.2025)**

#### **Bericht schreiben (15.03. – 20.03.):**
- Kapitel nach Aufgaben aufteilen (siehe Inhaltsverzeichnis).
- Wissenschaftliche Artikel recherchieren:
  - Plattformen: Google Scholar, IEEE Xplore.
- Zitierstandard festlegen (APA, IEEE).

#### **Feinschliff und Abgabe (21.03. – 24.03.):**
- Gemeinsame Korrektur.
- Formatierung überprüfen.

---

### **Zusatzempfehlungen:**
- Setzt wöchentliche Mini-Ziele und haltet euch daran.
- Nutzt Tools wie GitHub für Versionskontrolle und Aufgabenmanagement.
- Lernt iterativ: Beginnt mit einfachen Beispielen und steigert euch schrittweise. 