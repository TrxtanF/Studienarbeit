# Benötigte Bibliotheken importieren
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# SCHRITT 1: BITCOIN-DATEN LADEN
# ================================
# Wir holen uns die Bitcoin-Daten der letzten 8 Jahre
daten = yf.download('BTC-USD', start='2014-09-17', end='2025-01-28')  # Bitcoin-Daten

# MultiIndex der Spalten entfernen (nur die erste Ebene behalten)
daten.columns = daten.columns.get_level_values(0)  # Entferne die Ticker-Ebene (z. B. 'BTC-USD')

# Überprüfen, ob die Daten korrekt geladen wurden
print(daten.head())
print(daten.columns)  # Zeigt die Spaltennamen

# Verwenden der relevanten Spalten: Close, High, Low, Open, Volume
features = daten[['Close', 'High', 'Low', 'Open', 'Volume']].values  # Nimm alle fünf Spalten

# SCHRITT 2: DATEN VORBEREITEN
# =============================
# Skalieren der Daten: Werte zwischen 0 und 1
scaler = MinMaxScaler(feature_range=(0, 1))  # Min-Max-Skalierung
features = scaler.fit_transform(features)  # Skalierung anwenden

# Eingabe- und Zielsequenzen erstellen
sequenz_länge = 30  # Verwenden der letzten 30 Tage, um die nächsten 7 Tage vorherzusagen
X = []  # Eingabedaten
y = []  # Zielvariable (nächste Woche)

# Sequenzen aus den Daten erstellen
for i in range(sequenz_länge, len(features) - 7):  # -7, da wir 7 Tage vorhersagen
    X.append(features[i-sequenz_länge:i])  # Letzte 30 Tage als Eingabe
    y.append(features[i:i+7, 0])  # Die nächsten 7 Tage-Kurse als Ziel (nur den Schlusskurs)

# Umwandlung in NumPy-Arrays
X = np.array(X)
y = np.array(y)

# Daten für LSTM umformen
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # 3D-Form: (Samples, Timesteps, Features)

# SCHRITT 3: NEURONALES NETZWERK ERSTELLEN
# =========================================
# Erstellen eines LSTM-Modells
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),  # Erste LSTM-Schicht
    LSTM(50),  # Zweite LSTM-Schicht
    Dense(7, activation='linear')  # Ausgabe: Vorhersage der nächsten 7 Tage-Kurse
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='mean_squared_error')  # Verlustfunktion für Regression

# SCHRITT 4: DAS MODELL TRAINIEREN
# =================================
# Trainieren des Modells
print("\nTraining des Modells gestartet...")
model.fit(X, y, epochs=20, batch_size=32)  # Trainiere für 20 Epochen mit einer Batch-Größe von 32


# SCHRITT 5: TESTEN UND ENTSCHEIDUNG
# ===================================
# Vorhersage für die nächste Woche machen (letzte 30 Tage als Eingabe verwenden)
letzte_30_tage = features[-sequenz_länge:]  # Die letzten 30 Tage der Daten
letzte_30_tage = letzte_30_tage.reshape((1, sequenz_länge, X.shape[2]))  # Umformen für das Modell

# Vorhersage der nächsten 7 Tage
vorhersage = model.predict(letzte_30_tage)

# Dummy-Spalten hinzufügen, um die Dimension für den Scaler nachzuahmen
dummy_features = np.zeros((vorhersage.shape[0] * vorhersage.shape[1], features.shape[1] - 1))
vorhersage_komplett = np.hstack((vorhersage.flatten().reshape(-1, 1), dummy_features))

# Rückskalieren der Vorhersage
vorhersage_rueckskaliert = scaler.inverse_transform(vorhersage_komplett)[:, 0]  # Nur 'Close' zurückgeben

# Letzter bekannter Schlusskurs
letzter_kurs = scaler.inverse_transform(features[-1].reshape(1, -1))[0][0]

# Durchschnitt der vorhergesagten Kurse berechnen
durchschnitt_vorhersage = np.mean(vorhersage_rueckskaliert)

# Entscheidung treffen basierend auf der Vorhersage
print("\nLetzter Schlusskurs:", letzter_kurs)
print("Durchschnittlicher Kurs der nächsten Woche:", durchschnitt_vorhersage)

if durchschnitt_vorhersage > letzter_kurs * 1.02:  # Mehr als 2% Anstieg
    print("Entscheidung: KAUFEN (Kurs steigt deutlich)")
elif durchschnitt_vorhersage < letzter_kurs * 0.98:  # Mehr als 2% Rückgang
    print("Entscheidung: VERKAUFEN (Kurs fällt deutlich)")
else:
    print("Entscheidung: HALTEN (keine große Änderung)")
