import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Schritt 1: Erstellen von Beispieldaten (Schlusskurse 1. und 15. eines monats der letzten 4 jahre )
daten = {
    'Tage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
             57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
             75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
             93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
             109, 110, 111, 112, 113, 114],
    'Kurs': [8818.3, 9381.6, 9907.7, 8540.0, 5366.3, 6638.5, 6629.1, 8821.6, 9318.0, 
             10189.3, 9425.4, 9229.9, 9198.7, 11803.1, 11845.3, 11914.9, 10785.3, 10620.5, 
             11503.0, 13759.4, 15953.0, 18770.7, 19434.9, 29359.9, 36845.8, 33515.7, 
             47936.3, 49595.5, 55791.3, 58718.3, 63216.0, 57807.1, 46708.8, 36687.6, 
             40156.1, 33543.6, 31840.5, 39878.3, 46991.3, 48819.4, 48130.6, 48146.0, 
             61672.5, 60915.3, 63597.9, 57210.3, 48871.5, 47738.0, 43097.0, 38709.7, 
             44544.4, 44420.3, 39285.7, 46297.0, 40560.0, 38461.0, 31308.7, 29798.5, 
             22577.9, 19262.9, 20825.1, 23271.2, 24101.7, 20126.1, 19701.7, 19311.9, 
             19068.7, 20483.5, 16895.1, 16972.0, 17356.1, 16618.4, 20879.8, 23725.6, 
             24327.9, 23642.2, 24282.7, 28456.1, 30299.6, 28077.6, 27183.9, 26819.0, 
             25591.3, 30586.8, 30291.4, 29712.2, 29195.3, 25803.2, 26601.0, 27974.5, 
             27161.2, 35423.8, 37874.9, 38688.2, 41929.0, 44183.4, 42510.7, 43081.4, 
             51901.3, 62397.7, 69463.7, 69664.4, 63411.9, 58331.2, 66225.1, 67760.8, 
             66223.0, 62890.1, 64782.4, 65372.9, 57534.6, 57315.7, 59138.5, 60835.5]
}
df = pd.DataFrame(daten)

# Schritt 2: Unabhängige und abhängige Variablen definieren
X = df[['Tage']]  # Unabhängige Variable (Tage)
y = df['Kurs']    # Abhängige Variable (Aktienkurs)

# Schritt 3: Daten in Trainings- und Testdatensätze aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 4: Lineare Regression trainieren
modell = LinearRegression()
modell.fit(X_train, y_train)

# Schritt 5: Vorhersagen treffen
y_vorhersage = modell.predict(X_test)

# Schritt 6: Ergebnis visualisieren
plt.scatter(X, y, color='blue', label='Tatsächliche Kurse')
plt.plot(X, modell.predict(X), color='red', label='Vorhergesagte Kurse')
plt.xlabel('Tage')
plt.ylabel('Kurs')
plt.title('Aktienkurs Prognose mit Linearer Regression')
plt.legend()
plt.show()

# Vorhersage anzeigen
print(f"Tatsächliche Kurse: {y_test.values}")
print(f"Vorhergesagte Kurse: {y_vorhersage}")

# test
