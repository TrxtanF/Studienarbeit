import yfinance as yf
import mplfinance as mpf
import pandas as pd

def plot_candlestick_chart(ticker, start_date, end_date):
    """
    Lädt die Tageswerte eines Wertpapiers herunter und zeigt einen Candlestick-Chart.

    :param ticker: Das Tickersymbol des Wertpapiers (z.B. "AAPL" für Apple).
    :param start_date: Startdatum im Format "YYYY-MM-DD".
    :param end_date: Enddatum im Format "YYYY-MM-DD".
    """
    # Daten von yfinance abrufen
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    if data.empty:
        print(f"Keine Daten für {ticker} im angegebenen Zeitraum gefunden.")
        return

    # Debugging: Zeige die ursprünglichen Daten an
    print("Ursprüngliche Daten:")
    print(data.head())
    print("\nSpaltennamen der ursprünglichen Daten:", list(data.columns))

    # MultiIndex-Spaltenstruktur auflösen
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]  # Nur die oberste Ebene der Spalten verwenden

    # Debugging: Zeige die aufgelösten Spaltennamen an
    print("\nSpaltennamen nach Auflösung:", list(data.columns))

    # Fehlende Werte und nicht-numerische Werte entfernen
    try:
        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")  # Nicht-numerische Werte zu NaN konvertieren
        data = data.dropna()  # Nach der Konvertierung erneut NaN-Werte entfernen
    except Exception as e:
        print(f"Fehler bei der Datenbereinigung: {e}")
        return

    # Prüfen, ob der Index vom Typ Datetime ist
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Debugging: Bereinigte Daten anzeigen
    print("\nBereinigte Daten:")
    print(data.head())

    # Candlestick-Chart zeichnen
    try:
        mpf.plot(
            data,
            type="candle",
            style="yahoo",
            title=f"Candlestick-Chart für {ticker}",
            ylabel="Preis",
            volume=True,
            figratio=(16, 9),
            figscale=1.2
        )
    except Exception as e:
        print(f"Fehler beim Plotten: {e}")

# Beispielaufruf
ticker_symbol = "BTC-USD"  # Ersetze mit dem gewünschten Tickersymbol
start = "2024-01-01"
end = "2024-11-20"

plot_candlestick_chart(ticker_symbol, start, end)
