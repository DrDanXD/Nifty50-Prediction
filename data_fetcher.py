import yfinance as yf

def fetch_nifty50_data(period="2y", interval="1d"):
    """
    Pulls historical Nifty 50 stock prices using yfinance.

    Args:
        period (str): Time range for the data (e.g., "1y", "2y", "5d").
        interval (str): Data granularity (e.g., "1d", "1h", "5m").

    Returns:
        pandas.DataFrame: Cleaned DataFrame containing only 'Close' prices.
    """
    nifty_symbol = "^NSEI"  # Symbol for Nifty 50 on Yahoo Finance
    data = yf.download(nifty_symbol, period=period, interval=interval)
    data.dropna(inplace=True)  # Remove any rows with missing values
    return data[['Close']]     # Return only the closing price
