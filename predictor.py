import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_data(price_data, window_size=60):
    """
    Scales and reshapes data into sequences that an LSTM can understand.

    Args:
        price_data (DataFrame): The 'Close' price column.
        window_size (int): How many past days to look at for prediction.

    Returns:
        tuple: Features (X), labels (y), and the scaler object.
    """
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(price_data)

    X = []
    y = []

    for i in range(window_size, len(scaled_prices)):
        X.append(scaled_prices[i - window_size:i])
        y.append(scaled_prices[i])

    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """
    Builds a simple LSTM model for time-series forecasting.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential()

    # First LSTM layer with output going to next LSTM
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Prevent overfitting

    # Second LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Final output layer
    model.add(Dense(1))  # Predicting just one value: the stock price

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
