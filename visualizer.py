import matplotlib.pyplot as plt

def plot_predictions(actual_prices, predicted_prices):
    """
    Plots actual vs predicted stock prices.

    Args:
        actual_prices (array-like): The true prices from the dataset.
        predicted_prices (array-like): The prices predicted by the model.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color='blue', label='📈 Actual Nifty 50 Price')
    plt.plot(predicted_prices, color='red', label='🤖 Predicted Price')
    plt.title("Nifty 50 Stock Price Prediction")
    plt.xlabel("Time (Days)")
    plt.ylabel("Nifty 50 Index Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
