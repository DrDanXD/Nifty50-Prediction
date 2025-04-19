from data_fetcher import fetch_nifty50_data
from predictor import prepare_data, build_lstm_model
from visualizer import plot_predictions

def main():
    # Step 1: Fetch historical Nifty 50 data
    print("📥 Fetching real-time Nifty 50 data...")
    price_data = fetch_nifty50_data()
    print("✅ Data fetched successfully!")

    # Step 2: Prepare it for model training
    print("🧪 Preparing data for LSTM model...")
    X, y, scaler = prepare_data(price_data)
    print(f"✅ Data prepared: {X.shape[0]} sequences created.")

    # Step 3: Build the LSTM model
    print("🏗️ Building the LSTM model...")
    model = build_lstm_model((X.shape[1], 1))

    # Step 4: Train the model
    print("📚 Training the model (this may take a few seconds)...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    print("✅ Model training complete!")

    # Step 5: Make predictions
    print("📈 Making predictions...")
    predictions = model.predict(X)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

    # Step 6: Plot the results
    print("🖼️ Plotting actual vs predicted prices...")
    plot_predictions(actual_prices, predicted_prices)

if __name__ == "__main__":
    main()
