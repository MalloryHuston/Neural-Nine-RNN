#!/usr/bin/env python3
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from pandas.tseries.offsets import BDay  # For business-day handling


# Ask the user to enter a company's ticker symbol of their choice
company = input("Enter your favorite company's ticker symbol: ").upper()

# Fetch earnings dates dynamically
ticker_obj = yf.Ticker(company)
try:
    earnings_dates_df = ticker_obj.get_earnings_dates(limit=8)
    # Convert to datetime dates
    earning_dates = pd.to_datetime(
        earnings_dates_df['Earnings Date']
    ).dt.date.tolist()
except Exception as e:
    earnings_dates = []
    print(f"WARNING: Could not fetch earnings dates for {company}: {e}")

start = dt.datetime(2016, 1, 1)
end = dt.datetime(2024, 1, 1)

data = yf.download(company, start=start, end=end)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Load test data
test_start = dt.datetime(2024, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[
    len(total_dataset) - len(test_data) - prediction_days:
].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Update x-axis to show dates
dates = test_data.index  # Get the actual dates for the x-axis

# Plot the test predictions
plt.figure(dpi=100, figsize=(14, 7))
plt.plot(
    dates,
    actual_prices,
    color="black",
    label=f"Actual {company} Price"
)
plt.plot(
    dates,
    predicted_prices,
    color='green',
    label=f"Predicted {company} Price"
)

# Add dynamic earnings date markers (if available)
for ed in earnings_dates:
    if pd.Timestamp(ed) in dates:
        plt.axvline(x=ed, color='red', linestyle='--', linewidth=1)
        plt.text(
            ed,
            plt.ylim()[1] * 0.95,
            "Earnings",
            rotation=90,
            verticalalignment='top',
            horitzonalalignment='right',
            fontsize=9,
            color='red'
        )

plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()

# Format x-axis to display months and years
plt.gcf().autofmt_xdate()
plt.show()

# Predict the future price
days_ahead = int(
    input("Enter the number of *business* days from now to predict the "
          "stock price (i.e., excluding weekends and US market holidays): ")
)

# Start from the last known data
last_known_data = model_inputs[-prediction_days:]

# Predict the price for the specified number of business days ahead
predictions = []
for _ in range(days_ahead):
    real_data = np.array([last_known_data])
    real_data = np.reshape(
        real_data,
        (real_data.shape[0], real_data.shape[1], 1)
    )
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    # Append the new prediction to predictions list
    predictions.append(prediction[0][0])

    # Append the new prediction to last_known_data for the next iteration
    last_known_data = np.append(
        last_known_data, scaler.transform(prediction),
        axis=0
    )
    # Remove the first entry to maintain the window size
    last_known_data = last_known_data[1:]

# Calculate the projected business date
future_date = pd.Timestamp.today() + BDay(days_ahead)

# The final prediction output
print(
    f"The predicted {company} share price on {future_date.date()} "
    f"({days_ahead} business days from now) will most likely be: "
    f"${predictions[-1]:.2f}"
)

# Create csv-files directory if it doesn't exist
output_dir = "csv-files"
os.makedirs(output_dir, exist_ok=True)

# Save historical and predicted data to CSV
historical_close = [
    round(float(c), 2)
    for c in pd.to_numeric(data['Close'], errors='coerce')
]
historical_df = pd.DataFrame({
    'Date': data.index.date,
    f'{company}_Historical_Close': historical_close
})

future_dates = [
    (pd.Timestamp.today() + BDay(i + 1)).date()
    for i in range(days_ahead)
]

predicted_df = pd.DataFrame({
    'Date': future_dates,
    f'{company}_Predicted_Close': [round(p, 2) for p in predictions]
})

# Merge and save
full_df = pd.concat([historical_df, predicted_df], ignore_index=True)
filename = os.path.join(
    output_dir,
    f"{company}_full_predictions_{pd.Timestamp.today().date()}.csv"
)
full_df.to_csv(filename, index=False)

print(f"\n✅ Historical and predicted prices saved to '{filename}'")

# For debugging purposes: print the last real_data inverse transformed
'''
print(np.round(scaler.inverse_transform(real_data[-1]), 2))
'''
