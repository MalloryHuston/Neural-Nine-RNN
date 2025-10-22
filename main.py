#!/usr/bin/env python3
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


# Ask the user for the ticker symbol
company = input("Enter the ticker symbol of the company: ")

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

plt.figure(figsize=(12, 6))
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
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()

# Format x-axis to display months and years
plt.gcf().autofmt_xdate()
plt.show()

# Predict the future price
days_ahead = int(
    input("Enter the number of days from now to predict the stock price: "))

# Start from the last known data
last_known_data = model_inputs[-prediction_days:]

# Predict the price for the specified number of days ahead
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

    # Append the new prediction to last_known_data and remove the
    # first entry to maintain the window size
    last_known_data = np.append(
        last_known_data, scaler.transform(prediction),
        axis=0
    )
    last_known_data = last_known_data[1:]


# The final prediction is the stock price for
# the specified number of days ahead
print(
    f"The predicted {company} share price in {days_ahead} days will "
    f"most likely be: {predictions[-1]}"
)
