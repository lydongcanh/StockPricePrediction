import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

time_steps = 60
training_file_name = "msn_2016_2019"
test_file_name = "msn_2020_2021"

training_file = pd.read_csv(f"../Data/{training_file_name}.csv")
test_file = pd.read_csv(f"../Data/{test_file_name}.csv")

training_data = training_file.iloc[:len(training_file), 1:2].values
test_data = test_file.iloc[:len(test_file), 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_data_scaled = sc.fit_transform(training_data)

# Creating a data structure with time-steps and 1 output
X_train = []
y_train = []
for i in range(time_steps, len(training_data)):
    X_train.append(training_data_scaled[i-time_steps:i, 0])
    y_train.append(training_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build model
model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer="adam", loss="mean_squared_error")

# Load trained modal if available.
if path.exists(f"{training_file_name}__{test_file_name}_model.h5"):
    model.load_weights(f"{training_file_name}__{test_file_name}_model.h5")
else:
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    model.save(f"{training_file_name}__{test_file_name}_model.h5")

# Prepare the test data

# Getting the predicted stock prices
dataset_train = training_file.iloc[:len(training_file), 1:2]
dataset_test = test_file.iloc[:len(test_file), 1:2]

dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_steps:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(time_steps, len(dataset_test) + time_steps):
    X_test.append(inputs[i-time_steps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make Predictions using the test set
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Prices")
plt.plot(dataset_test.values, color = "red", label = "Real Prices")
plt.title("Masan Stock Prices Prediction Using LSTM Model")
plt.xlabel("Time")
plt.ylabel("Prices")
plt.legend()
plt.show()