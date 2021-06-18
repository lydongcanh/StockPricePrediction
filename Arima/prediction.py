import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

training_file = pd.read_csv("../Data/msn_2016_2019.csv")
test_file = pd.read_csv("../Data/msn_2020_2021.csv")

training_data = training_file["Close"].values
test_data = test_file["Close"].values

history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

plt.plot(model_predictions, color="blue", label="Predicted Prices")
plt.plot(test_data, color="red", label="Real Prices")
plt.title("Masan Stock Prices Prediction Using Arima Model")
plt.xlabel("Time")
plt.ylabel("Prices")
plt.legend()
plt.show()
