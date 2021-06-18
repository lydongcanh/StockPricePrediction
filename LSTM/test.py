import pandas as pd
from keras.layers import *

df=pd.read_csv("TSLA.csv")

training_set = df.iloc[1:2].values
test_set = df.iloc[:len(df), 1:2].values

# print("Training set")
# print(training_set)

print("Test set")
print(test_set)