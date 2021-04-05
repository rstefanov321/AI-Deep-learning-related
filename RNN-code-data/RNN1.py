# Recurrent Neural Network

import tensorflow as tf
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set 

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
#putting 1:2 => np array of 1 column

#Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


#Creating a data structure with 60 time steps and 1 output
#input
X_train = []
#output
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)    
    
#Reshaping the data
# the unit - indicators
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()
    
#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding a SECOND LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


#Adding the output layer
regressor.add(Dense(units=1)) #units corresponds to the dimension of the output layer. 

# Compiling the RNN
regressor.compile(optimizer = "adam", loss = "mean_squared_error")

#Fitting the RNN to the Training set 
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3 = Making the predictions and visualising the results. 

#Getting the real stock price of 2017

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0) 
#for vertical concatenation axis = 0

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values 
#.values to make a np array, otherwise it's a dataframe

inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

#input
X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

#3d Structure
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#predict the results
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = "red", label = "Real Google stock price in the first month of Jan 2017")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google stock price in the first month of Jan 2017")
plt.title("Google Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()









    
