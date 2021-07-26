import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def generate_model(input_dim, output_dim):
    # # create model
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))
    #model.add(Dense(output_dim, activation='softmax')) 
    # Compile model 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
    return model


def get_model_acc(X_train, X_test, X_pred, y_test, y_train):
    # Create inverse of y's (0 becomes 1, 1 becomes 0)
    y_train = hot_encode(X_train, y_train)
    y_test = hot_encode(X_test, y_test)

    # create model
    model = generate_model(len(X_train[0]), len(y_train[0]))

    # Fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)

    # Evaluate the model
    print(model.metrics_names)
    test_scores = model.evaluate(X_test, y_test, verbose=0)

    # X_train = np.concatenate((X_train, X_test))
    # y_train = np.concatenate((y_train, y_test))
    # model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
    
    # # Predict
    predictions = model.predict(X_pred)

    return test_scores, predictions


def hot_encode(X, Y_in):
	Y = np.zeros((X.shape[0], 2))
	for i, shots in enumerate(Y_in):
		#Under
		Y[i, 0] = int(shots == 0)
		#Over
		Y[i, 1] = int(shots == 1)
	return Y
