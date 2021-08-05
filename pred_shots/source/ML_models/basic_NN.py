import pandas as pd
import numpy as np

from tqdm import tqdm

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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

    return model


def get_model_acc(X_train, X_test, X_pred, y_train, y_test):
    # Create inverse of y's (0 becomes 1, 1 becomes 0)
    y_train = hot_encode(X_train, y_train)
    y_test = hot_encode(X_test, y_test)


    over_all = {
    "metrics": {
        "loss": [],
        "acc": [],
        "AUC": [],
        "number_of_bets_tested": X_test.shape[0]
        },
        "predictions": []
    }
    epochs = 150
    batch_size = 10
    times_to_run = 1
    print(f"epochs: {epochs}, batch_size: {batch_size}, times_to_run: {times_to_run}")
    for _ in tqdm(range(times_to_run)):
        # create model
        model = generate_model(len(X_train[0]), len(y_train[0]))

        # Fit the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate the model
        test_scores = model.evaluate(X_test, y_test, verbose=0)

        # Add to over_all
        over_all["metrics"]["loss"].append(test_scores[0])
        over_all["metrics"]["acc"].append(test_scores[1])
        over_all["metrics"]["AUC"].append(test_scores[2])

        # Add predictions
        model.fit(X_test, y_test, epochs=epochs, batch_size=batch_size, verbose=0)
        over_all["predictions"].append(model.predict(X_pred))
        #print("Predictions: ", model.predict(X_pred))

        # Reset the whole model
        tf.keras.backend.clear_session()


    results = {
        "loss": sum(over_all["metrics"]["loss"]) / len(over_all["metrics"]["loss"]),
        "acc": sum(over_all["metrics"]["acc"]) / len(over_all["metrics"]["acc"]),
        "AUC": sum(over_all["metrics"]["AUC"]) / len(over_all["metrics"]["AUC"]),
        "number_of_bets_tested": over_all["metrics"]["number_of_bets_tested"]
    }
    
    # Calculate average of predictions
    predictions = []
    for (index, preds) in enumerate(over_all['predictions']):
        if index == 0:
            for O_U_pred in preds:
                predictions.append(list(O_U_pred))
        else:
            for (i, O_U_pred) in enumerate(preds):
                for (j, value) in enumerate(O_U_pred):
                    predictions[i][j] += value

    for (i, O_U_pred) in enumerate(predictions):
        for (index, value) in enumerate(O_U_pred):
            predictions[i][index] = value / times_to_run

        for index in range(len(O_U_pred)):
            predictions[i][index] = predictions[i][index] / sum(predictions[i])
            predictions[i][index] = predictions[i][index] / sum(predictions[i])

    #print("Predictions: ", predictions)
    return results, predictions


def hot_encode(X, Y_in):
	Y = np.zeros((X.shape[0], 2))
	for i, shots in enumerate(Y_in):
		#under
		Y[i, 0] = int(shots == 0)
		#over
		Y[i, 1] = int(shots == 1)
	return Y


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)