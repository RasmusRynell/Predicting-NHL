from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def baseline_model(input_dim, output_dim):
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=input_dim, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    return model