from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import backend as K


def get_model(input_dim, output_dim, loss):
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model


def odds_loss(y_true, y_pred):
    """
    The function implements the custom loss function
    
    Inputs
    true : a vector of dimension batch_size, 7. A label encoded version of the output and the backp1_a and backp1_b
    pred : a vector of probabilities of dimension batch_size , 5.
    
    Returns 
    the loss value
    """
    win_home = y_true[:, 0:1]
    win_away = y_true[:, 1:2]
    draw = y_true[:, 2:3]
    no_bet = y_true[:, 3:4]
    home_odds = y_true[:, 4:5]
    away_odds = y_true[:, 5:6]
    draw_odds = y_true[:, 6:7]

    gain_loss_vector = K.concatenate([
        win_home * (home_odds - 1) + (1 - win_home) * -1,
        win_away * (away_odds - 1) + (1 - win_away) * -1,
        draw * (draw_odds - 1) + (1 - draw) * -1,
        no_bet
      ], axis=1)
    return -1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))