import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import models as models
import helper as helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import RFE
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, concatenate, Lambda, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorboard.plugins.hparams import api as hp
from tqdm import tqdm
import time


LOGDIR = f"logs/"

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_L2 = hp.HParam('l2', hp.RealInterval(0.001, 0.01))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LOSS = hp.HParam('loss', hp.Discrete(['categorical_crossentropy']))

METRIC_ACCURACY = 'accuracy'





### Gather and prepare data
dfs = []
for i in range(2010, 2021):
    dfs.append(pd.read_csv(f'./stats/seasons/{str(i) + str(i+1)}/{str(i) + str(i+1)}_done.csv', sep=';'))

# Append all df's in dfs to df
df = pd.concat(dfs)

df.drop(columns=['GameUrl','HomeScoreAfterOtAndSo', 'AwayScoreAfterOtAndSo', 'Date', 'HomeTeam', 'AwayTeam'], inplace=True)
#df.drop(columns=df.columns.difference(['OddsHome', 'OddsDraw', 'OddsAway', 'Result']), inplace=True)

# Add new column if row contains any NAN values
df['Contains_NANs'] = df.isnull().any(axis=1).astype(int)
# Set all nan values to 0
df.fillna(0, inplace=True)


### Set up the data
# Get first 80% of the data
train_data = df.iloc[:int(df.shape[0] * 0.8)]
test_data = df.iloc[int(df.shape[0] * 0.8):]

# Save test_data
test_data.to_csv('./stats/test_data.csv', sep=';', index=False)

# construct train test X and y
X_train = train_data.drop(columns=['Result'])
y_train = pd.get_dummies(train_data['Result'])


X_test = test_data.drop(columns=['Result'])
y_test = pd.get_dummies(test_data['Result'])


### Preprocess here in order to not leak data in scalaing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Calculate the class weights
relative = y_train["HOME"].sum()
class_weight = {
    0: relative / y_train['AWAY'].sum(),
    1: relative / y_train["DRAW"].sum(),
    2: relative / y_train["HOME"].sum()
}




with tf.summary.create_file_writer(f'logs/hparam_tuning_{int(time.time())}').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_LOSS, HP_L2],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

def train_test_model(hparams, X_train, y_train, X_test, y_test):
    model = tf.keras.models.Sequential([
        Dense(hparams[HP_NUM_UNITS], input_dim=X_test.shape[1], activation='relu', kernel_regularizer=regularizers.l2(hparams[HP_L2])),
        Dropout(0.1),
        Dense(hparams[HP_NUM_UNITS]/2, activation='relu', kernel_regularizer=regularizers.l2(hparams[HP_L2])),
        Dropout(0.1),
        Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss=hparams[HP_LOSS],
        metrics=['accuracy'],
    )

    model.fit(X_train, y_train, epochs=50, class_weight=class_weight, verbose=2)
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy

def run(run_dir, hparams, X_train, y_train, X_test, y_test):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams, X_train, y_train, X_test, y_test)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)



session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            for loss in HP_LOSS.domain.values:
                for l2 in (HP_L2.domain.min_value, HP_L2.domain.max_value):
                    hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                        HP_LOSS: loss,
                        HP_L2: l2,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, hparams, X_train, y_train, X_test, y_test)
                    session_num += 1