from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


dfs = []
for i in range(2010, 2021):
    dfs.append(pd.read_csv(f'./stats/{str(i) + str(i+1)}_done.csv', sep=';'))

# Append all df's in dfs to df
df = pd.DataFrame()
df = pd.concat(dfs)

df.drop(columns=['GameUrl','HomeScoreAfterOtAndSo', 'AwayScoreAfterOtAndSo', 'Date', 'HomeTeam', 'AwayTeam', 'OddsHome', 'OddsDraw', 'OddsAway'], inplace=True)

df.dropna(axis=0, inplace=True)


# construct train test split
X = df.drop(columns=['Result'])
y = pd.get_dummies(df['Result'])

# Min max scale X
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Save df to csv
df.to_csv('./stats/df.csv', sep=';', index=False)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=432, activation='LeakyReLU'))
    model.add(Dense(250, input_dim=500, activation='LeakyReLU'))
    model.add(Dense(124, input_dim=250, activation='LeakyReLU'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=200, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))