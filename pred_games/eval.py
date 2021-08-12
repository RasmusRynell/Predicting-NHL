import models as models
import helper as helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import RFE
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, concatenate, Lambda, BatchNormalization
from keras import Model
from tqdm import tqdm



tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)


### Gather and prepare data
dfs = []
for i in range(2010, 2021):
    dfs.append(pd.read_csv(f'./stats/seasons/{str(i) + str(i+1)}/{str(i) + str(i+1)}_done.csv', sep=';'))

# Append all df's in dfs to df
df = pd.concat(dfs)

df.drop(columns=['GameUrl','HomeScoreAfterOtAndSo', 'AwayScoreAfterOtAndSo', 'Date', 'HomeTeam', 'AwayTeam'], inplace=True)
# , 'OddsHome', 'OddsDraw', 'OddsAway'

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
y_train_ = pd.get_dummies(train_data['Result'])

y_train = pd.DataFrame(columns=['HOME', 'AWAY', 'DRAW', 'NO_BET', 'ODDS_HOME', 'ODDS_AWAY', 'ODDS_DRAW'])
y_train['HOME'] = (y_train_['HOME']).astype(float)
y_train['AWAY'] = (y_train_['AWAY']).astype(float)
y_train['DRAW'] = (y_train_['DRAW']).astype(float)
y_train["NO_BET"] = float(-1)
y_train["ODDS_HOME"] = (X_train['OddsHome']).astype(float)
y_train["ODDS_AWAY"] = (X_train['OddsAway']).astype(float)
y_train["ODDS_DRAW"] = (X_train['OddsDraw']).astype(float)



X_test = test_data.drop(columns=['Result'])
y_test_ = pd.get_dummies(test_data['Result'])

y_test = pd.DataFrame(columns=['HOME', 'AWAY', 'DRAW', 'NO_BET', 'ODDS_HOME', 'ODDS_AWAY', 'ODDS_DRAW'])
y_test['HOME'] = (y_test_['HOME']).astype(float)
y_test['AWAY'] = (y_test_['AWAY']).astype(float)
y_test['DRAW'] = (y_test_['DRAW']).astype(float)
y_test["NO_BET"] = float(-1)
y_test["ODDS_HOME"] = (X_test['OddsHome']).astype(float)
y_test["ODDS_AWAY"] = (X_test['OddsAway']).astype(float)
y_test["ODDS_DRAW"] = (X_test['OddsDraw']).astype(float)


### Preprocess here in order to not leak data in scalaing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### PCA
pca = PCA(n_components=250)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

relative = y_train["HOME"].sum()
class_weight = {
    0: relative / y_train['AWAY'].sum(),
    1: relative / y_train["DRAW"].sum(),
    2: relative / y_train["HOME"].sum(),
    3: 1,
    4: 1,
    5: 1
}



model = models.get_model(X_test.shape[1], y_test.shape[1]-3, models.odds_loss)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, batch_size=32, callbacks=[EarlyStopping(patience=25),ModelCheckpoint('odds_loss.hdf5',save_best_only=True)], shuffle=True)

print(f'Training Loss :{model.evaluate(X_train, y_train.values)}\nValidation Loss :{model.evaluate(X_test, y_test)}')

pred_ = model.predict(X_test)
pred = np.argmax(pred_, axis=1)

# Save pred to csv
df_pred = pd.DataFrame(columns=['HOME', 'AWAY', 'DRAW', 'NO_BET'], data=pred_)
svar = pd.concat([y_test, df_pred], axis=1)
svar.to_csv('./stats/pred.csv', sep=';', index=False)