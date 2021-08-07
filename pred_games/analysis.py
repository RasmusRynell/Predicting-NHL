from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from tensorflow.keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)


dfs = []
for i in range(2010, 2021):
    dfs.append(pd.read_csv(f'./stats/{str(i) + str(i+1)}_done.csv', sep=';'))

# Append all df's in dfs to df
df = pd.DataFrame()
df = pd.concat(dfs)

df.drop(columns=['GameUrl','HomeScoreAfterOtAndSo', 'AwayScoreAfterOtAndSo', 'Date', 'HomeTeam', 'AwayTeam', 'OddsHome', 'OddsDraw', 'OddsAway'], inplace=True)

df.dropna(axis=0, inplace=True)


# construct train test split
# Get first 80% of the data
train_data = df.iloc[:int(df.shape[0] * 0.8)]
test_data = df.iloc[int(df.shape[0] * 0.8):]

X_train = train_data.drop(columns=['Result'])
y_train = pd.get_dummies(train_data['Result']).values
X_test = test_data.drop(columns=['Result'])
y_test = pd.get_dummies(test_data['Result']).values
products = pd.get_dummies(test_data['Result']).columns

# Min max scale X
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save df to csv
#df.to_csv('./stats/df.csv', sep=';', index=False)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    return model

####
model = baseline_model()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

model.fit(X_train, y_train, epochs=20, batch_size=50, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping])

# Generate predictions for the test data
pred = model.predict(X_test)
pred = np.argmax(pred,axis=1) 

# Print score
y_compare = np.argmax(y_test,axis=1) 
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

# Print confusion matrix
print(metrics.confusion_matrix(y_compare, pred))