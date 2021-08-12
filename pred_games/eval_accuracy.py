import models as models
import helper as helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import class_weight
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

### Create and fit model
model = models.get_model_acc(X_test.shape[1], len(y_test.columns.to_list()))

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping], class_weight=class_weight)
# X_test.shape[1]


### Evaluate model
# Generate predictions for the test data
pred_ = model.predict(X_test)
pred = np.argmax(pred_, axis=1)

# Print score
y_compare = np.argmax(y_test.values,axis=1) 
score = metrics.accuracy_score(y_compare, pred)
print("\nAccuracy on test data: {}".format(score))
print(f'ROC AUC on test data: {metrics.roc_auc_score(y_compare, pred_, multi_class="ovr", average="weighted")}')

# Print confusion matrix
cm = metrics.confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print('\nConfusion matrix, without normalization')
print(cm)
plt.figure()
helper.plot_confusion_matrix(cm, y_test.columns.to_list())

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('\nNormalized confusion matrix')
print(cm_normalized)
plt.figure()
helper.plot_confusion_matrix(cm_normalized, y_test.columns.to_list(), title='Normalized confusion matrix')

plt.show()