from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from tensorflow.keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

import models as models
import helper as helper

tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)

### Gather and prepare data
dfs = []
for i in range(2010, 2021):
    dfs.append(pd.read_csv(f'./stats/{str(i) + str(i+1)}_done.csv', sep=';'))

# Append all df's in dfs to df
df = pd.concat(dfs)

df.drop(columns=['GameUrl','HomeScoreAfterOtAndSo', 'AwayScoreAfterOtAndSo', 'Date', 'HomeTeam', 'AwayTeam', 'OddsHome', 'OddsDraw', 'OddsAway'], inplace=True)

# Add new column if row contains any NAN values
df['Contains_NANs'] = df.isnull().any(axis=1).astype(int)
# Set all nan values to 0
df.fillna(0, inplace=True)

# Save df to csv
df.to_csv('./stats/df.csv', sep=';', index=False)


### Set up the data
# Get first 80% of the data
train_data = df.iloc[:int(df.shape[0] * 0.8)]
test_data = df.iloc[int(df.shape[0] * 0.8):]

# construct train test X and y
X_train = train_data.drop(columns=['Result'])
y_train = pd.get_dummies(train_data['Result']).values
# Save df to csv
pd.get_dummies(train_data['Result']).to_csv('./stats/y_train.csv', sep=';', index=False)

X_test = test_data.drop(columns=['Result'])
y_test = pd.get_dummies(test_data['Result'])

### Preprocess here in order to not leak data in sclaing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



### Create and fit model
model = models.baseline_model(X_test.shape[1], len(y_test.columns.to_list()))

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=5, verbose=1, mode='auto', restore_best_weights=True)

model.fit(X_train, y_train, epochs=200, batch_size=X_test.shape[1], validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=2)




### Evaluate model
# Generate predictions for the test data
pred_ = model.predict(X_test)
pred = np.argmax(pred_, axis=1)

# Print score
y_compare = np.argmax(y_test.values,axis=1) 
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy on test data: {}".format(score))
print(f'ROC AUC: {metrics.roc_auc_score(y_compare, pred_, multi_class="ovr", average="weighted")}')

# Print confusion matrix
cm = metrics.confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
helper.plot_confusion_matrix(cm, y_test.columns.to_list())

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
helper.plot_confusion_matrix(cm_normalized, y_test.columns.to_list(), 
        title='Normalized confusion matrix')

plt.show()