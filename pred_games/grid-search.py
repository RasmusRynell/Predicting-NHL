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
from tqdm import tqdm
import time

NUM_OF_DENSE_ = [[256, 128, 64, 32]]#, [128, 64, 32, 16]]

DROPOUT_ = [0.4]#[0.4,0.6]

L2_ = [0.0]
L1_ = [0.0]
pca_comp = [50]#[0, 50, 100, 150, 200, 250]

epochs = 50
batch_size = 32

NAME_ = f"grid-search_"

RANDOM_SEED = 124
tf.random.set_seed(RANDOM_SEED)

def run_grid_search(X_train_, y_train_, X_test_, y_test_, loss, class_weight):
    for pca_value in pca_comp:
        if pca_value == 0:
            X_train = X_train_
            X_test = X_test_
        else:
            pca_ = PCA(n_components=pca_value)
            X_train = pca_.fit_transform(X_train_)
            X_test = pca_.transform(X_test_)
        for l1 in tqdm(L1_):
            for l2 in tqdm(L2_):
                for dropout in tqdm(DROPOUT_):
                    for NUM_OF_DENSE in tqdm(NUM_OF_DENSE_):
                        name = f"{NAME_}{int(time.time())}_l1={l1}_l2={l2}_dropout={dropout}_" + "".join([f"dense={dense}" for dense in NUM_OF_DENSE]) + f"_pca_comp={pca_value}" + f"_SEED={RANDOM_SEED}"
                        model = Sequential()
                        model.add(Dense(NUM_OF_DENSE[0], input_dim=X_test.shape[1], activation='relu', kernel_regularizer=regularizers.l1(l1)))

                        for i in range(1, len(NUM_OF_DENSE)):
                            model.add(Dense(NUM_OF_DENSE[i], activation='relu', kernel_regularizer=regularizers.l2(l2)))
                            model.add(Dropout(dropout))

                        model.add(Dense(y_test.shape[1], activation='softmax'))
                        opt = Adam(learning_rate=1e-2, decay=1e-4)
                        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

                        tensorboard = TensorBoard(log_dir=f'./logs/{name}')
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),\
                            class_weight=class_weight, shuffle=True, callbacks=[tensorboard], verbose=0)


if __name__ == "__main__":
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

    # Run grid search
    run_grid_search(X_train, y_train, X_test, y_test, 'categorical_crossentropy', class_weight)