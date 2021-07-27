from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import os
import glob
from source.ML_models.basic_NN import get_model_acc as run_basic_NN


def predict_game(data, game_ids, target):
   print(f"Game ID: {game_ids} Target: {target}")

   # Split the data to pred and train/test
   X_train, X_test, X_pred, y_train, y_test = create_split(data.copy(), game_ids, target)

   # Scale the data
   scaled_X_train, scaled_X_test, scaled_X_pred = scale_data(X_train, X_test, X_pred)

   # Create run and evaluate the model
   metrics, predictions = run_basic_NN(scaled_X_train, scaled_X_test, scaled_X_pred, y_train, y_test)

   # Build the results
   results = {
       row["gamePk"]: {"over": predictions[index][0], "under": predictions[index][1], "metrics": metrics} for index, row in X_pred.iterrows()
   }

   print("")
   print(metrics)
   print("-")
   print(predictions)

   return results


def create_split(data, game_ids, target):
   target = "O_" + str(target)

   # Remove data with gamePk less than 1
   data = data[data["gamePk"] > 20090000]

   # Remove date column
   data.drop(columns=['date'], inplace=True)

   # Set the date to a datetime object and set it as the index
   data.set_index('gamePk', inplace=True)

   # Create a new dataframe with only the columns we want
   relevant_features = get_correlating_features(data, target, 0.035)
   # add "gamePk", "date"
   relevant_features = relevant_features.append(pd.Series(["gamePk", "date"]))
   drop_this = [x for x in data.columns if x not in relevant_features]
   data.drop(columns=drop_this, axis=1, inplace=True)

   # Drop rows containing NAN
   data.dropna(inplace=True)

   # Drop targets not targetting
   default_targets = ["O_1.5", "O_2.5", "O_3.5", "O_4.5"]
   default_targets.remove(target)
   data.drop(columns=default_targets, axis=1, inplace=True)

   # Get indexes of every game in game_ids
   indexes = [data.index.get_loc(int(x)) for x in game_ids]

   # Split the data at the lowest index in the indexes list
   train_test_data = data.iloc[:min(indexes)].copy().reset_index()
   pred_data = data.iloc[min(indexes):].copy()

   # Remove rows with gamePk not in game_ids
   dont_remove_this = [pred_data.index.get_loc(int(x)) for x in game_ids]
   pred_data = pred_data.iloc[dont_remove_this].copy().reset_index()

   # Create X and y for training and testing
   X_train, X_test, y_train, y_test = train_test_split(train_test_data.drop([target], axis=1), train_test_data[target], test_size=0.2, random_state=42, shuffle=False)

   # Create X and y for prediction
   X_pred = pred_data.drop([target], axis=1)

   #print(f"Cols: {X_train.columns}")
   print(len(X_train.columns))

   return (X_train, X_test, X_pred, y_train, y_test)


def scale_data(X_train, X_test, X_pred):
   '''
   Scale the data
   '''
   scaler = StandardScaler()

   X_train = scaler.fit_transform(X_train.copy())
   X_test = scaler.transform(X_test.copy())
   X_pred = scaler.transform(X_pred.copy())
   
   return X_train, X_test, X_pred


def get_correlating_features(data, pred_this, threshold):
   '''
   Get the features that are correlated with the target
   '''
   cor = data.corr()
   cor_target = abs(cor[pred_this])
   return cor_target[cor_target >= threshold]