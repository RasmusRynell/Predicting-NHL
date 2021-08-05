from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import os
import glob
from source.ML_models.basic_NN import get_model_acc as run_basic_NN


def predict_game(data, game_ids, target, player_id):
   # print(f"Game ID: {game_ids} Target: {target}")

   # Split the data to pred and train/test
   X_train, X_test, X_pred, y_train, y_test, y_pred = create_split(data.copy(), game_ids, target, player_id)
   if X_train is None:
      return None

   # Scale the data
   scaled_X_train, scaled_X_test, scaled_X_pred = scale_data(X_train, X_test, X_pred)

   # Create run and evaluate the model
   metrics, predictions = run_basic_NN(scaled_X_train, scaled_X_test, scaled_X_pred, y_train, y_test)

   for index, row in X_pred.iterrows():
      if row.values[0] is None:
         row.values[0] = -1

   return {
       str(int(float(row["gamePk"]))): {
           "pred": {
               "under": predictions[index][0],
               "over": predictions[index][1],
               "metrics": metrics,
           },
           "ans": int(y_pred.iloc[index].values[0]),
       }
       for index, row in X_pred.iterrows()
   }


def create_split(data, game_ids, target, player_id):
   target = "ans_O_" + str(target)

   # Check so that we have enough data to split
   if len(data) < 100:
      print(f"Not enough data to split {player_id}, {target}")
      return None, None, None, None, None, None

   # Remove data with gamePk less than 1
   data = data[data["gamePk"] > 20090000]

   # Remove date column
   data.drop(columns=['date'], inplace=True)

   # Set the date to a datetime object and set it as the index
   data.set_index('gamePk', inplace=True)

   # Create a new dataframe with only the columns we want
   relevant_features = get_correlating_features(data, target, 0.09)

   # add "gamePk", "date"
   relevant_features = relevant_features.append(pd.Series(["gamePk", "date"]))
   drop_this = [x for x in data.columns if x not in relevant_features]
   data.drop(columns=drop_this, axis=1, inplace=True)


   # Drop rows containing NAN
   data.dropna(inplace=True)

   # Drop targets not targetting
   default_targets = ["ans_O_1.5", "ans_O_2.5", "ans_O_3.5", "ans_O_4.5"]
   default_targets.remove(target)
   data.drop(columns=default_targets, axis=1, inplace=True)

   # Check if all ids are in the data if not, remove the ids
   for id in game_ids.copy():
      if int(id) not in data.index:
         print(f"{id} was not found in data for {player_id}")
         game_ids.remove(id)

   if len(game_ids) < 1:
      return None, None, None, None, None, None

   # Get indexes of every game in game_ids
   indexes = [data.index.get_loc(int(x)) for x in game_ids]

   # Split the data at the lowest index in the indexes list
   train_test_data = data.iloc[:min(indexes)].copy().reset_index()
   pred_data = data.iloc[min(indexes):].copy().reset_index()

   # Remove rows with gamePk not in game_ids (without chaning the order!)
   pred_data = pred_data[pred_data["gamePk"].astype(str).isin(game_ids)]

   # Create X and y for training and testing
   X_train, X_test, y_train, y_test = train_test_split(train_test_data.drop([target], axis=1), train_test_data[target], test_size=0.2, random_state=42, shuffle=False)

   # Create X and y for prediction and reset indexes
   X_pred = pred_data.drop([target], axis=1)
   y_pred = pred_data[target]
   X_pred = X_pred.reset_index().drop(columns=["index"])
   y_pred = y_pred.reset_index().drop(columns=["index"])

   #print(f"Cols: {X_train.columns}")
   print("Num of cols:", len(X_train.columns))

   return (X_train, X_test, X_pred, y_train, y_test, y_pred)


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