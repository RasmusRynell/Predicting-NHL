from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import os
import glob


def predict_game(data, game_ids, target):
   cleaned_data = clean_data(data.copy())

   #X_train, y_train, X_test, y_test = 

   return {}


def clean_data(data):
   # Set the date to a datetime object and set it as the index
   data['date'] = pd.to_datetime(data['date'])
   data.set_index('date', inplace=True)

   print(data)
   return data
