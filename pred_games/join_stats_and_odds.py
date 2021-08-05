from tqdm import tqdm
import pandas as pd
import numpy as np
import json

stats_df = pd.read_csv('./stats/2021-done.csv', sep=';')
odds_df = pd.read_csv('./stats/NHL_odds_and_results.csv', sep=';')


# turn "Date" column into datetime and remove 6 hours
odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%Y-%m-%d %H:%M:%S')
odds_df['Date'] = odds_df['Date'] - pd.Timedelta(hours=1, minutes=0, seconds=0)

# save df
odds_df.to_csv('./stats/NHL_odds_and_results_fixed.csv', sep=';', index=False)