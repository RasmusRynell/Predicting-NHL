from tqdm import tqdm
import pandas as pd
import numpy as np
import json

stats_df = pd.read_csv('./stats/2021-done.csv', sep=';')
odds_df = pd.read_csv('./stats/NHL_odds_and_results.csv', sep=';')


# Turn "Date" column into datetime and remove 6 hours
odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%Y-%m-%d %H:%M:%S')
odds_df['Date'] = (odds_df['Date'] - pd.Timedelta(hours=6, minutes=0, seconds=0)).dt.date.astype(str)


# Add odds_df to stats_df where "Date", "HomeTeam" and "AwayTeam" are all the same (Side note: this also removes games with no odds)
df = stats_df.merge(odds_df, on=['Date', 'HomeTeam', 'AwayTeam'], how='inner')


# Create two groups, the home teams and the away teams
is_home_group = df.groupby('IsHome')

# Get both groups
home_teams = is_home_group.get_group(True).reset_index()
away_teams = is_home_group.get_group(False).reset_index()

home_teams['AwayTeam_PDO'] = away_teams['PDO']
home_teams['AwayTeam_PDO_rates'] = away_teams['PDO_rates']
home_teams['AwayTeam_PDO_5v5'] = away_teams['PDO_5v5']
home_teams['AwayTeam_PDO_5v5_rates'] = away_teams['PDO_5v5_rates']

df = home_teams


# save df
df.to_csv('./stats/done.csv', sep=';', index=False)