import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# Read json file
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return json.loads(data)

json_file = read_json('./stats/NHL_odds_and_results.json')

# Create empty df
df = pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'HomeScoreAfterOtAndSo', 'AwayScoreAfterOtAndSo', 'OddsHome', 'OddsAway', 'OddsDraw', 'Result', 'GameUrl'])

# Loop trough all season in json file
for season in tqdm(json_file['league']['seasons']):
    print(season['name'])
    for game in tqdm(season['games']):
        # Create empty df
        df_game = pd.DataFrame(
            {
                'Date': [game['game_datetime']],
                'HomeTeam': [game['team_home']],
                'AwayTeam':[game['team_away']],
                'HomeScoreAfterOtAndSo': [game['score_home']],
                'AwayScoreAfterOtAndSo': [game['score_away']],
                'OddsHome': [game['odds_home']],
                'OddsAway': [game['odds_away']],
                'OddsDraw': [game['odds_draw']],
                'Result': [game['outcome']],
                'GameUrl': [game['game_url']]
            }
        )
        # Add df_game as new row in df with all columns
        df = df.append(df_game, ignore_index=True)

# Reverse rows
df = df.iloc[::-1]

# Save df to csv
df.to_csv('./stats/NHL_odds_and_results.csv', index=False, sep=';')