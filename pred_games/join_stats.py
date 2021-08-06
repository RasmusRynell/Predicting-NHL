import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm


def join_seasons_stats(season):
    try:
        df_all_str = pd.read_csv(f'./stats/seasons/{season}/{season}-all_str.csv')
        df_all_str_rates = pd.read_csv(f'./stats/seasons/{season}/{season}-all_str_rates.csv')
        df_5v5 = pd.read_csv(f'./stats/seasons/{season}/{season}-5v5.csv')
        df_5v5_rates = pd.read_csv(f'./stats/seasons/{season}/{season}-5v5_rates.csv')
    except FileNotFoundError:
        print(f'ERROR: No data for {season}')
        return

    # Set index to be "Game"
    df_all_str.set_index('Game', inplace=True)
    df_all_str_rates.set_index('Game', inplace=True)
    df_5v5.set_index('Game', inplace=True)
    df_5v5_rates.set_index('Game', inplace=True)

    # Sort on index
    df_all_str.sort_index(inplace=True)
    df_all_str_rates.sort_index(inplace=True)
    df_5v5.sort_index(inplace=True)
    df_5v5_rates.sort_index(inplace=True)

    # Drop columns without a name
    df_all_str.drop(df_all_str.columns[1], axis=1, inplace=True)
    df_all_str_rates.drop(df_all_str_rates.columns[1], axis=1, inplace=True)
    df_5v5.drop(df_5v5.columns[1], axis=1, inplace=True)
    df_5v5_rates.drop(df_5v5_rates.columns[1], axis=1, inplace=True)

    # Drop columns
    df_all_str.drop(['Attendance', 'TOI'], axis=1, inplace=True)
    df_all_str_rates.drop(['Attendance', 'TOI', 'Team'], axis=1, inplace=True)
    df_5v5.drop(['Attendance', 'TOI', 'Team'], axis=1, inplace=True)
    df_5v5_rates.drop(['Attendance', 'TOI', 'Team'], axis=1, inplace=True)

    # Append "_5v5" to all column names in df2
    df_all_str_rates.columns = [x + '_rates' for x in df_all_str_rates.columns]
    df_5v5.columns = [x + '_5v5' for x in df_5v5.columns]
    df_5v5_rates.columns = [x + '_5v5_rates' for x in df_5v5_rates.columns]

    # Concat on index
    df = pd.concat([df_all_str, df_all_str_rates, df_5v5, df_5v5_rates], axis=1)

    # Split "Game" into "Date" and teams
    df['Date'] = df.index.str.split(' - ').str[0]
    df['HomeTeam'] = df.index.str.split(' - ').str[1].str.split(', ').str[1]
    df['AwayTeam'] = df.index.str.split(' - ').str[1].str.split(', ').str[0]

    # Remove everything only after last " " in 'AwayTeam'
    df['HomeTeam'] = df['HomeTeam'].str.replace(r'\s\d\d*$', '', regex=True)
    df['AwayTeam'] = df['AwayTeam'].str.replace(r'\s\d\d*$', '', regex=True)

    # Read names from config json file
    with open('./configs/team_names.json') as json_data:
        team_names = json.load(json_data)

    df['HomeTeam'] = df.apply(lambda x: team_names[x.HomeTeam.lower()], axis=1)
    df['AwayTeam'] = df.apply(lambda x: team_names[x.AwayTeam.lower()], axis=1)


    # If the value in 'Team' contains the value in 'HomeTeam'
    df['IsHome'] = df.apply(lambda x: x.HomeTeam in x.Team, axis=1).astype(int)


    # fill in '-' (NANs) with 0
    df = df.apply(lambda x: x.replace(to_replace='-', value=0.0), axis=1)

    # Check if df.shape[0] is divisible by 2
    if df.shape[0] % 2 != 0:
        print(f'WARNING: df.shape[0] is not divisible by 2. {df.shape[0]}')

    cols_for_this_game = df.columns.to_list().copy()
    for col in [x for x in cols_for_this_game if x not in exclude_avr]:
        df[f'{col}_avr_10_games'] = df.groupby(['Team'])[col].transform(lambda x: x.rolling(10, min_periods=0).mean().shift(1))
        df[f'{col}_avr_season'] = df.groupby(['Team'])[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(1))

    data = []
    cols = ["Diff_" + x for x in df.columns]# + ["AwayTeam_" + x for x in df.columns]

    # Loop through df index with a step of 2
    for i in range(0, df.shape[0], 2):
        # Get the data from df at index i and i+1
        df1 = df.iloc[i]
        df2 = df.iloc[i+1]

        home_team = df1 if df1.IsHome == 1 else df2
        away_team = df1 if df1.IsHome == 0 else df2

        data.append([])
        #for team in [home_team, away_team]:
        for col in df.columns:
            if col not in exclude_avr:
                data[-1].append(float(home_team[col]) - float(away_team[col]))
            else:
                data[-1].append(home_team[col])

    df = pd.DataFrame(data=data, columns=cols)

    for key, index in rename_cols.items():
        df[index] = df[key]

    # Remove unwanted columns
    cols_for_this_game = ["Diff_" + x for x in cols_for_this_game]# + ["AwayTeam_" + x for x in cols_for_this_game]
    df.drop(cols_for_this_game, axis=1, inplace=True)

    print(df.shape)

    # Save to csv
    df.to_csv(f'./stats/seasons/{season}/{season}-together.csv', sep=';', index=False)

rename_cols = {
    "Diff_Date": "Date",
    "Diff_HomeTeam": "HomeTeam",
    "Diff_AwayTeam": "AwayTeam"
}

exclude_avr = [
    "Team",
    "Date",
    "AwayTeam",
    "HomeTeam",
    "IsHome"
]

if __name__ == "__main__":
    for i in tqdm(range(2010, 2021)):
        join_seasons_stats(str(i) + str(i+1))

