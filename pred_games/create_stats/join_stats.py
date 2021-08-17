import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm
import itertools


def join_seasons_stats(season):
    dfs = []
    try:
        dfs.append(["", pd.read_csv(f'./stats/seasons/{season}/{season}-all_str.csv')])
        dfs.append(["_all_str_rates", pd.read_csv(f'./stats/seasons/{season}/{season}-all_str_rates.csv')])
        dfs.append(["_5v5", pd.read_csv(f'./stats/seasons/{season}/{season}-5v5.csv')])
        dfs.append(["_5v5_rates", pd.read_csv(f'./stats/seasons/{season}/{season}-5v5_rates.csv')])
        dfs.append(["_5v5_adj", pd.read_csv(f'./stats/seasons/{season}/{season}-5v5_adj.csv')])
        dfs.append(["_5v5_adj_rates", pd.read_csv(f'./stats/seasons/{season}/{season}-5v5_adj_rates.csv')])
    except FileNotFoundError:
        print(f'ERROR: No data for {season}')
        return

    # Set index to be "Game"
    for df in dfs:
        df[1].set_index('Game', inplace=True)

    # Sort on index
    for df in dfs:
        df[1].sort_index(inplace=True)

    # If any column name contains "/60" remove that
    for df in dfs:
        for col in df[1].columns:
            if "/60" in col:
                df[1].rename(columns={col: col.replace("/60", "")}, inplace=True)

    # Drop columns without a name and drop all columns that contain % in the name
    for df in dfs:
        df[1].drop([df[1].columns[1], "PDO"], axis=1, inplace=True)
        df[1] = df[1].drop(df[1].filter(regex='%').columns, axis=1)

    # Create new stats
    for df in dfs:
        df[1] = create_new_stats(df[1])

    # Drop columns
    for i ,df in enumerate(dfs):
        if i == 0:
            df[1].drop(['Attendance'], axis=1, inplace=True)
        else:
            df[1].drop(['Attendance', 'TOI', 'Team'], axis=1, inplace=True)

    # Append identifier to all column names in df
    for df in dfs:
        df[1].columns = [f'{col}{df[0]}' for col in df[1].columns]

    

    # Concat on index
    df = pd.concat([x[1] for x in dfs], axis=1)

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

    df['HomeTeam'] = df.apply(lambda x: team_names[x.HomeTeam.lower()].lower(), axis=1)
    df['AwayTeam'] = df.apply(lambda x: team_names[x.AwayTeam.lower()].lower(), axis=1)


    # If the value in 'Team' contains the value in 'HomeTeam'
    df['IsHome'] = df.apply(lambda x: x.HomeTeam.lower() in x.Team.lower(), axis=1).astype(int)


    # fill in '-' (NANs) with 0
    df = df.apply(lambda x: x.replace(to_replace='-', value=0.0), axis=1) # A bit Iffy

    adjust_stats(df, season)
    #calculate_stats(df, season)

def create_new_stats(df):
    #for col in df.columns:
        #print(col)
    df['P'] = df.apply(lambda x: 2 if x.GF > x.GA else int(x.TOI > 60) , axis=1)
    df['POAP'] = df['P']/2
    return df

def adjust_stats_run(df_in):
    # Season long averages for league
    home_games_stats = df_in[df_in.IsHome == 1]
    away_games_stats = df_in[df_in.IsHome == 0]

    league_season_avr_before_game = pd.DataFrame(index=df_in.index)
    home_league_season_avr_before_game = pd.DataFrame(index=df_in.index)
    away_league_season_avr_before_game = pd.DataFrame(index=df_in.index)
    for col in [x for x in df_in.columns if x not in exclude_avr]:
        league_season_avr_before_game[col] = df_in[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(1))
        home_league_season_avr_before_game[col] = home_games_stats[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(1))
        away_league_season_avr_before_game[col] = away_games_stats[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(1))

    # Season long averages for teams
    team_season_avr_before_game = pd.DataFrame(index=df_in.index)
    home_team_season_avr_before_game = pd.DataFrame(index=df_in.index)
    away_team_season_avr_before_game = pd.DataFrame(index=df_in.index)
    team_season_avr_before_game['Team'] = df_in['Team']
    home_team_season_avr_before_game['Team'] = home_games_stats['Team']
    away_team_season_avr_before_game['Team'] = away_games_stats['Team']
    for col in [x for x in df_in.columns if x not in exclude_avr]:
        team_season_avr_before_game[col] = df_in.groupby(['Team'])[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(2))
        home_team_season_avr_before_game[col] = home_games_stats.groupby(['Team'])[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(1))
        away_team_season_avr_before_game[col] = away_games_stats.groupby(['Team'])[col].transform(lambda x: x.rolling(1000, min_periods=1).mean().shift(1))

    # Remove duplicate rows
    league_season_avr_before_game = league_season_avr_before_game[~league_season_avr_before_game.index.duplicated(keep='first')]
    # (Multiply rows by 2 in order to make calculations easier in the future)
    league_season_avr_before_game = pd.DataFrame(np.repeat(league_season_avr_before_game.values, 2, axis=0), columns=league_season_avr_before_game.columns, index=df_in.index)

    home_league_season_avr_before_game = home_league_season_avr_before_game[~home_league_season_avr_before_game.index.duplicated(keep='first')]
    away_league_season_avr_before_game = away_league_season_avr_before_game[~away_league_season_avr_before_game.index.duplicated(keep='first')]
    home_team_season_avr_before_game = home_team_season_avr_before_game[~home_team_season_avr_before_game.index.duplicated(keep='first')]
    away_team_season_avr_before_game = away_team_season_avr_before_game[~away_team_season_avr_before_game.index.duplicated(keep='first')]

    avr_stats = pd.DataFrame(index=df_in.index)
    for col in exclude_avr:
        avr_stats[col] = df_in[col]

    # Loop trough all columns 2 at a time
    cols = [x for x in df_in.columns.tolist() if x not in exclude_avr]
    if len(cols) % 2 != 0:
        print("WARNING")
    for stat_for, stat_against in zip(cols[0::2], cols[1::2]):

        delta_for = (team_season_avr_before_game[stat_for] - league_season_avr_before_game[stat_for]).fillna(0)
        delta_against = (team_season_avr_before_game[stat_against] - league_season_avr_before_game[stat_against]).fillna(0)


        # Do some magic with indexes
        delta_for = delta_for.reset_index()
        delta_against = delta_against.reset_index()
        new_indexes = [i+1 if i % 2 == 0 else i-1 for i in range(len(delta_for))]
        delta_for.index = new_indexes
        delta_against.index = new_indexes
        delta_for.sort_index(inplace=True)
        delta_against.sort_index(inplace=True)
        delta_for.set_index('Game', inplace=True)
        delta_against.set_index('Game', inplace=True)



        avr_stats[stat_for] = df_in[stat_for] - delta_against[stat_against]
        avr_stats[stat_against] = df_in[stat_against] - delta_for[stat_for]

    return avr_stats


def adjust_stats(df_in, season):
    df = adjust_stats_run(df_in)
    for _ in tqdm(range(10)):
        df = adjust_stats_run(df)

    # save to csv
    df.to_csv(f'./{season}-adjusted-stats.csv', index=True, header=True, sep=';')

    return df


rename_cols = {
    "Diff_Date": "Date",
    "Diff_HomeTeam": "HomeTeam",
    "Diff_AwayTeam": "AwayTeam",
    "Diff_TOI": "TOI"
}

exclude_avr = [
    "Team",
    "Date",
    "AwayTeam",
    "HomeTeam",
    "IsHome",
    "TOI"
]

if __name__ == "__main__":
    #for i in tqdm(range(2010, 2021)):
        #join_seasons_stats(str(i) + str(i+1))
    join_seasons_stats("20102011")