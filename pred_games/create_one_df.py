from tqdm import tqdm
import pandas as pd
import numpy as np
import json

def add_odds(season):
    stats_df = pd.read_csv(f'./stats/seasons/{season}/{season}-together.csv', sep=';')
    odds_df = pd.read_csv('./stats/odds/NHL_odds_and_results.csv', sep=';')


    # Turn "Date" column into datetime and remove 6 hours
    odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%Y-%m-%d %H:%M:%S')
    odds_df['Date'] = (odds_df['Date'] - pd.Timedelta(hours=6, minutes=0, seconds=0)).dt.date.astype(str)


    # Add odds_df to stats_df where "Date", "HomeTeam" and "AwayTeam" are all the same (Side note: this also removes games with no odds)
    df = stats_df.merge(odds_df, on=['Date', 'HomeTeam', 'AwayTeam'], how='inner')

    print(f"Missing {(stats_df.shape[0] - df.shape[0])} odds for season {season}")


    # save df
    df.to_csv(f'./stats/seasons/{season}/{season}_done.csv', sep=';', index=False)



if __name__ == '__main__':
    for i in tqdm(range(2010, 2021)):
        add_odds(str(i)+str(i+1))