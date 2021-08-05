import numpy as np
import pandas as pd
import re

df_all_str = pd.read_csv('./stats/2021-all_str.csv')
df_all_str_rates = pd.read_csv('./stats/2021-all_str_rates.csv')
df_5v5 = pd.read_csv('./stats/2021-5v5.csv')
df_5v5_rates = pd.read_csv('./stats/2021-5v5_rates.csv')

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
df['AwayTeam'] = df.index.str.split(' - ').str[1].str.split(', ').str[0]
df['HomeTeam'] = df.index.str.split(' - ').str[1].str.split(', ').str[1]

# Remove everything only after last " " in 'AwayTeam'
df['AwayTeam'] = df['AwayTeam'].str.replace(r'\s\d$', '', regex=True)
df['HomeTeam'] = df['HomeTeam'].str.replace(r'\s\d$', '', regex=True)

# If the value in 'Team' contains the value in 'HomeTeam'
df['IsHome'] = df.apply(lambda x: x.HomeTeam in x.Team, axis=1).astype(int)


# fill in '-' (NANs) with 0
df = df.apply(lambda x: x.replace(to_replace='-', value=0.0), axis=1)


# Save to csv
df.to_csv('./stats/2021-done.csv', sep=';', index=False)