import sys
import glob
import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from source.db_models.bets_models import *
from source.db_models.nhl_models import *
from source.bets_handler import *
from source.nhl_handler import *
from source.nhl_gen import *
from source.predict import *
from tqdm import tqdm
from datetime import date, datetime, timedelta
import json
import csv


# Global db's
# Create engines
engine_bets = create_engine('sqlite:///external/databases/bets.db', echo=False, future=False)
engine_nhl = create_engine('sqlite:///external/databases/testing.db', echo=False, future=False)

# Bind tables
Base_bets.metadata.create_all(bind=engine_bets)
Base_nhl.metadata.create_all(bind=engine_nhl)

# Create session factory
Session_bets = sessionmaker(bind=engine_bets)
Session_nhl = sessionmaker(bind=engine_nhl)








# Update nhl db (session to nhl db) (add if doesn't exists)
def update_nhl_db(season=None):
    global Session_nhl
    nhl_session = Session_nhl()

    if not season:
        fill_teams_and_persons(nhl_session)
        for season in tqdm(range(2010, 2021)):
            fill_all_games_from_season(nhl_session, str(season) + str(season+1))
    else:
        fill_all_games_from_season(season)

    nhl_session.commit()
    nhl_session.close()


# Add nicknames to nhl_db (all) (session to nhl db)
def add_nicknames_nhl_db():
    global Session_nhl
    nhl_session = Session_nhl()
    with open("./external/nicknames/player_nicknames.csv", encoding='utf-8') as f:
        datareader = csv.reader(f)
        for row in datareader:
            add_person_nickname(row[0], row[1], nhl_session)

    with open("./external/nicknames/team_nicknames.csv", encoding='utf-8') as f:
        datareader = csv.reader(f)
        for row in datareader:
            add_team_nickname(row[0], row[1], nhl_session)

    nhl_session.commit()
    nhl_session.close()


# Add bets (all) (session to nhl and bet_session db)
def update_bets_db(file = None):
    global Session_bets
    global Session_nhl
    bet_session = Session_bets()
    nhl_session = Session_nhl()

    oldpwd=os.getcwd()
    os.chdir("./external/saved_bets")
    if not file:
        for file in tqdm(glob.glob("*")):
            add_file_to_db(file, nhl_session, bet_session)
    else:
        add_file_to_db(file, nhl_session, bet_session)
    os.chdir(oldpwd)

    bet_session.commit()
    bet_session.close()
    nhl_session.close()


def generate_csv(player_id):
    global Session_nhl
    nhl_session = Session_nhl()

    data = generate_data_for(player_id, nhl_session, 5, "all")

    nhl_session.close()
    return data


def save_csv(df, path):
    df.to_csv(f"./external/player_data/{path}", sep=';', encoding='utf-8', index=False)


# A function that check if data of player contains data for game
def get_data_from_file(player_id):
    # Loop through all files in folder
    oldpwd=os.getcwd()
    os.chdir("./external/player_data/")
    for file in glob.glob("*"):
        file_name = file.split(".")[0]
        
        if str(player_id) == str(file_name):
            os.chdir(oldpwd)
            return pd.read_csv("./external/player_data/" + file, sep=';', encoding='utf-8')
        
    os.chdir(oldpwd)
    print("No sufficient file found, generating one")

    data = generate_csv(player_id)
    save_csv(data, str(player_id) + ".csv")

    return data


def predict_games(org_bets):
    for player_id, values in tqdm(org_bets.items()):
        over_under_games = {}
        for game_id, game in values.items():
            for sites in game['odds']:
                for site_name, bets in sites.items():
                    for O_U in bets.keys():
                        if O_U not in list(over_under_games.keys()):
                            over_under_games[O_U] = []
                        over_under_games[O_U].append(game_id)

        for target, games in over_under_games.items():
            # Load data for player
            data = get_data_from_file(player_id)

            predictions = predict_game(data, target, games)

            for game_id, bets in predictions.items():
                org_bets[player_id][game_id]['predictions'][target] = bets

                






        # for target in tqdm(game["target"]):
        #     print(f"Predicting player: {game['player_id']} in game {game['game_ids']}")

        #     # Todo: Generate data if doesn't exist
        #     data = get_data_from_file(game['player_id'], game['date'])

        #     # Generate predictions
        #     predictions = predict_game(data, game['game_ids'], target)

        #     # Add predictions to results
        #     results.append({'player_id': game['player_id'],
        #                     'predictions': predictions
        #             })
        
    return org_bets