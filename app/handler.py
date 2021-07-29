import sys
import glob
import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql.expression import true
from source.db_models.bets_models import *
from source.db_models.nhl_models import *
from source.bets_handler import *
from source.nhl_handler import *
from source.nhl_gen import *
from source.predict import *
from source.evaluate_bets import *
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
            for site_name, bets in game['odds'].items():
                for O_U in bets.keys():
                    if O_U not in list(over_under_games.keys()):
                        over_under_games[O_U] = []
                    over_under_games[O_U].append(game_id)

        for target, games in tqdm(over_under_games.items()):
            
            # Remove duplicates from games
            games = list(set(games))

            predictions = None
            if check_if_predictions_exists(player_id, target):
                # Load predictions dict from file
                predictions = load_predictions(player_id, target)
            else:
                # Load data for player
                data = get_data_from_file(player_id)

                predictions = predict_game(data, games, target, player_id)

                if predictions is None:
                    print(f"No prediction for {player_id} {target}")
                else:
                    # Save dict of predictions to file
                    with open("./external/predictions/temp/" + str(player_id) + "_" + str(target) + "_preds.json", 'w') as fp:
                        json.dump(predictions, fp)

            if predictions != None:
                for game_id, bets in predictions.items():
                    org_bets[player_id][str(int(float(game_id)))]['data'][target] = bets

            else:
                print(f"Skipped {player_id} {target}")

    return org_bets


def check_if_predictions_exists(player_id, target):
    oldpwd=os.getcwd()
    os.chdir("./external/predictions/temp/")
    for file in glob.glob("*"):
        file_name = file.split(".json")[0]
        if file_name != "test":
            file_player_id, file_target, _ = file_name.split("_")
            if str(player_id) == str(file_player_id) and str(target) in file_target:
                os.chdir(oldpwd)
                return True
    os.chdir(oldpwd)
    return False

def load_predictions(player_id, target):
    print(player_id, target)
    with open("./external/predictions/temp/" + str(player_id) + "_" + str(target) + "_preds.json", 'r') as fp:
        return json.load(fp)


def evaluate_bets(bets):
    calculate_roi_with_unit_size(bets)
    calculate_roi_with_kelly(bets)


def get_bets():
    bet_session = Session_bets()

    res = get_all_bets(bet_session)

    bet_session.close()

    return res