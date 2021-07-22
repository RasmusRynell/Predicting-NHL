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
from source.eval import *
from source.predict import *
from tqdm import tqdm
from datetime import date, datetime, timedelta
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

    data, file_name = generate_data_for(player_id, nhl_session, 5, "all")

    nhl_session.close()
    return (data, file_name)


def save_csv(df, path):
    df.to_csv(f"./external/csvs/data_csvs/{path}", sep=';', encoding='utf-8', index=False)


def evaluate_setup(config):
    return run_eval_pipeline(config)




# A function that check if data of player contains data for game
def get_data_from_file(player_id, game_id, date):
    date = datetime.strptime(date, '%Y-%m-%d')
    print(date)
    # Loop through all files in folder
    oldpwd=os.getcwd()
    os.chdir("./external/csvs/data_csvs/")
    for file in glob.glob("*"):
        file_name = file.split(".")[0]
        file_player_id, file_date = file_name.split("_")
        file_date = datetime.strptime(file_date, '%Y-%m-%d-%H-%M-%S')
        
        if str(player_id) == str(file_player_id) and date <= file_date:
            os.chdir(oldpwd)
            return pd.read_csv("./external/csvs/data_csvs/" + file, sep=';', encoding='utf-8')
        
    os.chdir(oldpwd)
    print("No file sufficient found")
    return False


def predict_games(games):
    for game in games:
        print(f"Predicting player: {game['player_id']} in game {game['game_id']}")

        # Todo: Check if game is already predicted (If not, predict else use old prediction) (maybe add some sort of id to know if its used same type of prediction config)
        
        # Todo: Generate data if doesn't exist
        data = get_data_from_file(game['player_id'], game['game_id'], game['date'])

        # Todo: Predict game

        # Read config from file
        config = None
        with open(game['config']) as f:
            config = json.loads(f.read())

        pred_df = predict_game(data, config)

        # Todo: Save prediction