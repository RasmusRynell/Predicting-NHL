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

    os.chdir("./external/saved_bets")
    if not file:
        for file in tqdm(glob.glob("*")):
            add_file_to_db(file, nhl_session, bet_session)
    else:
        add_file_to_db(file, nhl_session, bet_session)

    bet_session.commit()
    bet_session.close()
    nhl_session.close()


def generate_csv(player_id):
    global Session_nhl
    nhl_session = Session_nhl()

    generate_data_for(player_id, nhl_session, 5, "all", f"./external/csvs/{player_id}.csv")

    nhl_session.close()