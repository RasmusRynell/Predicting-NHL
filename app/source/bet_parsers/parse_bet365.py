
import json
from datetime import datetime as dt
from datetime import timedelta
from unidecode import unidecode

def read_file(file):
    date = file.split("/")[-1].split(".")[0]
    betting_site = file.split("/")[-1].split(".")[1]

    try:
        with open(file, "r", encoding='utf-8') as reader:
            lines = reader.readlines()
    except Exception as e:
        return []

    matches = {}
    current_key = ""
    for line in lines:
        line = line.replace("\n","")
        if(line.find(" @ ") > 0):
            current_key = line
            matches[current_key] = []
        elif len(line) > 2:
            matches[current_key].append(line)

    games_player = {}
    for match in matches:
        games_player[match] = []
        for i in range(1, len(matches[match]),2):
            if(unidecode(matches[match][i].lower()) == "over"):
                break
            else:
                games_player[match].append(matches[match][i])
    games_over_data = {}
    for match in matches:
        game = []
        start_saving = False
        games_over_data[match] = []
        for i in range(1, len(matches[match])):
            if(start_saving):
                if(matches[match][i].lower() == "under"):
                    break
                else:
                    games_over_data[match].append(matches[match][i])
            else:
                start_saving = unidecode(matches[match][i].lower()) == "over"
    games_under_data = {}
    game = []
    for match in matches:
        start_saving = False
        games_under_data[match] = []
        for i in range(1, len(matches[match])):
            if(start_saving):
                if(matches[match][i].find(" @ ") > 0):
                    break
                else:
                    games_under_data[match].append(matches[match][i])
            else:
                start_saving = matches[match][i] == "Under"
    res = []
    for match in games_player:   
        home_team = match.split(" @ ")[1]
        away_team = match.split(" @ ")[0]
        for i in range(len(games_player[match])):
            player_name = unidecode(games_player[match][i].lower())
            player_target = games_over_data[match][2*i]
            player_odds_O = games_over_data[match][2*i+1]
            player_odds_U = games_under_data[match][2*i+1]
            player_info = [date, player_name, home_team, away_team, betting_site, player_odds_O, player_odds_U, player_target]
            res.append(player_info)

    return res