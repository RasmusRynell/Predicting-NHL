
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
    matches= {}
    current_key = ""
    for line in lines:
        line = line.replace("\n","")
        if(line.find(" @ ") > 0):
            current_key = line
            matches[current_key] = []
        elif line != "":
            matches[current_key].append(line)
    res = []

    for match, value in matches.items():
        home_team = unidecode(match.split(" @ ")[1])
        away_team = unidecode(match.split(" @ ")[0])
        for i in range(0, len(value), 4):
            info = matches[match][i:(i+4)]
            if(len(info) > 1):
                player_name = info[0].split(", ")[0].split(" ")
                player_name_first = player_name[0].lower()
                player_name_last = player_name[1].lower()
                player_name = unidecode(player_name_first+" "+player_name_last)
                player_target = info[0].split(" ")[-1]
                player_odds_O = info[1]
                player_odds_U = info[3]
                player_info = [date, player_name, home_team, away_team, betting_site, player_odds_O, player_odds_U, player_target]
                res.append(player_info)
    return res