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
        if(line.find(" - ") > 0):
            current_key = line
            matches[current_key] = []
        elif line != "":
            matches[current_key].append(line)
    res = []
    for match in matches:
        home_team = unidecode(match.split(" - ")[0])
        away_team = unidecode(match.split(" - ")[1])
        for i in range(0, len(matches[match]), 3):
            name_list = matches[match][i].split(" ")[:-4]
            first_name = name_list[0]
            last_name = name_list[1]
            player_name = first_name + " " + last_name
            player_target = matches[match][i+1].split(" ")[1][1:4]
            player_odds_O = matches[match][i+1].split(" ")[1].split(")")[1]
            player_odds_U = matches[match][i+2].split(" ")[1].split(")")[1]
            player_info = [date, player_name, home_team, away_team, betting_site, player_odds_O, player_odds_U, player_target]
            res.append(player_info)

    return res


#print(read_file("../../../saved_bets/2021-03-23.ss"))


