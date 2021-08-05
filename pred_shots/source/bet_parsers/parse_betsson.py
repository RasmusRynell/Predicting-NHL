
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

    tmp = []
    for line in lines:
        if len(line) > 1:
            line = line.replace("\n","")
            tmp.append(line)
    find = "–"
    res = []
    for i in range(len(tmp)):
        if tmp[i] == find:
            res.append(i)

    res.append(len(tmp)+1)
    for i in range(len(res)-1):
        game = tmp[res[i]-1:res[i+1]-1]
        key = str(game[0]) + " - " + str(game[2])
        matches[key] = game[3:]

    res = []
    for match in matches:
        home_team = unidecode(match.split(" - ")[0])
        away_team = unidecode(match.split(" - ")[1])
        for i in range(0, len(matches[match]), 4):
            info = matches[match][i:(i+4)]
            player_name = ""
            name = info[0].split(" ")[:-2]
            for i in range(len(name)):
                if i == len(name)-1:
                    player_name += name[i]
                else:
                    player_name += name[i] + " "
            
            player_name = unidecode(player_name)
            try:
                player_target = info[0].split(" ")[-1]
                player_odds_O = info[1]
                player_odds_U = info[3]
                player_info = [date, player_name, home_team, away_team, betting_site, player_odds_O, player_odds_U, player_target]
                res.append(player_info)
            except:
                print(player_name)
                raise("wrong")
    return res