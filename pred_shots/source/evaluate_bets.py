from datetime import datetime
import json


def calculate_roi_with_unit_size(games):
    # sourcery skip: merge-else-if-into-elif
    days = convert_into_days(games.copy())

    cash = 100

    # Pritty print days
    #print(json.dumps(days, indent=4))


    # Loop trough all the days
    for date, game_information in days:
        for gamePk, player_games_information in game_information.items():
            for playerInfo in player_games_information:
                if playerInfo['data'] != {}:
                    best_odds = get_best_odds(playerInfo['odds'])

                    for O_U_bet, value in best_odds.items():
                        if O_U_bet in playerInfo['data']:
                            cash -= 1
                            if playerInfo['data'][O_U_bet]['pred']['over'] > 0.5:
                                if playerInfo['data'][O_U_bet]['ans'] == 1:
                                    cash += float(value['over']['value'])
                            else:
                                if playerInfo['data'][O_U_bet]['ans'] == 0:
                                    cash += float(value['under']['value'])
        if cash < 0:
            print("Error: negative cash, cant continue")
            return
        print(f"Cash on day {date}: {cash}")
    print("Final roi with unit bet:", cash/100)


def calculate_roi_with_kelly(games):
    days = convert_into_days(games.copy())

    cash = 100
    kelly_ratio = 0.10
    # Loop trough all the days
    for date, game_information in days:
        current_cash = cash
        for playerId, player_games_information in game_information.items():
            for game in player_games_information:
                if game['data'] != {}:
                    best_odds = get_best_odds(game['odds'])

                    for O_U_bet, value in best_odds.items():
                        if O_U_bet in game['data']:
                            # Calculate the kelly
                            b = float(value['over']['value']) - 1
                            p = float(game['data'][O_U_bet]['pred']['over'])
                            q = float(game['data'][O_U_bet]['pred']['under'])
                            kelly_over = (b*p-q) / b

                            b = float(value['under']['value']) - 1
                            p = float(game['data'][O_U_bet]['pred']['under'])
                            q = float(game['data'][O_U_bet]['pred']['over'])
                            kelly_under = (b*p-q) / b

                            if kelly_over > 0:
                                current_cash -= cash * kelly_over * kelly_ratio
                                if game['data'][O_U_bet]['ans'] == 1:
                                    current_cash += float(value['over']['value']) * kelly_over * kelly_ratio
                            if kelly_under > 0:
                                current_cash -= cash * kelly_under * kelly_ratio
                                if game['data'][O_U_bet]['ans'] == 0:
                                    current_cash += float(value['under']['value']) * kelly_under * kelly_ratio
                            print(current_cash)
        if current_cash < 0:
            print("Error: negative cash")
            #raise Exception("Negative cash")
        cash = current_cash
        print(f"Cash on day {date}: {cash}")

    print("Final roi kelly bet:", cash/100)



def convert_into_days(games):
    res = {}
    for playerId, information in games.items():
        for gamePk, gameInformation in information.items():
            if gameInformation['game_date'] not in res:
                res[gameInformation['game_date']] = {}
            if gamePk not in res[gameInformation['game_date']]:
                res[gameInformation['game_date']][gamePk] = []
            gameInformation['playerId'] = playerId
            res[gameInformation['game_date']][gamePk].append(gameInformation)

    # Sort dict by date
    res = sorted(res.items(), key = lambda x:datetime.strptime(x[0], '%Y-%m-%d'), reverse=False)

    return res

def get_best_odds(all_odds):
    best_odds = {}
    for site, info in all_odds.items():
        for odds_type, odds in info.items():
            if odds_type not in best_odds:
                best_odds[odds_type] = {"over": {"site": site, "value": odds["over"]}, "under": {"site": site, "value": odds["under"]}}
            else:
                if best_odds[odds_type]["over"]["value"] < odds["over"]:
                    best_odds[odds_type]["over"] = {"site": site, "value": odds["over"]}
                if best_odds[odds_type]["under"]["value"] < odds["under"]:
                    best_odds[odds_type]["under"] = {"site": site, "value": odds["under"]}
    # pritty print the best odds
    return best_odds