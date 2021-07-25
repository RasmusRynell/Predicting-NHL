import source.bet_parsers.parse_betsson as betsson
import source.bet_parsers.parse_betway as betway
import source.bet_parsers.parse_unibet as unibet
import source.bet_parsers.parse_bet365 as bet365
import source.bet_parsers.parse_wh as wh
import source.bet_parsers.parse_ss as ss
from source.db_models.bets_models import *
from source.db_models.nhl_models import *
from datetime import date, datetime, timedelta
from sqlalchemy import func, and_, or_, not_


def add_file_to_db(file, nhl_session, bets_session):
    end = file.split(".")[-1]

    bets = []
    if end == "bet365":
        bets = bet365.read_file(file)
    elif end == "betsson":
        bets = betsson.read_file(file)
    elif end == "betway":
        bets = betway.read_file(file)
    elif end == "unibet":
        bets = unibet.read_file(file)
    elif end == "wh":
        bets = wh.read_file(file)
    elif end == "ss":
        bets = ss.read_file(file)

    for bet in bets:
        add_bet_to_db(bet, nhl_session, bets_session)


def get_game_pk(home_team, away_team, time, nhl_session):
    delta_12_h = timedelta(hours=12)
    time = time + timedelta(hours=24)
    lower_time = time - delta_12_h
    upper_time = time + delta_12_h
    ids = nhl_session.query(Game).filter(and_(Game.homeTeamId == home_team, \
                                              Game.awayTeamId == away_team, \
                                              Game.gameDate > lower_time, \
                                              Game.gameDate < upper_time \
                                              )).all()
    if len(ids) == 0:
        print("No games found...")
        return -1
    elif len(ids) > 1:
        print("more than 1 game found...")
        return -1
    else:
        return ids[0].gamePk



def add_bet_to_db(bet, nhl_session, bets_session):

    player_id = get_player_id_from_name(bet[1], nhl_session)
    home_team = get_team_id_from_name(bet[2], nhl_session)
    away_team = get_team_id_from_name(bet[3], nhl_session)
    time = datetime.strptime(bet[0], '%Y-%m-%d')

    game_pk = get_game_pk(home_team, away_team, time, nhl_session)

    if game_pk == -1:
        print("Game was not found, not adding bets {}".format(bet))
    elif not bets_session.query(Bet).filter(and_(Bet.gamePk == game_pk,
                                              Bet.playerId == player_id, \
                                              func.lower(Bet.site) == func.lower(bet[4]), \
                                              func.lower(Bet.overUnder) == func.lower(str(bet[7]).replace(",", ".")) \
                                                  )).all():
        try:
            new_bet = Bet()
            new_bet.playerId = player_id
            new_bet.homeTeamId = home_team
            new_bet.awayTeamId = away_team
            new_bet.dateTime = time
            new_bet.site = bet[4]
            new_bet.gamePk = game_pk
            new_bet.overUnder = str(bet[7]).replace(",", ".")
            new_bet.oddsOver = str(bet[5]).replace(",", ".")
            new_bet.oddsUnder = str(bet[6]).replace(",", ".")

            bets_session.add(new_bet)
        except:
            print("Something went wrong, did not add {}".format(bet))
    else:
        print("Bet already exists")
        return


def get_player_id_from_name(name, nhl_session):
    ids = nhl_session.query(PersonNicknames).filter(func.lower(PersonNicknames.nickname) == (func.lower(name))).all()
    if len(ids) == 0:
        ids = nhl_session.query(Person).filter(func.lower(Person.fullName).contains(func.lower(name))).all()

    if len(ids) > 1:
        print("More than one id for:")
        print(name)
        raise "More than one id..."

    if len(ids) == 0:
        print("Cant find a player with that name for:")
        print(name)
        raise "Cant find a player with that name..."

    return ids[0].id




def get_team_id_from_name(name, nhl_session):
    ids = nhl_session.query(TeamNicknames).filter(func.lower(TeamNicknames.nickname) == (func.lower(name))).all()
    if len(ids) == 0:
        ids = nhl_session.query(Team).filter(func.lower(Team.name).contains(func.lower(name))).all()

    if len(ids) > 1:
        print("More than one id for:")
        print(name)
        raise "More than one id..."

    if len(ids) == 0:
        print("Cant find a player with that name for:")
        print(name)
        raise "Cant find a player with that name..."

    return ids[0].id
