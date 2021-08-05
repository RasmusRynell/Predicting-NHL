from sqlalchemy import func, and_, or_, not_
from source.db_models.nhl_models import *
import traceback
import gevent.monkey
gevent.monkey.patch_all(thread=False, select=False)
import requests
import grequests
import datetime
import json

def convert_string_to_time(time_sec):
    arr = time_sec.split(":")
    h = 0
    m = int(arr[0])
    s = int(arr[1])
    while m > 59:
        h += 1
        m -= 60
    return datetime.time(h, m, s)


def add_game_to_db(session, game):
    try:
        new_game = Game()

        new_game.gamePk = game["gamePk"]
        new_game.gameType = game["gameType"] if "gameType" in game else None
        new_game.season = game["season"] if "season" in game else None
        new_game.gameDate = datetime.datetime.strptime(game["gameDate"], "%Y-%m-%dT%H:%M:%SZ") if "gameDate" in game else None
        new_game.abstractGameState = game["status"]["abstractGameState"] if "status" in game and "abstractGameState" in game["status"] else None
        new_game.codedGameState = game["status"]["codedGameState"] if "status" in game and "codedGameState" in game["status"] else None
        new_game.detailedState = game["status"]["detailedState"] if "status" in game and "detailedState" in game["status"] else None
        new_game.statusCode = game["status"]["statusCode"] if "status" in game and "statusCode" in game["status"] else None
        new_game.startTimeTBD = game["status"]["startTimeTBD"] if "status" in game and "startTimeTBD" in game["status"] else None
        new_game.homeTeamId = game["teams"]["home"]["team"]["id"]
        new_game.awayTeamId = game["teams"]["away"]["team"]["id"]

        session.add(new_game)
    except Exception as e:
        print(e)
        traceback.print_exc()

def hook_factory(*factory_args, **factory_kwargs):
        def add_game_stats_to_db(res, *args, **kwargs):
            try:
                info = factory_kwargs["info"]
                session = info["session"]

                res = res.json()

                for home_or_away in ("home", "away"):
                    # Add team stats
                    rev_home_away = "home" if home_or_away == "away" else "away"
                    team_info = res["teams"][home_or_away]
                    rev_team_info = res["teams"][rev_home_away]

                    new_team_stats = TeamStats()

                    new_team_stats.gamePk = info["gamePk"]
                    new_team_stats.teamId = team_info["team"]["id"]
                    new_team_stats.isHome = (home_or_away == "home")
                    if "teamSkaterStats" in team_info["teamStats"]:
                        res_team_stats = team_info["teamStats"]["teamSkaterStats"]
                        rev_res_team_stats = rev_team_info["teamStats"]["teamSkaterStats"]

                        new_team_stats.goals = res_team_stats["goals"]
                        new_team_stats.pim = res_team_stats["pim"]
                        new_team_stats.shots = res_team_stats["shots"]
                        new_team_stats.powerPlayPercentage = res_team_stats["powerPlayPercentage"]
                        new_team_stats.powerPlayGoals = res_team_stats["powerPlayGoals"]
                        new_team_stats.powerPlayOpportunities = res_team_stats["powerPlayOpportunities"]
                        new_team_stats.faceOffWinPercentage = res_team_stats["faceOffWinPercentage"]
                        new_team_stats.blocked = res_team_stats["blocked"]
                        new_team_stats.takeaways = res_team_stats["takeaways"]
                        new_team_stats.giveaways = res_team_stats["giveaways"]
                        new_team_stats.hits = res_team_stats["hits"]

                        new_team_stats.goalsAgainst = rev_res_team_stats["goals"]
                        new_team_stats.pimAgainst = rev_res_team_stats["pim"]
                        new_team_stats.shotsAgainst = rev_res_team_stats["shots"]
                        new_team_stats.powerPlayPercentageAgainst = rev_res_team_stats["powerPlayPercentage"]
                        new_team_stats.powerPlayGoalsAgainst = rev_res_team_stats["powerPlayGoals"]
                        new_team_stats.powerPlayOpportunitiesAgainst = rev_res_team_stats["powerPlayOpportunities"]
                        new_team_stats.faceOffWinPercentageAgainst = rev_res_team_stats["faceOffWinPercentage"]
                        new_team_stats.blockedAgainst = rev_res_team_stats["blocked"]
                        new_team_stats.takeawaysAgainst = rev_res_team_stats["takeaways"]
                        new_team_stats.giveawaysAgainst = rev_res_team_stats["giveaways"]
                        new_team_stats.hitsAgainst = rev_res_team_stats["hits"]

                    new_team_stats.wins = info["stats"][team_info["team"]["id"]]["wins"]
                    new_team_stats.losses = info["stats"][team_info["team"]["id"]]["losses"]
                    new_team_stats.ot = info["stats"][team_info["team"]["id"]]["ot"]
                    new_team_stats.leagueRecordType = info["stats"][team_info["team"]["id"]]["type"]
                    new_team_stats.score = info["stats"][team_info["team"]["id"]]["score"]

                    session.add(new_team_stats)

                    # Add all players stats
                    for player_id in team_info["players"]:
                        player_info = team_info["players"][player_id]

                        if player_info["position"]["code"] == "G":
                            if "goalieStats" in player_info["stats"]:
                                res_goalie_stats = player_info["stats"]["goalieStats"]

                                new_goalie_stats = GoalieStats()

                                new_goalie_stats.playerId = player_info["person"]["id"]
                                new_goalie_stats.gamePk = info["gamePk"]
                                new_goalie_stats.position = player_info["position"]["code"]
                                new_goalie_stats.team = team_info["team"]["id"]

                                new_goalie_stats.timeOnIce = convert_string_to_time(res_goalie_stats["timeOnIce"])
                                new_goalie_stats.assists = res_goalie_stats["assists"]
                                new_goalie_stats.goals = res_goalie_stats["goals"]
                                new_goalie_stats.pim = res_goalie_stats["pim"]
                                new_goalie_stats.shots = res_goalie_stats["shots"]
                                new_goalie_stats.saves = res_goalie_stats["saves"]
                                new_goalie_stats.powerPlaySaves = res_goalie_stats["powerPlaySaves"]
                                new_goalie_stats.shortHandedSaves = res_goalie_stats["shortHandedSaves"]
                                new_goalie_stats.evenSaves = res_goalie_stats["evenSaves"]
                                new_goalie_stats.shortHandedShotsAgainst = res_goalie_stats["shortHandedShotsAgainst"]
                                new_goalie_stats.evenShotsAgainst = res_goalie_stats["evenShotsAgainst"]
                                new_goalie_stats.powerPlayShotsAgainst = res_goalie_stats["powerPlayShotsAgainst"]
                                new_goalie_stats.decision = res_goalie_stats["decision"] if "decision" in res_goalie_stats else None
                                new_goalie_stats.savePercentage = res_goalie_stats["savePercentage"] if "savePercentage" in res_goalie_stats else None
                                new_goalie_stats.powerPlaySavePercentage = res_goalie_stats["powerPlaySavePercentage"] if "powerPlaySavePercentage" in res_goalie_stats else None
                                new_goalie_stats.evenStrengthSavePercentage = res_goalie_stats["evenStrengthSavePercentage"] if "evenStrengthSavePercentage" in res_goalie_stats else None

                                session.add(new_goalie_stats)

                        else:
                            if "skaterStats" in player_info["stats"]:
                                res_skater_stats = player_info["stats"]["skaterStats"]

                                new_skater_stats = SkaterStats()
                                
                                new_skater_stats.playerId = player_info["person"]["id"]
                                new_skater_stats.gamePk = info["gamePk"]
                                new_skater_stats.position = player_info["position"]["code"]
                                new_skater_stats.team = team_info["team"]["id"]

                                new_skater_stats.timeOnIce = convert_string_to_time(res_skater_stats["timeOnIce"])
                                new_skater_stats.assists = res_skater_stats["assists"]
                                new_skater_stats.goals = res_skater_stats["goals"]
                                new_skater_stats.shots = res_skater_stats["shots"]
                                new_skater_stats.hits = res_skater_stats["hits"]
                                new_skater_stats.powerPlayGoals = res_skater_stats["powerPlayGoals"]
                                new_skater_stats.powerPlayAssists = res_skater_stats["powerPlayAssists"]
                                new_skater_stats.penaltyMinutes = res_skater_stats["penaltyMinutes"]
                                new_skater_stats.faceOffWins = res_skater_stats["faceOffWins"]
                                new_skater_stats.faceoffTaken = res_skater_stats["faceoffTaken"]
                                new_skater_stats.takeaways = res_skater_stats["takeaways"]
                                new_skater_stats.giveaways = res_skater_stats["giveaways"]
                                new_skater_stats.shortHandedGoals = res_skater_stats["shortHandedGoals"]
                                new_skater_stats.shortHandedAssists = res_skater_stats["shortHandedAssists"]
                                new_skater_stats.blocked = res_skater_stats["blocked"]
                                new_skater_stats.plusMinus = res_skater_stats["plusMinus"]
                                new_skater_stats.evenTimeOnIce = convert_string_to_time(res_skater_stats["evenTimeOnIce"])
                                new_skater_stats.powerPlayTimeOnIce = convert_string_to_time(res_skater_stats["powerPlayTimeOnIce"])
                                new_skater_stats.shortHandedTimeOnIce = convert_string_to_time(res_skater_stats["shortHandedTimeOnIce"])

                                session.add(new_skater_stats)

            except Exception as e:
                print(e)
                print(info["gamePk"])
                traceback.print_exc()

        return add_game_stats_to_db

def fill_all_games_from_season(session, season):

    res = requests.get('https://statsapi.web.nhl.com/api/v1/schedule?season={}'.format(season))
    res = res.json()

    base = "https://statsapi.web.nhl.com/api/v1/game/"
    urls = []

    games_that_dont_have_to_be_updated = {}
    games = session.query(Game).filter(Game.statusCode == 7)
    for game in games:
        games_that_dont_have_to_be_updated[game.gamePk] = game

    if "dates" in res:
        for date in res["dates"]:
            for game in date["games"]:
                if game["gameType"] == "R" or game["gameType"] == "P":
                    if game["gamePk"] not in games_that_dont_have_to_be_updated.keys():
                        if session.query(Game).filter(Game.gamePk == game["gamePk"]).first():
                            remove_gamePk(session, game["gamePk"])

                        add_game_to_db(session, game)
                        game_pk = game["gamePk"]
                        urls.append((base + str(game_pk) + "/boxscore", { "session": session, "gamePk": str(game_pk), "stats" : {game["teams"]["home"]["team"]["id"]: {"wins": game["teams"]["home"]["leagueRecord"]["wins"],
                                                                                                                        "losses": game["teams"]["home"]["leagueRecord"]["losses"],
                                                                                                                        "ot": game["teams"]["home"]["leagueRecord"]["ot"] if "ot" in game["teams"]["home"]["leagueRecord"] else "",
                                                                                                                        "type": game["teams"]["home"]["leagueRecord"]["type"],
                                                                                                                        "score": game["teams"]["home"]["score"]},
                                                                                    game["teams"]["away"]["team"]["id"]: {"wins": game["teams"]["away"]["leagueRecord"]["wins"],
                                                                                                                        "losses": game["teams"]["away"]["leagueRecord"]["losses"],
                                                                                                                        "ot": game["teams"]["away"]["leagueRecord"]["ot"] if "ot" in game["teams"]["away"]["leagueRecord"] else "",
                                                                                                                        "type": game["teams"]["away"]["leagueRecord"]["type"],
                                                                                                                        "score": game["teams"]["away"]["score"]}}}))

    rs = (grequests.get(u[0], hooks={'response': [hook_factory(info=u[1])]}) for u in urls)
    responses = grequests.map(rs)


def fill_teams_and_persons(session):
    res = requests.get('https://statsapi.web.nhl.com/api/v1/teams?expand=team.roster')
    res = res.json()

    all_players_in_db = [player.id for player in session.query(Person).all()]

    for team in res["teams"]:
        if not session.query(Team).filter(Team.id == team["id"]).first():
            new_team = Team()

            new_team.id = team["id"]
            new_team.name = team["name"]
            new_team.teamName = team["teamName"]

            session.add(new_team)

        if "roster" in team:
            for player in team["roster"]["roster"]:

                # end me :()
                if session.query(Person).filter(Person.id == player["person"]["id"]).first():
                    session.query(Person).filter(Person.id == player["person"]["id"]).update({
                        "fullName": player["person"]["fullName"],
                        "positionCode": player["position"]["code"],
                        "updated":datetime.datetime.utcnow()
                    })

                else:
                    new_person = Person()

                    new_person.id = player["person"]["id"]
                    new_person.fullName = player["person"]["fullName"]
                    new_person.positionCode = player["position"]["code"]

                    session.add(new_person)


def remove_gamePk(session, gamePk):
    curr_game = session.query(Game).filter(Game.gamePk == gamePk).first()

    all_goalie_stats = session.query(GoalieStats).filter(GoalieStats.gamePk == curr_game.gamePk)
    for stats in all_goalie_stats:
        session.delete(stats)

    all_skater_stats = session.query(SkaterStats).filter(SkaterStats.gamePk == curr_game.gamePk)
    for stats in all_skater_stats:
        session.delete(stats)

    all_team_stats = session.query(TeamStats).filter(TeamStats.gamePk == curr_game.gamePk)
    for stats in all_team_stats:
        session.delete(stats)

    session.delete(curr_game)

    # commit session because we need this to happend before we try to add again
    session.commit()


def add_person_nickname(nickname, name, nhl_session):
    name = name.replace("\"", "")

    ids = []

    if name.isnumeric():
        ids = nhl_session.query(Person).filter(Person.id == int(name)).all()
    else:
        ids = nhl_session.query(Person).filter(func.lower(
            Person.fullName).contains(func.lower(name))).all()

    if len(ids) > 1:
        print("More than one id for:")
        print(name)
        return
        print(nhl_session.query(Person).filter(
            func.lower(Person.fullName).contains(func.lower(name))))
        raise "More than one id..."

    if len(ids) == 0:
        print("Cant find a player with that name for:")
        print(name)
        return
        print(nhl_session.query(Person).filter(
            func.lower(Person.fullName).contains(func.lower(name))))
        raise "Cant find a player with that name..."

    if not nhl_session.query(PersonNicknames).filter(and_(PersonNicknames.id == ids[0].id, func.lower(PersonNicknames.nickname) == func.lower(nickname))).first():
        new_nickname = PersonNicknames()

        new_nickname.id = ids[0].id
        new_nickname.nickname = nickname

        nhl_session.add(new_nickname)


def add_team_nickname(nickname, name, nhl_session):
    name = name.replace("\"", "")
    ids = nhl_session.query(Team).filter(func.lower(
        Team.name).contains(func.lower(name))).all()

    if len(ids) > 1:
        print("More than one id for:")
        print(name)
        return
        raise "More than one id..."

    if len(ids) == 0:
        print("Cant find a team with that name for:")
        print(name)
        print(nhl_session.query(Team).filter(func.lower(Team.name).contains(func.lower(name))))
        return
        raise "Cant find a team with that name..."

    if not nhl_session.query(TeamNicknames).filter(and_(TeamNicknames.id == ids[0].id, func.lower(TeamNicknames.nickname) == func.lower(nickname))).first():
        new_nickname = TeamNicknames()

        new_nickname.id = ids[0].id
        new_nickname.nickname = nickname

        nhl_session.add(new_nickname)
