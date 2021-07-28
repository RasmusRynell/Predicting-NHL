from source.db_models.bets_models import *
from source.db_models.nhl_models import *
from datetime import date, datetime, timedelta
from sqlalchemy import func, and_, or_, not_, asc, desc
import pandas as pd
import sqlalchemy
from sqlalchemy import select
from sqlalchemy.orm import aliased
from tqdm import tqdm
import csv

def generate_data_for(player_id, nhl_session, games_to_go_back, season):
    PlayerTeamStats = aliased(TeamStats)
    OppTeamStats = aliased(TeamStats)
    query = (
        select(SkaterStats, Game, PlayerTeamStats, OppTeamStats)
        .where(SkaterStats.playerId == player_id)
        .join(Game, SkaterStats.gamePk == Game.gamePk)
        .join(PlayerTeamStats, and_(SkaterStats.gamePk == PlayerTeamStats.gamePk, PlayerTeamStats.teamId == SkaterStats.team))
        .join(OppTeamStats, and_(SkaterStats.gamePk == OppTeamStats.gamePk, OppTeamStats.teamId != SkaterStats.team))
        .order_by(asc(Game.gameDate))
    )
    playerStatsForGames = pd.read_sql(query, nhl_session.bind)

    playerStatsForGames.columns = [u + "_Skater" for u in SkaterStats.__table__.columns.keys()]\
                                + [u + "_Game" for u in Game.__table__.columns.keys()] \
                                + [u + "_PlayerTeam" for u in PlayerTeamStats.__table__.columns.keys()] \
                                + [u + "_OppTeam" for u in OppTeamStats.__table__.columns.keys()]

    # df_total = pd.DataFrame()
    # if season == "all":
    #     for i in tqdm(range(2008, 2025)):
    #         season = str(i) + str(i+1)
    #         new_df = add_games_back(playerStatsForGames[playerStatsForGames.season_Game == str(season)], games_to_go_back)
    #         df_total = pd.concat([df_total, new_df])

    # else:
    #     df_total = add_games_back(playerStatsForGames[playerStatsForGames.season_Game == str(season)], games_to_go_back)

    playerStatsForGames["O_1.5"] = (playerStatsForGames["shots_Skater"] > 1.5).astype(int)
    playerStatsForGames["O_2.5"] = (playerStatsForGames["shots_Skater"] > 2.5).astype(int)
    playerStatsForGames["O_3.5"] = (playerStatsForGames["shots_Skater"] > 3.5).astype(int)
    playerStatsForGames["O_4.5"] = (playerStatsForGames["shots_Skater"] > 4.5).astype(int)

    df = clean_data(playerStatsForGames)
    df = generate_prediction_data(df)

    # One hot encode the categorical variables
    df = pd.get_dummies(df, columns=one_hot_cols)

    return df

def add_games_back(df, games_to_go_back):
    df_total = pd.DataFrame()
    for i in range(1, games_to_go_back + 1):
        dfc = df.copy()
        dfc = dfc.shift(periods=i)
        dfc.columns = [u + "_{}_games_back".format(i) for u in dfc.head()]
        df_total = pd.concat([df_total, dfc], axis=1)
    df = pd.concat([df, df_total], axis=1)
    return df


def generate_prediction_data(df):
    # Group the data by season
    df_grouped = df.groupby(['season'])

    # Create empty df to fill
    final_df = pd.DataFrame()

    # loop through the seasons
    for season, season_df in df_grouped:
        for stat in forbidden_stats:
            # Calculate the EMA for each season
            season_df[f'{stat}_ema_1_games_back'] = season_df[stat].ewm(span=1, min_periods=1).mean().shift(1).copy()#.fillna(0)
            season_df[f'{stat}_ema_3_season_back'] = season_df[stat].ewm(span=3, min_periods=1).mean().shift(1).copy()#.fillna(0)
            season_df[f'{stat}_ema_10_season_back'] = season_df[stat].ewm(span=10, min_periods=1).mean().shift(1).copy()#.fillna(0)
            season_df[f'{stat}_ema_1_season_back'] = season_df[stat].ewm(span=10000, min_periods=1).mean().shift(1).copy()#.fillna(0)

        # Save data to the final df
        final_df = pd.concat([final_df, season_df])
        
    for stat in forbidden_stats:
        final_df[f'{stat}_ema_carrier'] = final_df[stat].ewm(span=100000, min_periods=1).mean().shift(1)#.fillna(0)

    # Remove forbidden_stats from df
    final_df = final_df.drop(forbidden_stats, axis=1)

    return final_df


def clean_data(df):

    # Drop the unnecessary columns
    df.drop(remove_cols, axis=1, inplace=True)

    # Rename columns
    df.rename(columns=rename_cols, inplace=True)

    # In each column of "fill_with_zeros" where there is not a number, put a zero
    for col in fill_with_zeros:
        df[col].replace(r'^\s*$', 0, regex=True, inplace=True)

    # Convert timestamps to datetime objects
    for col in time_to_sec:
        df[col] = pd.to_timedelta(df[col].astype(str)).dt.total_seconds().astype(int)

    return df


remove_cols = [
   "added_Skater",
   "updated_Skater",
   "gamePk_Game",
   "abstractGameState_Game",
   "detailedState_Game",
   "statusCode_Game",
   "startTimeTBD_Game",
   "homeTeamId_Game",
   "awayTeamId_Game",
   "added_Game",
   "updated_Game",

   "gamePk_PlayerTeam",
   "teamId_PlayerTeam",
   "leagueRecordType_PlayerTeam",
   "added_PlayerTeam",
   "updated_PlayerTeam",

   "gamePk_OppTeam",
   "leagueRecordType_OppTeam",
   "added_OppTeam",
   "updated_OppTeam",
   "isHome_OppTeam"
]

rename_cols = {
   "playerId_Skater": "playerId",
   "gamePk_Skater": "gamePk",
   "gameDate_Game": "date",
   "codedGameState_Game": "gameState",
   "isHome_PlayerTeam": "isHome",
   "season_Game": "season"
}

one_hot_cols = [
    "position_Skater",
    "gameType_Game",
    "season"
]

fill_with_zeros = [
    "ot_OppTeam",
    "ot_PlayerTeam"
]

time_to_sec = [
    "timeOnIce_Skater",
    "evenTimeOnIce_Skater",
    "powerPlayTimeOnIce_Skater",
    "shortHandedTimeOnIce_Skater"
]

forbidden_stats = [
    "timeOnIce_Skater",
    "assists_Skater",
    "goals_Skater",
    "shots_Skater",
    "hits_Skater",
    "powerPlayGoals_Skater",
    "powerPlayAssists_Skater",
    "penaltyMinutes_Skater",
    "faceOffWins_Skater",
    "faceoffTaken_Skater",
    "takeaways_Skater",
    "giveaways_Skater",
    "shortHandedGoals_Skater",
    "shortHandedAssists_Skater",
    "blocked_Skater",
    "plusMinus_Skater",
    "evenTimeOnIce_Skater",
    "powerPlayTimeOnIce_Skater",
    "shortHandedTimeOnIce_Skater",
    "goals_PlayerTeam",
    "pim_PlayerTeam",
    "shots_PlayerTeam",
    "powerPlayPercentage_PlayerTeam",
    "powerPlayGoals_PlayerTeam",
    "powerPlayOpportunities_PlayerTeam",
    "faceOffWinPercentage_PlayerTeam",
    "blocked_PlayerTeam",
    "takeaways_PlayerTeam",
    "giveaways_PlayerTeam",
    "hits_PlayerTeam",
    "goalsAgainst_PlayerTeam",
    "pimAgainst_PlayerTeam",
    "shotsAgainst_PlayerTeam",
    "powerPlayPercentageAgainst_PlayerTeam",
    "powerPlayGoalsAgainst_PlayerTeam",
    "powerPlayOpportunitiesAgainst_PlayerTeam",
    "faceOffWinPercentageAgainst_PlayerTeam",
    "blockedAgainst_PlayerTeam",
    "takeawaysAgainst_PlayerTeam",
    "giveawaysAgainst_PlayerTeam",
    "hitsAgainst_PlayerTeam",
    "wins_PlayerTeam",
    "losses_PlayerTeam",
    "ot_PlayerTeam",
    "score_PlayerTeam",
    "goals_OppTeam",
    "pim_OppTeam",
    "shots_OppTeam",
    "powerPlayPercentage_OppTeam",
    "powerPlayGoals_OppTeam",
    "powerPlayOpportunities_OppTeam",
    "faceOffWinPercentage_OppTeam",
    "blocked_OppTeam",
    "takeaways_OppTeam",
    "giveaways_OppTeam",
    "hits_OppTeam",
    "goalsAgainst_OppTeam",
    "pimAgainst_OppTeam",
    "shotsAgainst_OppTeam",
    "powerPlayPercentageAgainst_OppTeam",
    "powerPlayGoalsAgainst_OppTeam",
    "powerPlayOpportunitiesAgainst_OppTeam",
    "faceOffWinPercentageAgainst_OppTeam",
    "blockedAgainst_OppTeam",
    "takeawaysAgainst_OppTeam",
    "giveawaysAgainst_OppTeam",
    "hitsAgainst_OppTeam",
    "wins_OppTeam",
    "losses_OppTeam",
    "ot_OppTeam",
    "score_OppTeam"
]