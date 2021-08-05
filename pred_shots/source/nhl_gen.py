from source.db_models.bets_models import *
from source.db_models.nhl_models import *
from datetime import date, datetime, timedelta
from sqlalchemy import func, and_, or_, not_, asc, desc
import pandas as pd
import sqlalchemy
from sqlalchemy import select
from sqlalchemy.orm import aliased
from tqdm import tqdm
from source.nhl_handler import *
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


    playerStatsForGames["ans_O_1.5"] = (playerStatsForGames["shots_Skater"] > 1.5).astype(int)
    playerStatsForGames["ans_O_2.5"] = (playerStatsForGames["shots_Skater"] > 2.5).astype(int)
    playerStatsForGames["ans_O_3.5"] = (playerStatsForGames["shots_Skater"] > 3.5).astype(int)
    playerStatsForGames["ans_O_4.5"] = (playerStatsForGames["shots_Skater"] > 4.5).astype(int)

    df = clean_data(playerStatsForGames)
    df = generate_prediction_data(df, nhl_session)

    # One hot encode the categorical variables
    df = pd.get_dummies(df, columns=one_hot_cols)

    return df


def replace_team_data(df, nhl_session):
    df.drop(forbidden_stats_PlayerTeam, axis=1, inplace=True)
    df.drop(forbidden_stats_OppTeam, axis=1, inplace=True)

    # Create empty df to fill
    final_df = pd.DataFrame()

    # Loop trough each game
    for i, row in df.iterrows():
        playerTeamId = row['teamId_PlayerTeam']
        oppTeamId = row['teamId_OppTeam']
        gamePk = row['gamePk']
        season = row['season']

        # Get the team stats for the player and the opponent
        playerTeamStats = get_team_stats(playerTeamId, season, nhl_session, "PlayerTeam")
        oppTeamStats = get_team_stats(oppTeamId, season, nhl_session, "OppTeam")

        # Get the current games team stats
        current_game_org_stats = df[df.gamePk == gamePk].reset_index()
        current_game_team_stats = playerTeamStats[playerTeamStats.gamePk == gamePk].reset_index()
        current_game_opp_stats = oppTeamStats[oppTeamStats.gamePk == gamePk].reset_index()

        # Construct the new row
        result = pd.concat([current_game_org_stats, current_game_team_stats], axis=1, join='inner')
        result = pd.concat([result, current_game_opp_stats], axis=1, join='inner')

        # Remove all duplicate columns in result
        result = result.loc[:,~result.columns.duplicated()]

        # Save to final df
        final_df = pd.concat([final_df, result], axis=0, ignore_index=True)

    return final_df


def get_team_stats(teamId, season, nhl_session, suffix):
    # Get the team stats for this game
    query = (
        select(Game, TeamStats)
        .where(and_(TeamStats.teamId == teamId, Game.season == season))
        .join(Game, TeamStats.gamePk == Game.gamePk)
        .order_by(asc(Game.gameDate))
    )
    teamStats = pd.read_sql(query, nhl_session.bind)
    teamStats.columns = [u + "_Game" for u in Game.__table__.columns.keys()] \
                      + [u + "_" + suffix for u in TeamStats.__table__.columns.keys()]

    # Remove unwanted columns
    teamStats = clean_data(teamStats, 'ignore')

    forbidden_stats = forbidden_stats_PlayerTeam if suffix == "PlayerTeam" else forbidden_stats_OppTeam

    # Create new columns for the team stats
    for stat in forbidden_stats:
        teamStats[f'{stat}_ema_1_game_back'] = teamStats[stat].ewm(span=1, min_periods=1).mean().shift(1).copy()
        teamStats[f'{stat}_ema_3_game_back'] = teamStats[stat].ewm(span=3, min_periods=1).mean().shift(1).copy()
        teamStats[f'{stat}_ema_10_game_back'] = teamStats[stat].ewm(span=10, min_periods=1).mean().shift(1).copy()
        teamStats[f'{stat}_ema_1_season_back'] = teamStats[stat].ewm(span=10000, min_periods=1).mean().shift(1).copy()

    teamStats = teamStats.drop(forbidden_stats, axis=1)

    return teamStats


def generate_prediction_data(df, nhl_session):
    # Remove unwanted team data and replace it with reasonable values
    df = replace_team_data(df, nhl_session)

    # Group the data by season
    df_grouped = df.groupby(['season'])

    # Create empty df to fill
    final_df = pd.DataFrame()

    # loop through the seasons
    for season, season_df in df_grouped:
        for stat in forbidden_stats_Skater:
            # Calculate the EMA for each season
            season_df[f'{stat}_ema_1_games_back'] = season_df[stat].ewm(span=1, min_periods=1).mean().shift(1).copy()
            season_df[f'{stat}_ema_3_season_back'] = season_df[stat].ewm(span=3, min_periods=1).mean().shift(1).copy()
            season_df[f'{stat}_ema_10_season_back'] = season_df[stat].ewm(span=10, min_periods=1).mean().shift(1).copy()
            season_df[f'{stat}_ema_1_season_back'] = season_df[stat].ewm(span=10000, min_periods=1).mean().shift(1).copy()

        # Save data to the final df
        final_df = pd.concat([final_df, season_df])

    # Remove forbidden_stats from df
    final_df = final_df.drop(forbidden_stats_Skater, axis=1)

    return final_df


def clean_data(df, errors='raise'):

    # Drop the unnecessary columns
    df.drop(remove_cols, axis=1, inplace=True, errors=errors)

    # Rename columns
    df.rename(columns=rename_cols, inplace=True, errors=errors)

    # In each column of "fill_with_zeros" where there is not a number, put a zero
    for col in fill_with_zeros:
        if col in df.columns:
            df[col].replace(r'^\s*$', 0, regex=True, inplace=True)

    # Convert timestamps to datetime objects
    for col in time_to_sec:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col].astype(str)).dt.total_seconds().astype(int)

    return df


remove_cols = [
   "added_Skater",
   "team_Skater",
   "updated_Skater",
   "gamePk_Skater",
   "abstractGameState_Game",
   "detailedState_Game",
   "statusCode_Game",
   "startTimeTBD_Game",
   "homeTeamId_Game",
   "awayTeamId_Game",
   "added_Game",
   "updated_Game",

   "gamePk_PlayerTeam",
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
   "gamePk_Game": "gamePk",
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

forbidden_stats_Skater = [
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
    "shortHandedTimeOnIce_Skater"
]

forbidden_stats_PlayerTeam = [
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
    "score_PlayerTeam"
]

forbidden_stats_OppTeam = [
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