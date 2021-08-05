import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Time, Float
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

Base_bets = declarative_base()


class Bet(Base_bets):
    __tablename__ = "bet"

    id = Column('id', Integer, primary_key=True)
    playerId = Column('playerId', String(50))
    homeTeamId = Column('homeTeamId', String(50))
    awayTeamId = Column('awayTeamId', String(50))
    dateTime = Column('dateTime', DateTime)

    site = Column('site', String(50))
    overUnder = Column('overUnder', Float)
    oddsOver = Column('oddsOver', Float)
    oddsUnder = Column('oddsUnder', Float)
    
    gamePk = Column('gamePk', Integer)

    added = Column('added', DateTime, default=datetime.datetime.utcnow)
