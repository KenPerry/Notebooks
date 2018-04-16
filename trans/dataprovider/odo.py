from trans.dataprovider.base import DataProviderBase

import pandas as pd


# Date utiities
import datetime as dt
from datetime import date
from datetime import timedelta
import dateutil.parser as dup
from datetime import timedelta

# SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, bindparam
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float,create_engine, bindparam
from sqlalchemy.orm import Session

from sqlalchemy.sql import and_, or_, not_, func

tableName = "prices"

Base = declarative_base()

class Price(Base):
    __tablename__ = tableName
    Ticker = Column(String(255), primary_key=True)
    Date   = Column(Date, primary_key=True)
    AdjClose = Column(Float)
    Close   = Column(Float)
    High    = Column(Float)
    Low     = Column(Float)
    Open    = Column(Float)
    Volume  = Column(Float)


class ODO(DataProviderBase):
    def __init__(self, dbURL, debug=False, **params):
        self.dbURL    = dbURL
        self.Debug = debug

        if "echo" in params:
            echo = params["echo"]
        else:
            echo=False

        if "start_default" in params:
            self.start_default = params["start_default"]
        else:
            self.start_default = dup.parse("01/01/2000")
            
        engine = create_engine(dbURL, echo=echo)
        self.engine = engine

        session = self.createSession()



    # INTERNAL methods: take df as argument
    #-------------------------------------------------------

    
    # EXTERNAL methods: takes df from SELF.df
    #-------------------------------------------------------
    def createSession(self):
        engine = self.engine
        session = Session(bind=engine)

        self.session = session
        return session

    def get(self, tickers=None, start=None, end=None):
        """
        Create one Dataframe, with data from all the tickers
        The Dataframe:
        - index is Date (as a string, same as in the per-ticker files)
        - columns: MultiIndex
        -  level 0: attribute
        -  level 1: ticker
        """

        session = self.session
        conditions = []

        query = session.query(Price)

        if tickers:
            # NOTE: if tickers is too long, the in_ may break in SQL.  Better to put tickers in a temp table and join with it to limit the tickers retrieved
            conditions.append( Price.Ticker.in_(tickers) )
            
        if start:
            conditions.append(  Price.Date >= start )

        if end:
            conditions.append(  Price.Date <= end )

        if len(conditions) > 0:
            query = query.filter( and_( *conditions ) )
            
        df = pd.read_sql(query.statement, session.bind)

        # Modify
        df = self.modify(df)

        # Create a  Datetime column from Date
        df[ "Dt" ] = df.loc[:,"Date"].map( lambda  s: pd.to_datetime(s, infer_datetime_format=True) )
        df = df.drop("Date", axis=1)

        # Set index to be (Dt, Ticker)
        df.set_index(["Dt", "Ticker"], inplace=True)

        # Turn thin table into wide by moving Ticker index to column
        df_w = df.unstack(level=1)

        return df_w
