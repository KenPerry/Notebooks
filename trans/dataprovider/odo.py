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
from sqlalchemy import Column, Integer, Float, String, Date, create_engine, bindparam
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError


from sqlalchemy.sql import and_, or_, not_, func

tableName = "prices"

# Base = declarative_base()


class ODO(DataProviderBase):
    def __init__(self, dbURL, debug=False, **params):
        """
        Obtains data from a database.

        We need to know several things about the database:
        dbURL: it's URL
        
        The format in which records are stored.
        This is obtained by passing in a DataProvider.  
        The DataProvider implements a helper function (recordConstructor) which defines the storage format for the data from a given DataProvider.
        So the DataProvider passed in now (for reading) should be the same as used by the DataStore in the past (when writing).

        params:
        -------
        "provider": a DataProvider
        "declarative_base": A sqlalchemy.ext.declarative.declarative_base.
              This is needed  by the DataProvider's recordConstructor


        """
        self.dbURL    = dbURL
        self.Debug = debug

        
        if "declarative_base" in params:
            self.decBase = params["declarative_base"]

        if "provider" in params:
            self.provider = params["provider"]

            decBase = self.decBase
            
            # Create the class for the records of the database
            recConstructor = self.provider.recordConstructor(decBase)
            self.recConstructor = recConstructor
        else:
            print("ODO.__init__: MUST provide \"provider\" param in order to know records in db are stored.")
            return

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

    def modify(self, df):
        # Input df  is "thin": one column each for Date, Ticker, and attributes (i.e., df.columns ins NOT MultiIndex)
    
        # Rename columns
        df = df.rename( columns={ "AdjClose": "Adj Close"} )

        # Create a  Datetime column from Date
        df[ "Dt" ] = df.loc[:,"Date"].map( lambda  s: pd.to_datetime(s, infer_datetime_format=True) )
        df = df.drop("Date", axis=1)

        # Set index to be (Dt, Ticker)
        df.set_index(["Dt", "Ticker"], inplace=True)

        # Turn thin table into wide by moving Ticker index to column
        df_w = df.unstack(level=1)

        return df_w

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
        rc = self.recConstructor

        conditions = []

        query = session.query(rc)

        if tickers:
            # NOTE: if tickers is too long, the in_ may break in SQL.  Better to put tickers in a temp table and join with it to limit the tickers retrieved
            conditions.append( rc.Ticker.in_(tickers) )
            
        if start:
            conditions.append(  rc.Date >= start )

        if end:
            conditions.append(  rc.Date <= end )

        if len(conditions) > 0:
            query = query.filter( and_( *conditions ) )
            
        df = pd.read_sql(query.statement, session.bind)

        # Modify: rename columns, turn df from thin to wide
        df_w = self.modify(df)

        return df_w
