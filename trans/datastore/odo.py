from trans.datastore.base import DataStoreBase

import pandas as pd

from trans.data import GetData
gd = GetData()

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

# Base = declarative_base()


class ODO(DataStoreBase):
    def __init__(self, dbURL, provider, debug=False, **params):
        self.dbURL    = dbURL
        self.provider = provider

        if "declarative_base" in params:
            self.decBase = params["declarative_base"]

        # Create the class for the records of the database
        decBase = self.decBase
        recConstructor = self.provider.recordConstructor(decBase)
        self.recConstructor = recConstructor

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
    def setup_database(self):
        dbURL = self.dbURL
        engine = self.engine

        decBase = self.decBase
        decBase.metadata.drop_all(engine)
        decBase.metadata.create_all(engine)
    
    def createSession(self):
        engine = self.engine
        session = Session(bind=engine)

        self.session = session
        return session

    def existing(self):
        session = self.session
        rc = self.recConstructor
        
        existingDates = {}
        query = session.query(rc, rc.Ticker, func.min(rc.Date).label("min_date"), func.max(rc.Date).label("max_date")).group_by(rc.Ticker)

        results = query.all()

        # Iterate through results
        for res in results:
            existingDates[ res.Ticker ] = (res.min_date, res.max_date)

        self.existingDates = existingDates

        tickers = list(existingDates.keys())
        return tickers

    def modify_in(self, df):
        df = df.rename( columns={ "Adj Close": "AdjClose"} )

        return df

    def modify_out(self, df):
        df = df.rename( columns={ "AdjClose": "Adj Close"} )

        return df

    def get_one(self, ticker, start, end):
        """
        Get data from Web for single ticker for date range from start to end

        Parameters:
        - ticker: string
        - start, end: datetime

        Returns:
        status, DataFrame

        - status: True on success
        - DataFrame: resulting data
        """

        # status, df =  gd.get_one(ticker, start, end)
        #if status:
        #    df = self.modify_in(df)

        status = False
        
        provider = self.provider
        df = provider.get(tickers=[ticker], start=start, end=end)
        if not df.empty:
            # Convert from provider "standard format"
            df.columns = df.columns.droplevel(level=1)

            # Convert to Price table format
            df = self.modify_in(df)

            status = True
        else:
            print("get_one: {t} returns no data.".format(t=ticker))

        return status, df
            
    def extend(self, ticker, start, end):
        """
        Get data for a single ticker for date range from start to end, 
        If a file containing data for this ticker already exists, extend it to end if needed.

        Note:
        The Date column in a file is a date string (YYYY-MM-YY); the Date column from the Web is a datetime (sub-day resolution).
        Extend concatenates, resulting in the higher resolution Date when extending occurs.

        Parameters:
        - ticker: string
        - start, end: datetime


        Returns:
        status, DataFrame

        - status: True if new data has been obtained
        - DataFrame: the data (updated or original)
        """

        existingDates = self.existingDates
        
        changed = False
        df = pd.DataFrame()

        # Does this ticker exist already ?
        if ticker in existingDates.keys():
            (min_d, max_d) = existingDates[ticker]
            # Find the first and last date for which we already have data
            last_dt  = dt.datetime.combine( max_d, dt.time.min)
            first_dt = dt.datetime.combine( min_d, dt.time.min)

            oneDayTimeDelta = dt.timedelta(days=1)

            # Get data for dates following lastd
            if start is None:
                start = last_dt + oneDayTimeDelta
            elif (start > first_dt):
                start = last_dt + oneDayTimeDelta
            else:
                # May be the case that there is NO data to be had prior to first_dt, so is a waste to try to get it
                start = last_dt + oneDayTimeDelta
                # end = first_dt - oneDayTimeDelta

            if (start < end):
                print("Extend {t} from {l} beginning on {sn} to {e}.".format(t=ticker,
                                                                             l =dt.datetime.strftime(last_dt, "%m/%d/%Y"),
                                                                             sn=dt.datetime.strftime(start, "%m/%d/%Y"),
                                                                             e =dt.datetime.strftime(end, "%m/%d/%Y"))
                      )

                status, df = self.get_one(ticker, start, end)

                # Did get_one really extend df ?
                # NOTE: the file index are Dates (MM/DD/YYYY), get_one index is DateTime
                if status and ( (df.index.max()  > last_dt) or (df.index.min() < first_dt) ):
                    changed = True
                    self.update(ticker, df)

        else:
            # Ticker does not exist yet
            start = start or self.start_default
            status, df = self.get_one(ticker, start, end)

            if status:
                print("Creating {t} from {s} to {e}".format(t=ticker, s=start, e=end))
                changed = True
                self.update(ticker, df)
            
        return changed, df

    
    def get_data(self, tickers, start, end):
        """
        Get data for each ticker in tickers, from date range start to end
        - tickers: list of tickers
        - start, end: datetimes

        Returns:
        nothing

        """

        # Make sure that self.existingDates (map from ticker to latest date) exists
        existingDates = self.existing()

        changed_tickers = []
        
        # Get data for each ticker
        for ticker in tickers:
            changed, df = self.extend(ticker, start, end)
            
            # just in case your connection breaks, we'd like to save our progress!
            if changed:
                changed_tickers.append(ticker)
            else:
                print('Already have up-to-date {}.'.format(ticker))

        return changed_tickers

        
        
    def update(self, ticker, df):
        session = self.session
        rc = self.recConstructor
        
        # Delete any existing rows that will be overwritten from df
        dates_up = df.index.tolist()
        dates_up_min, dates_up_max = df.index.min(), df.index.max()

        # query  = session.query(Price).filter(and_(Price.Ticker == ticker, Price.Date.in_(dates_up)))
        # query  = session.query(Price).filter(and_(Price.Ticker == ticker, Price.Date >= dates_up_min, Price.Date <= dates_up_max))
        query  = session.query(rc).filter(and_(rc.Ticker == ticker, rc.Date >= dates_up_min, rc.Date <= dates_up_max))
        
        rows_num =  query.delete(synchronize_session="fetch") # or = False
        print("Ticker {t}, Deleted {n} rows between {s} and {e}".format(t=ticker, n=rows_num, s=dates_up_min, e=dates_up_max))
        

        # Add ticker as column of df
        df["Ticker"] = ticker

        # Move date from index to columns
        df = df.reset_index()
        # df["Date"] = pd.to_datetime( df["Date"])

        
        # Insert the rows of df
        # Convert each row of the df to a dict
        rows = df.to_dict("records")

        inserted_num = 0
        queued_max = 2000

        # Insert one row at a time
        for row in rows:
            #rec = Price(**row)
            rec = rc(**row)
    
            session.add(rec)

            # Don't allow too many rows to be queued in the session
            inserted_num += 1

            if (inserted_num >= queued_max):
                try:
                    print("Update {t}: flushing after {n} rows inserted.".format(t=ticker, n=inserted_num))
                    
                    session.flush()
                except SQLAlchemyError as e:
                    print("Flush error: {}".format(e))

                inserted_num = 0
                
        # Flush anything queued in the session
        try:
            session.flush()
        except SQLAlchemyError as e:
            print("Flush error: {}".format(e))


        # Commit
        session.commit()    

    def combine_data(self, tickers=None, start=None, end=None):
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

        # query = session.query(Price)
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

        # Modify
        df = self.modify_out(df)

        # Create a  Datetime column from Date
        df[ "Dt" ] = df.loc[:,"Date"].map( lambda  s: pd.to_datetime(s, infer_datetime_format=True) )
        df = df.drop("Date", axis=1)

        # Set index to be (Dt, Ticker)
        df.set_index(["Dt", "Ticker"], inplace=True)

        # Turn thin table into wide by moving Ticker index to column
        df_w = df.unstack(level=1)

        return df_w
    
    def wide_to_thin(self, df):
        # Move ticker (level 1 of df.columns) to index, then make it a regular column
        df_t = df.stack(level=1)
        df_t.reset_index(inplace=True)

        return df_t
    
