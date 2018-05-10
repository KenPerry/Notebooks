from trans.dataprovider.base import DataProviderBase

# SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, Date, create_engine, bindparam
from sqlalchemy import Table, MetaData

import pandas as pd
import trans.gtrans as gt

import datetime as dt
from datetime import timedelta
import dateutil.relativedelta as rd

tableName = "prices"

meta = MetaData()

class Alphavantage(DataProviderBase):
    colMap = { "open":   "Open",
               "high":   "High",
               "low":    "Low",
               "close":  "Close",
               "adjusted_close": "AdjClose",
               "volume": "Volume",
               "dividend_amount": "Div",
               "split_coefficient": "Factor",
               "timestamp": "Date"
    }
    
    def __init__(self, **params):
        av_ak_file="/home/ubuntu/Notebooks/alphavantage_apkey.txt"
        with open(av_ak_file, "r") as fp:
            apikey = fp.read().rstrip()
            self.apikey = apikey

        self.url="https://alphavantage.co/query"

        av_func= "TIME_SERIES_DAILY_ADJUSTED"
        url_format = "https://www.alphavantage.co/query?function={f}&datatype=csv&apikey={ak}".format(f=av_func, ak=apikey)
        url_format = url_format + "&outputsize={os}&symbol={s}"
        self.url_format = url_format


    def get(self, tickers=None, start=None, end=None):
        """
        Return a DataFrame in "standard format" containing the given tickers, in the date range from start to end

        Parameters
        ----------
        tickers: list of strings
        start, end: DateTime
        """

        source = "Alphavantage"

        # Determine the size
        # The choices are:
        # - "full": gets entire history
        # - "compact": gets only the most recent 100 days
        
        today = dt.datetime.combine( dt.date.today(), dt.time.min)
        if (end != today):
            # End not today, must use full
            size = "full"
        elif ( (end - rd.relativedelta(days=100)) <= start):
            # Less than the most recent 100 days: use compact
            size = "compact"
        else:
            size = "full"

    
        dfs = []                                                                                                                      
        url_format = self.url_format

        keys= set( list(self.colMap.keys()) )
        
        for ticker in tickers:
            succeed, tries = False, 0

            while( (tries < 2) and not succeed):
                try:
                    url = url_format.format(s=ticker, os=size)
                    df = pd.read_csv(url)

                    if ( len( list( keys - set(df.columns.tolist()) ) ) == 0 ):
                        succeed = True
                    else:
                        print("get: {src} bad column names for {t}: {cn}, keys = {k}".format(src=source, t=ticker, cn=df.columns.tolist(), k=keys ))
                        tries +=1
                except Exception as e:
                    tries += 1
                    print("get: {src} exception for {t}: {et} - {ea}".format(src=source, t=ticker, et=type(e), ea=e))
                    print("get: {src} error for {t}, re-try {tn}.".format(src=source, t=ticker, tn=tries))

                  
            if succeed:
                # Retain only the mapped columns
                df = df.loc[:, list(self.colMap.keys()) ]
                df = df.rename( columns=self.colMap)

                # Add Date as an index (datetime)
                df.loc[:, "Date"] = df.loc[:,"Date"].map( lambda  s: pd.to_datetime(s, infer_datetime_format=True))
                df.set_index("Date", inplace=True)
                df.sort_index(axis=0, inplace=True)
            else:
                df = pd.DataFrame()
            
            dfs.append(df)

        # Combine per-ticker dataframes
        df_big = pd.concat(dfs, axis=1, keys=tickers)

        # Make the first level column index be attribute; the second will be ticker
        df_big = df_big.swaplevel(axis=1,i=0,j=1)

        # Always a good idea to sort index after concat
        gt.good_housekeeping(df_big, inplace=True)

        return df_big
    

    def recordConstructor(self, Base):
        class Price(Base):
            __tablename__ = tableName
            # extend_existing = True
            #__table__ = Table(tableName, meta, extend_existing=True)
            Ticker = Column(String(255), primary_key=True)
            Date   = Column(Date, primary_key=True)
            Close  = Column(Float)
            AdjClose = Column(Float)
            High    = Column(Float)
            Low     = Column(Float)
            Open    = Column(Float)
            Volume  = Column(Float)
            Div     = Column(Float)
            Factor  = Column(Float)


        return Price

