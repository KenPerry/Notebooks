from trans.dataprovider.PDR.base import PDRBase

# SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, Date, create_engine, bindparam

import trans.gtrans as gt

tableName = "prices"

class Tiingo(PDRBase):
    colMap = { "open":   "Open",
               "high":   "High",
               "low":    "Low",
               "close":  "Close",
               "adjClose": "AdjClose",
               "volume": "Volume",
               "divCash": "Div"
    }
    
    def __init__(self, **params):
        super().__init__("tiingo", **params)

    def modify(self, df):
        # Drop the ticker from the index
        df.index = df.index.droplevel(0)
        df.index = df.index.rename('Date')

        # Retain only the mapped columns
        df = df.loc[:, list(self.colMap.keys()) ]
        df = df.rename( columns=self.colMap)

        gt.good_housekeeping(df, inplace=True)
        
        return df

    def recordConstructor(self, Base):
        class Price(Base):
            __tablename__ = tableName
            Ticker = Column(String(255), primary_key=True)
            Date   = Column(Date, primary_key=True)
            Close   = Column(Float)
            AdjClose = Column(Float)
            High    = Column(Float)
            Low     = Column(Float)
            Open    = Column(Float)
            Volume  = Column(Float)
            Div     = Column(Float)

        return Price

