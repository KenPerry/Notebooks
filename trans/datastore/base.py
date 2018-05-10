from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe


from datetime import timedelta

"""
Base class for DataStore

DataStore is used to obtain external data and store it in a database.
It
- uses a DataProvider to obtain data (e.g., Yahoo)
  The DataProvider dp implements a method "recConstructor" that defines the storage shape.

- stores the data in a database, using the DataProvider's storage shape



"""

class DataStoreBase:
    def __init__(self, debug=False, **params):
        self.Debug = debug
        print("init: need to override")


    def setup_database(self):
        print("init: need to override")

    
    def createSession(self):
        print("init: need to override")
  
    def existing(self):
        print("init: need to override")
        tickers = []
        return tickers

    def modify_in(self, df):
        print("init: need to override")
        df = pd.DataFrame()

        return df

    def modify_out(self, df):
        print("init: need to override")
        df = pd.DataFrame()

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
        
        print("init: need to override")
        status, df =  False, pd.DataFrame()
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

        print("init: need to override")
        changed = False
        df = pd.DataFrame()
            
        return changed, df

    
    def get_data(self, tickers, start, end):
        """
        Get data for each ticker in tickers, from date range start to end
        - tickers: list of tickers
        - start, end: datetimes

        Returns:
        nothing

        """

        print("init: need to override")
        changed_tickers = []
        
        return changed_tickers

        
        
    def update(self, ticker, df):
        print("init: need to override")

    def combine_data(self, tickers=None, start=None, end=None):
        """
        Create one Dataframe, with data from all the tickers
        The Dataframe:
        - index is Date (as a string, same as in the per-ticker files)
        - columns: MultiIndex
        -  level 0: attribute
        -  level 1: ticker
        """

        print("init: need to override")
        df_w = pd.DataFrame()
        return df_w
    
