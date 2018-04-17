from trans.dataprovider.base import DataProviderBase

import pandas as pd

import datetime as dt
import dateutil.parser as dup
import pandas_datareader.data as web

import trans.gtrans as gt

"""
Base class for Pandas DataReader (PDR) based DataProvider

"""

class PDRBase(DataProviderBase):
    def __init__(self, source, debug=False, **params):
        self.source = source
        self.Debug = debug

        # Additional arguments for web.DataReader
        add_args = {}
        
        if "access_key" in params:
            self.access_key = params["access_key"]
            add_args["access_key"] = params["access_key"]


        self.add_args = add_args
        
    def get(self, tickers=None, start=None, end=None):
        """
        Create one Dataframe, with data from all the tickers
        The Dataframe:
        - index is Date (as a string, same as in the per-ticker files)
        - columns: MultiIndex
        -  level 0: attribute
        -  level 1: ticker
        """

        # Get the data source, any additional arguments needed
        source = self.source
        add_args = self.add_args
        
        start_d, end_d = dt.datetime.strftime(start, "%m/%d/%Y"), dt.datetime.strftime(end, "%m/%d/%Y")
        dfs = []

        for ticker in tickers:
            succeed, tries = False, 0

            while( (tries < 2) and not succeed):
                try:
                    df = web.DataReader(ticker, source, start, end, **add_args)

                    # Duplicates are returned sometimes.  Drop them
                    df.drop_duplicates(inplace=True)

                    succeed = True
                except Exception as e:
                    tries += 1
                    print("get: {src} exception for {t}: {et} - {ea}".format(src=source, t=ticker, et=type(e), ea=e))
                    print("get: {src} error for {t}, re-try {tn}.".format(src=source, t=ticker, tn=tries))


            df = self.modify(df)
            
            dfs.append(df)

        # Combine per-ticker dataframes
        df_big = pd.concat(dfs, axis=1, keys=tickers)

        # Make the first level column index be attribute; the second will be ticker
        df_big = df_big.swaplevel(axis=1,i=0,j=1)

        # Always a good idea to sort index after concat
        gt.good_housekeeping(df_big, inplace=True)

        return df_big

