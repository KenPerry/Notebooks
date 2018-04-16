from trans.dataprovider.base import DataProviderBase

import pandas as pd

import datetime as dt
import dateutil.parser as dup
import pandas_datareader.data as web

import trans.gtrans as gt

class Yahoo(DataProviderBase):
    def __init__(self, debug=False, **params):
        self.Debug = debug


    def get(self, tickers=None, start=None, end=None):
        """
        Create one Dataframe, with data from all the tickers
        The Dataframe:
        - index is Date (as a string, same as in the per-ticker files)
        - columns: MultiIndex
        -  level 0: attribute
        -  level 1: ticker
        """

        start_d, end_d = dt.datetime.strftime(start, "%m/%d/%Y"), dt.datetime.strftime(end, "%m/%d/%Y")
        dfs = []
        
        for ticker in tickers:
            succeed, tries = False, 0


            while( (tries < 2) and not succeed):
                try:
                    df = web.DataReader(ticker, "yahoo", start, end)

                    # Duplicates are returned sometimes.  Drop them
                    df.drop_duplicates(inplace=True)

                    succeed = True
                except Exception as e:
                    tries += 1
                    print("get: Yahoo exception for {t}: {et} - {ea}".format(t=ticker, et=type(e), ea=e))
                    print("get: Yahoo error for {t}, re-try {tn}.".format(t=ticker, tn=tries))


            df.index.rename('Date', inplace=True)

            dfs.append(df)

        # Combine per-ticker dataframes
        df_big = pd.concat(dfs, axis=1, keys=tickers)
        # df_big.index.name = "Date"

        # Make the first level column index be attribute; the second will be ticker
        df_big = df_big.swaplevel(axis=1,i=0,j=1)

        # Always a good idea to sort index after concat
        gt.good_housekeeping(df_big, inplace=True)

        return df_big

