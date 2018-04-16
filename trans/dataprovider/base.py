from datetime import timedelta

"""
Base class for DataProvicer

Retrieves a DataFrame int "standard format" from a provider.
If the provider's native format is not "standard format", the modify method is intended to convert it.

"Standard format":
Create one Dataframe, with data from all the tickers
The Dataframe:
- index is named "Date" (a date value is of type string)
- columns: MultiIndex
-  level 0: attribute
-  level 1: ticker
"""

class DataProviderBase:
    def __init__(self, debug=False, **params):
        self.Debug = debug


    def modify(self, df):
        """
        Modify the DataFrame df returned by the provider so that it conforms with the proper output DataFrame format.
        """
        return df

    def get(self, tickers=None, start=None, end=None):
        """
        Return a DataFrame in "standard format" containing the given tickers, in the date range from start to end

        Parameters
        ----------
        tickers: list of strings
        start, end: DateTime
        """

        print("init: need to override")
        df_w = pd.DataFrame()
        return df_w
    
