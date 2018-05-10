from datetime import timedelta

"""
Base class for DataProvicer

Retrieves a DataFrame in "standard format" from a provider.
If the provider's native format is not "standard format", the modify method is intended to convert it.

"Standard format":
Create one Dataframe, with data from all the tickers
The Dataframe:
- index is named "Date" (a date value is of type string)
- columns: MultiIndex
-  level 0: attribute
-  level 1: ticker

The DataProvider dp implements a method "recConstructor" that defines the storage shape.  This is used only as a helper for DataStore to assist in storing this type of data.

"""

class DataProviderBase:
    def __init__(self, debug=False, **params):
        self.Debug = debug


    def modify(self, df):
        """
        Modify the DataFrame returned by get so as to render it suitable to insert into the database by ODO.
        - the database record format (i.e., table layout) is specified by the recordConstructor method.
        - so modify should take the DataFrame returned by PDEBase.get and restrict and rename the columns to conform to the database format:
        - No ticker column
        - Date and attribute columns (single level columns, NOT MultiIndex)

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

        print("get: need to override")
        df_w = pd.DataFrame()

        return df_w
    
    def recordConstructor(self, Base):
        """
        Return a sub-type of Base (which is constructed by caller as Base = declarative_base(), for sqlalchemy.ext.declarative.declarative_base)
        This is the database record format (i.e., table layout) for storing records returned by PDRBase.get, as modified by the modify method
        """
        print("recordConstructor: need to override")
