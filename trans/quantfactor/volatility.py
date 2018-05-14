from trans.quantfactor.base import QuantFactorBase

import pandas as pd
from datetime import timedelta

idx = pd.IndexSlice

"""
Implement the Volatility factor

Derived method "fit" computes the volatility of returns over a window
"""

class Volatility(QuantFactorBase):

    def fit(self, df, start, end):
        """
        Parameters:
        -----------
        df: DataFrame containing daily returns of each member of the universe
        start, end: 
        """
        
        # NOTE: s,e are DateTimes so s:e are all the rows INCLUSIVE of e (as opposed to if s,e were integers, in which case last row is (e-1)
        thisDf = df.loc[start:end,:]

        if self.Debug:
            print("create_period_attr: {s} to {e}".format(s=s, e=e))

        # Return a list
        return thisDf.std().tolist()
        
