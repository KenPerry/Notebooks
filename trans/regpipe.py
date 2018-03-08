import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
from datetime import timedelta

import re

from trans.gtrans import *
from trans.reg import Reg, RegAttr

idx = pd.IndexSlice

class RegPipe:
    """

    """
    
    def __init__(self, df, debug=False):
        self.df = df.copy()
        self.debug = debug

        self.reg = Reg(df)
        self.regAttr = RegAttr(df)


    def indCol(self, col):
        self.indCol = col
        

    def addConst():
        """
        Add constant column (needed only for attribution, not regression)
        """

        self.reg.addConst(("Pct", "1"), 1)
        
        
    def regress(self, start, end, window, step):
        reg = self.reg
        indCol = self.indCol
        
        ma = reg.modelCols( [ indCol ])

        beta_df = reg.rollingModelAll( *ma, 
                                       start, end,
                                       step
                                       )

        self.beta_df = beta_df


    def indexUnionBeta(self, freq_index):
        """
        Re-shapes self.beta_df

        The modified self.beta_df index is the UNION of the original index f self.beta_df and the index freq_index
        It DIFFERS from pd.reindex(freq_index) as follows:
        - we get the UNION of the dates in the original and in freq_index
        - pd.reindex(foo) will create a DataFrame whose index is exactly foo
        """
        
        beta_df = self.beta_df

        # Create an empty DataFrame with the index freq_index
        empty_df = pd.DataFrame(index=freq_index)

        # Join beta_df with the empty dataframe, resulting in an index that is the union of the indices of the two constituents
        beta_df = pd.concat( [beta_df, empty_df], axis=1)
        
        self.beta_df = beta_df
        
    def rollBeta(self, periods, fill_method):
        """
        fill_method: {"ffill", "bfill"}
        - ffill: forward fill
        - bfill: backwards fill
        """
 
        beta_df = self.beta_df

        beta_r_pl = make_pipeline( ShiftTransformer(periods),
                                   FillNullTransformer(method=fill_method),
                         )
        
        beta_rolled_df = beta_r_pl.fit_transform(beta_df)

        self.beta_rolled_df = beta_rolled_df
        

    def attrib(self):
        reg = self.reg
        regAttr = self.regAttr
        
        df  = reg.data

        beta_df = self.beta_rolled_df

    
        indCols = [ ("Pct", "1"), self.indCol ]

        sensAttrs = beta_df.columns.get_level_values(0).unique().tolist()

        regAttr.setSens(beta_df)

        depTickers = regAttr.depTickersFromSensAttrs(sensAttrs )
        depCols = [ ("Pct", t) for t in depTickers ]

        regAttr.addConst(("Pct", "1"), 1)

        # TO DO: assumes the df in reg contains both dependents/independents (indCols, depCols) AND sensitivities in same df
        retAttr_df =regAttr.retAttrib(
            indCols,
            depCols, 
            sensAttrs)

        self.retAttr_df = retAttr_df

        
        

        
