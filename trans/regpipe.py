import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from datetime import timedelta

import re


from trans.gtrans import DataFrameConcat

idx = pd.IndexSlice

class RegPipe:
    """
    TO DO:
    reg.sensAttrs:  
       assumes data in self.data; make it take df as arg ?
       assumes betas in same df as data, that's why sensAttrs needs to serve

    depTickersFromSensAttrs(sensAttrs ):
       assumes data in self.data; make it take df as arg ?
          that's why needs to search for non-sens as dependent

    reg.retAttrib:
       assumes sensitivities, indVars and depVars are all in same df
        reg is ret_and_rolled_beta_df


    - 
    """
    
    def __init(self, df, debug=False):
        self.df = df
        self.debug = debug

        self.reg = Reg(df)


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
        
        ma = .modelCols( [ indCol ])

        beta_df = reg.rollingModelAll( *ma, 
                                       start, end,
                                       step
                                       )

        self.beta_df = beta_df

 
    def rollBeta(self):
        beta_df = self.beta-df

        # To DO: Only need betaAttrs = ..; and GenSelectAttrsTransformer(betaAttrs) IF df contains something OTHER than betas, but beta_df ONLY contains betas
        rab = Reg(beta_df)
        betaAttrs = rab.sensAttrs('^Beta \d+$')

        beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaAttrs),
                                   ShiftTransformer(1),
                                   FillNullTransformer(method="ffill"),
                                   GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
        
        beta_rolled_df = beta_r_pl.fit_transform(beta_df)

        self.beta_rolled_df = beta_rolled_df
        

    def attrib(self):
        indCols = [ ("Pct", "1"), self.indCol ]

        # TO DO: only need attrib.sensAttrs IF df contains something OTHER than rolled betas
        sensAttrs = attrib.sensAttrs('^Beta \d+ rolled fwd$')

        depTickers = attrib.depTickersFromSensAttrs(sensAttrs )
        depCols = [ ("Pct", t) for t in depTickers ]

        attrib.addConst(("Pct", "1"), 1)

        # TO DO: assumes the df in reg contains both dependents/independents (indCols, depCols) AND sensitivities in same df
        retAttr_df =reg.retAttrib(
            indCols,
            depCols, 
            sensAttrs)

        self.retAttr_df = retAttr_df

        
        

        
