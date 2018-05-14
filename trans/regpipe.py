import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
from datetime import timedelta

import re

from trans.gtrans import *
from trans.reg import Reg, RegAttr

idx = pd.IndexSlice


### TO DO:
####  regAttr: the independent variables (inlcuding the constant) MUST have the same attribute, stored in self.attr (Don't think regAttr module requires this, so needless assumption)
####     see:
#####        depCols = [ (self.attr, t) for t in depTickers ]
#####        indCols_w_intercept.insert(0, (self.attr, "1"))
####  Sensitivity attribute is "Beta" (hard-coded).  Use this to find dep. tickers (level 1).  Then need to find depCols (assumes single attribute, noq)

class RegPipe:
    """

    """
    
    def __init__(self, df, attr=None, debug=False):
        if attr == None:
            attr = "Pct"
            print( "DEPRECATED: {cls}:__init__ called w/o \"attr\" arg., defaulting to {defv}.".format(cls=type(self), defv=attr) )

        self.attr = attr
        self.df = df.copy()
        self.debug = debug

        self.reg = Reg(df)


    def indCols(self, cols):
        """
        Set independent variable column names to cols
        """
        
        self.indCols = cols
        

    def addConst(self):
        """
        Add constant column (needed only for attribution, not regression)
        """

        self.reg.addConst((self.attr, "1"), 1)
        
        
    def regress(self, start, end, window, step):
        """
        Rolling regression of size window,  from end backwards to start, in steps of size step
        """
        
        reg = self.reg
        indCols = self.indCols
        
        ma = reg.modelCols( indCols )

        beta_df = reg.rollingModelAll( *ma, 
                                       start, end,
                                       window,
                                       step
                                       )

        self.beta_df = beta_df

    def regressSingle(self):
        """
        Single regression using entire DataFrame (in self.reg.data)
        """

        df = self.df
        start, end = df.index.min(), df.index.max()
        window = end - start + timedelta(days=1)
        step   = window

        return self.regress(start, end, window, step)
        
    def rollBeta(self, periods, fill_method):
        """
        Roll and fill the betas contained in self.beta_df

        self.beta_df usually sparse (e.g., every "step" days, where step is the increment of rolling regression).
        Result of rollBeta is completely filled DataFrame (using the fill_method).
        Before rolling, the betas can be shifted in time to accomodate in/out of sample calculations

        - periods=0, fill_method="bfill": can use to calculate regression residual
        - periods=1, fill_method="ffill": can use to calculate forward prediction/outperformance


        Parameters
        ----------
        periods: integer amount (positive/negative) to roll beta forward/backward before filling

        fill_method: {"ffill", "bfill"}
        - ffill: forward fill
        - bfill: backwards fill

            
        Returns
        ----------
        beta_rolled_df DataFrame
        """

        regAttr = self.regAttr
        beta_df = regAttr.beta_df

        beta_r_pl = make_pipeline( ShiftTransformer(periods),
                                   FillNullTransformer(method=fill_method),
                         )
        
        beta_rolled_df = beta_r_pl.fit_transform(beta_df)

        return beta_rolled_df


    def attrib_setup(self, df, beta_df, periods, fill_method):
        """
        Set self's internal attributes needed for attribution:
        - df: DataFrame containing independant and dependant varialbes
        - beta_df: DataFrame containing sensitivities of dependent to independent
        """

        # Join the two DataFrames so that independent/dependent variables and sensitivities guaranteed to have same index
        ## Use df_keys so we can separate the two parts later
        concatTrans = DataFrameConcat( [ df, beta_df ], df_keys=[ "Data", "Sens"])
        common_df = concatTrans.fit_transform(pd.DataFrame())

        # Need to be able to separate the joined DataFrame back into it's pieces
        sensAttrs = beta_df.columns.get_level_values(0).unique().tolist()

        # Separate the joined parts back into the individual components (but now having a common index)
        rp_df      =  common_df.loc[:, idx["Data",:,:]]
        rp_df.columns = rp_df.columns.droplevel(0)
                                       
                                       
        rp_beta_df =  common_df.loc[:, idx["Sens",:,:] ]
        rp_beta_df.columns = rp_beta_df.columns.droplevel(0)
        
        # Create an attribution object
        # Place the data and sensitivities in it
        regAttr = RegAttr(rp_df)
        self.regAttr = regAttr


        # Place the sensitivities into regAttr
        regAttr.setSens(rp_beta_df)

        # Roll the betas so they have the same frequency as the data
        beta_rolled_df = self.rollBeta(periods, fill_method)
        regAttr.setSens(beta_rolled_df)


        # The dependents are derived from those tickers that have sensitivities
        depTickers = regAttr.depTickersFromSensAttrs(sensAttrs)
        depCols = [ (self.attr, t) for t in depTickers ]

        (iCols, dCols) = self.reg.modelCols( self.indCols )

        print("{cls}:attrib_setup: depCols = {depC}, dCols = {dC}".format(cls=type(self), depC=depCols, dC=dCols))
              
        regAttr.depCols = depCols

        # Add a constant column to the data (for the intercept attribution)
        regAttr.addConst((self.attr, "1"), 1)

        

    def attrib(self):
        """
        Run the attribution:
        - attribution of the return (of each column in self.regAttr.depCols)
        - return is attributed to intercept and the independent variable columns (self.indCols)
        - return attribution attributed to a variable is based on the sensitivity of the dependent to that variable (based on the sensitivites set by self.regAttr.setSens)

        -- this is usually preceded by a call to self.attrib_setup
        """
        
        regAttr = self.regAttr

        depCols = regAttr.depCols
        
        indCols_w_intercept = [ c for c in self.indCols ]
        indCols_w_intercept.insert(0, (self.attr, "1"))

        retAttr_df =regAttr.retAttrib(
            indCols_w_intercept,
            depCols
            )

        self.retAttr_df = retAttr_df

        
        

        
