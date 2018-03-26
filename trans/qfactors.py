import numpy as np
import pandas as pd

import datetime as dt
import os
import re

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline

from trans.data import GetData
from trans.gtrans import *


idx = pd.IndexSlice

class MomentumPipe:
    def __init__(self, universe, **params):
        """
        Parameters
        ----------
        universe: List, list of tickers in the universe
        """
        
        self.universe = universe

    def load_prices(self, start=None, end=None, cal="SPY"):
        """
        Create a DataFrame with prices of tickers in the universe self.universe
        
        Parameters
        ----------
        cal: String. Ticker (usually NOT in universe) to use as a calendar
        """

        self.cal = cal

        # Get the data for the tickers in self.universe
        price_df = GetDataTransformer(self.universe, cal_ticker=cal).fit_transform( pd.DataFrame())

        # Limit the output to date range from start to end
        if (start):
            price_df = price_df.loc[ start:, ]
        if (end):
            price_df = price_df.loc[ :end, ]
        
        self.price_df = price_df

        return price_df
        
    def set_endDates(self, endDates=[]):
        """
        Create a period DataFrame (self.period_df) containing prices only with dates in endDates
        NOTE: endDates MUST be in self.price_df.index
        """
            
        self.endDates = endDates
        price_df = self.price_df
            
        period_df = price_df.loc[ endDates,: ]
        self.period_df = period_df

    def create_dailyReturns(self, price_attr, ret_attr, price_shifted_attr=None, periods=1 ):
        """
        Create daily returns from self.price_df
        Needed to create the factor portfolio returns
        """
        
        price_df    = self.price_df

        if (not price_shifted_attr):
            price_shifted_attr = price_attr + " prior"

        daily_ret_pl = GenRetAttrTransformer( price_attr, price_shifted_attr, ret_attr, 1 )
        daily_ret_df = daily_ret_pl.fit_transform( price_df )

        self.daily_ret_df = daily_ret_df
        self.daily_ret_attr = ret_attr

        return daily_ret_df


    def create_periodReturns(self, price_attr, ret_attr, price_shifted_attr=None,  periods=1 ):
        """
        Create period returns from self.period_df
        Needed to create ranking, to determine weights for factor portfolio return construction
        """

        period_df = self.period_df

        if (not price_shifted_attr):
            price_shifted_attr = price_attr + " prior"

        period_ret_pl = GenRetAttrTransformer( price_attr, price_shifted_attr, ret_attr, periods )
        period_ret_df = period_ret_pl.fit_transform( period_df )

        self.period_ret_df = period_ret_df
        self.period_ret_attr = ret_attr

        return period_ret_df
        

    def create_ranks(self, rank_attr="Rank"):
        """
        Create cross-sectional ranks of the period returns, and apply them as FORWARD daily ranks in the subsequent period
        - i.e., rank period_ret_df.loc[p, idx[ self.period_ret_attr, :] ], for end-of-period date p
        - apply these ranks forward on a daily basis, to the next period (p+1)_: assign these ranks to daily_ret_df[ f:g, ; ] for (f,g) being the daily dates in period (p+1)

        The ranking 
        rank_attr  is the Rank attribute in the next_period_rank_df
        """
        universe = self.universe

        daily_ret_df = self.daily_ret_df
        daily_ret_attr = self.daily_ret_attr

        period_ret_df = self.period_ret_df
        period_ret_attr = self.period_ret_attr

        # Get the tickers for which the daily_ret_df has returns
        rank_univ = daily_ret_df.loc[:, idx[daily_ret_attr,:]].columns.get_level_values(1).unique().tolist()

        # Exclude calendar ticker from ranking if it is not in self. universe
        if (not self.cal in set(universe)):
            rank_univ = list( set(rank_univ) - set( [self.cal ]) )

        self.rank_attr = rank_attr
        
        # Create the ranks
        # n.b., the ranks are based on period returns (period_ret_attr of the period_ret_df) but the period ranks are pushed forward (ffill) into the daily daily_ret_df
        ##      so next_period_rank_df is of daily frequency, and the rank for day d is the rank for the period end date preceding d
        ##      i.e., the rank is computed based on period returns as of the preceding end date, and then pushed forward daily into the next period
        next_period_rank_pl = GenRankEndOfPeriodAttrTransformer(
            period_ret_df,
            period_ret_attr,
            rank_univ,
            rank_attr
        )
        
        next_period_rank_df = next_period_rank_pl.fit_transform( daily_ret_df )

        self.daily_rank_df = next_period_rank_df

        return next_period_rank_df

    def create_factor(self, factor_ret_attr=None):
        daily_ret_df = self.daily_ret_df
        daily_ret_attr = self.daily_ret_attr

        daily_rank_df = self.daily_rank_df
        rank_attr     = self.rank_attr

        if (not factor_ret_attr):
            factor_ret_attr = daily_ret_attr + " Factor"
            
        wt_attr = "weight"
        portret_col = "Port"

        # Define a function that assigns -1 to low ranks and +1 to hi ranks
        def rank_func(s):
            size = s.size
            lo_rank, hi_rank = .2 * size, .8 * size
            return ( (s <= lo_rank) * -1 + (s > hi_rank)*1 )
        
        portret_pl = make_pipeline(  GenRankToPortRetTransformer(
            daily_ret_attr,
            daily_rank_df,
            rank_attr,
            rank_func,
            wt_attr,
            portret_col
        )
                          )

        port_ret_df = portret_pl.fit_transform( daily_ret_df )
        self.factor_ret_df = port_ret_df

        return port_ret_df

