from datetime import timedelta


from trans.gtrans import *
from trans.date_manip import Date_Manipulator


"""
Base class for QuantFactor

A QuantFactor is:
- defined over a universe of tickers
- is a long/short portfolio over this universe
- the longs and shorts are determined by an attribute which is used to rank remembers of the universe;  the lowest ranked names are shorted, the highest ranked names are held long

The essence of this framework is a derived method "fit" which actually performs the ranking.  Many factors will share a common work-flow (and code) with the exception of the method "fit"
that is specialized per factor.

Simplified workflow:
--------------------
- Set universe and dataprovider (for getting price data for members of universe)
  - qf = QuantFactor(universe=universe, dataProvider=dataprovider)

- Do all steps of detailed workflow in standard way
  - qf.create(start=s, end=e,
               price_attr="Adj Close", ret_attr="Ret", rank_attr="Factor",
               window=timedelta(days=30)
             )

Work flow:
----------
- Instantiate
  - qf = QuantFactor(universe=universe, dataProvider=odr)

  Define the universe of tickers.
  Give a DataProvider instance that will be used to retrieve prices for all members of the universe

- Get daily prices (needed to compute daily returns, which are used to create daily returns for factor portfolio
  - qf.load_prices(start=start, end=end_

  Load daily prices series of members of the universe, with date ranging from start to end, using the DataProvider passed during instantiation of the QuantFactor object

  
- Compute daily returns from prices
  - qf.create_dailyReturns(price_attr, ret_attr)

  Compute the daily returns of each member of the universe, using the price level attribute price_attr and creating a return attribute named ret_attr

- Set dates on which to compute the attributes on which members of the universe will be ranked.
  -  qf.set_endDates( endDates )

  The universe members are ranked only on endDates (e.g., end of each month).  The ranks are used to create the long/short porfolio for dates following the ranking date.
  For example, to rank on last day of each month:

    dm = Date_Manipulator( v.price_df.index )
    eom_in_idx = dm.periodic_in_idx_end_of_month(end)
    qf.set_endDates( eom_in_idx )

- Create DataFrame with a attribute on which to rank (low-frequency: computed only for qf.endDates)
  - qf.create_period_attr(df, start, end, windowTimeDelta, attr_name)

  Create a DataFrame limited to the ranking dates (i.e., qf.endDates) and, for each member of the universe on each ranking date, compute an attribute (named attr_name) to be used for ranking.
  The DataFrame is created by calling the derived method (which must be overridden) "fit", called as:
    self.fit(df, s, e)

  The fit method takes the passed DataFrame df, and dates e (which is an element of endDates) and s (s = e - windowTimeDelta + oneDayTimeDelta).
  This allows the ranking on date e to be a function of the DataFrame df over the date rnage s to e, i.e., df.loc[ s:e, ]

  So, for example:
  - fit can be used with a df containing prices to compute a period return over the preceding windowTimeDelta days ending on e (i.e., a momentum factor)
  - fit can be used with a df containing returns to computing the standard deviation over the preceding windowTimeDelta days ending on e (i.e, a volatility factor)


= Create ranks: on each date in f.endDates: create a cross-sectional rank using the prevously created DataFrame with ranking attribute
  - qf.create_ranks()

  Create a DataFrame with daily frequency in which all members of the universe are ranked.
  The ranking itself is performed only on endDates but those (low-frequency) ranks are pushed forward to each day of the subsequent ranking period, e.g.,
  the ranks computed at endDates[i] (e.g., end of month M) are pushed forward to each day between (enddates[i]+1) and endDates[i+1] (e.g., end of month M+1)

- Create factor (returns of factor portfolio)
  - qf.create_factor()

  Create a DataFrame with daily frequency with the long/short factor portfolio.
  On each day, using the ranks for that day, the lowest ranked names are held short and the highest ranked names are held long.
  The Long/short portfolios value for the day is formed by multiplying the daily returns (computed by qf.create_dailyReturns) by weights (-1, 0, or +1) derived from the ranks (i.e., hold short, don't hold, hold long)

"""

class QuantFactorBase:
    def __init__(self, universe=None, debug=False, **params):
        """
        Create an instance, over a universe of tickers

        Parameters
        ----------
        universe: List, list of tickers in the universe
        """
        
        self.universe = universe

        self.Debug = debug

        if "dataProvider" in params:
            dp = params["dataProvider"]
            print("{cls}:__init__: dataProvider passed".format(cls=type(self)))
        else:
            dp = GetData()

        self.dp = dp

    def load_prices(self, start=None, end=None, cal="SPY"):
        """
        Create a DataFrame with prices of tickers in the universe self.universe
        
        Parameters
        ----------
        cal: String. Ticker (usually NOT in universe) to use as a calendar
        """

        dp = self.dp
        
        self.cal = cal

        # Get the data for the tickers in self.universe
        price_df = GetDataProviderTransformer(self.universe, cal_ticker=cal, dataProvider=dp).fit_transform( pd.DataFrame())

        # Limit the output to date range from start to end
        if (start):
            price_df = price_df.loc[ start:, ]
        if (end):
            price_df = price_df.loc[ :end, ]
        
        self.price_df = price_df

        return price_df

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


    def create_period_attr(self, df, start, end, windowTimeDelta, attr_name):
        """
        Create a DataFrame limited to the ranking dates (i.e., self.endDates) and, for each member of the universe on each ranking date, compute an attribute (named attr_name) to be used for ranking.
        The DataFrame is created by calling the derived method (which must be overridden) "fit", called as:
        self.fit(df, start, end)

        The fit method takes the passed DataFrame df, and dates e (which is an element of endDates) and s (s = e - windowTimeDelta + oneDayTimeDelta).
        This allows the ranking on date e to be a function of the DataFrame df over the date rnage s to e, i.e., df.loc[ s:e, ]


        Parameters
        ----------
        df:         DataFrame with high-frequency index
        start, end: datetimes for start and end
        windowTimeDelta: timedelta.  For each date e in self.endDates, compute the rank for date e using slice of df from (s = e - windowTimeDelta + oneDayTimeDelta) to e
        attr_name: string. Name of attribute created.
        
        Creates attributes
        ------------------
        period_df: DataFrame with low-frequency index
        
        """

        self.daily_df =df

        oneDayTimeDelta = timedelta(days=1)
        colNames =  [ "Dt" ] + df.columns.get_level_values(1).tolist()
        colNames = pd.MultiIndex.from_tuples( [ (attr_name, c) for c in df.columns.get_level_values(1).tolist() ] )
        
        results = []

        endDates = self.endDates
        
        # Produce a ranking attribute for each date in endDates
        for e in endDates:
            s = e - windowTimeDelta + oneDayTimeDelta
            result = None
            
            if (s >= start):
                result  = self.fit(df, s, e)

            results.append( result )

        # Build DataFrame from the rows
        period_df = pd.DataFrame( results, columns= colNames, index=endDates)
        period_df.index.rename("Dt", inplace=True)
        period_df = period_df.sort_index()

        self.period_df = period_df
        self.period_attr = attr_name

        return period_df

    def fit(self, attr_name):
        print("fit: need to override")


    def set_endDates(self, endDates=[]):
        """
        Set the dates on which ranking will be performed.
        NOTE: endDates MUST be in self.price_df.index
        """
            
        self.endDates = endDates

    def create_ranks(self, rank_attr="Rank"):
        """
        Create cross-sectional ranks on each date in self.endDates, and apply them as FORWARD daily ranks in the subsequent period
        - i.e., rank period_df.loc[p, idx[ self.period_attr, :] ], for end-of-period date p
        - apply these ranks forward on a daily basis, to the next period (p+1)_: assign these ranks to daily_ret_df[ f:g, ; ] for (f,g) being the daily dates in period (p+1)

        The ranking 

        Paramters:
        ----------
        rank_attr: sring.  Is the name of the Rank attribute in the created DataFrame
        """
        
        universe = self.universe

        daily_ret_df = self.daily_ret_df
        daily_ret_attr = self.daily_ret_attr

        period_df = self.period_df
        period_attr = self.period_attr

        # Get the tickers for which the daily_ret_df has returns
        rank_univ = daily_ret_df.loc[:, idx[daily_ret_attr,:]].columns.get_level_values(1).unique().tolist()

        # Exclude calendar ticker from ranking if it is not in self. universe
        if (not self.cal in set(universe)):
            rank_univ = list( set(rank_univ) - set( [self.cal ]) )

        self.rank_attr = rank_attr
        
        # Create the ranks
        # n.b., the ranks are based on period returns (period_attr of the period_df) but the period ranks are pushed forward (ffill) into the daily daily_ret_df
        ##      so next_period_rank_df is of same frequency as daily_ret_df (e.g., daily), and the rank for day d is the rank for the period end date preceding d
        ##      i.e., the rank is computed based on period returns as of the preceding end date, and then pushed forward daily into the next period
        next_period_rank_pl = GenRankEndOfPeriodAttrTransformer(
            period_df,
            period_attr,
            rank_univ,
            rank_attr
        )

        # n.b., period_df is used for the ranking, but the ranks are pushed forward with the same frequency as daily_ret_df (the only part of daily_ret_df that is used is it's index !)
        next_period_rank_df = next_period_rank_pl.fit_transform( daily_ret_df )

        self.daily_rank_df = next_period_rank_df

        return next_period_rank_df

    def create_factor(self, **params):
        """
        Create a DataFrame with daily frequency with the long/short factor portfolio.
        On each day, using the ranks for that day, the lowest ranked names are held short and the highest ranked names are held long.
        The Long/short portfolios value for the day is formed by multiplying the daily returns (computed by qf.create_dailyReturns) by weights (-1, 0, or +1) derived from the ranks (i.e., hold short, don't hold, hold long)
        """
        
        daily_ret_df = self.daily_ret_df
        daily_ret_attr = self.daily_ret_attr

        daily_rank_df = self.daily_rank_df
        rank_attr     = self.rank_attr

        if "factor_attr" in params:
            factor_attr = params[ "factor_attr" ]
        else:
            factor_attr = daily_ret_attr + " Factor"

        # Define the lower and upper fractions (of the defined ranks) for the short/long sub-portfolios
        if "pct" in params:
            frac = params[ "pct" ] * .01
        else:
            frac = 0.20
            
        wt_attr = "weight"
        portret_col = "Port"

        # Define a function that assigns -1 to low ranks and +1 to hi ranks
        def rank_func(s):
            # Take upper/lower tail of DEFINED ranks
            s_defined = s[ s.isnull() == False ]
            size = s_defined.size

            # Make sure there are equal numbers in the low and high buckets
            lo_rank = int(frac * size)
            hi_rank = size - lo_rank + 1
            
            return (s <= lo_rank) * -1 + (s >= hi_rank)*1
        
        portret_pl = make_pipeline(  GenRankToPortRetTransformer(
            daily_ret_attr,
            daily_rank_df,
            rank_attr,
            rank_func,
            wt_attr,
            portret_col
        )
                          )

        port_df = portret_pl.fit_transform( daily_ret_df )
        self.factor_df = port_df

        return port_df
        
    def create(self,
               start=None, end=None,
               price_attr=None, ret_attr=None, rank_attr=None,
               window=None
    ):
        

        # Get daily prices for members of universe
        daily_price_df = self.load_prices(start=start, end=end)

        # Create daily returns from prices
        daily_ret_df = self.create_dailyReturns(price_attr, ret_attr )

        # Create end-of-month as endDates
        dm = Date_Manipulator( self.price_df.index )
        eom_in_idx = dm.periodic_in_idx_end_of_month(end)
        self.set_endDates( eom_in_idx )

        # Create attribute on which to rank.  Created only for dates in self.endDates
        self.create_period_attr( self.daily_ret_df.loc[:, idx[ret_attr,:]], start, end, window, rank_attr)

        # Create ranks for each date e in self.endDates but push them forward in time so they have same frequency as self.daily_ret_df
        daily_rank_df = self.create_ranks()

        # Create factor returns
        factor_df = self.create_factor()

        return factor_df
    