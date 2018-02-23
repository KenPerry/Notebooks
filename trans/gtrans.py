import numpy as np
import pandas as pd

import datetime as dt
import os
import re

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline

from trans.data import GetData



idx = pd.IndexSlice

class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: [] 
    
    """
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains selected columns of X      
        """
        trans = X[self.columns].copy()

        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self
    



class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    
    Parameters
    ----------
    impute : Boolean, default False
        
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise 
        an array of the form [n_elements, 1]
    
    """
    
    def __init__(self, func, axis=0, impute = False):
        self.func = func
        self.impute = impute
        self.series = pd.Series()
        self.axis   = axis

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame
        
        Parameters
        ----------
        X : DataFrame
            
        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X 
        """
        
        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func, axis=self.axis).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
                
        Returns
        ----------
        self  
        """
        
        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func, axis=self.axis).copy()
        return self
    
    



class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers
    
    Fit several DataFrame transformers and provides a concatenated
    Data Frame
    
    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers
        
    """ 
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        concatted :  pandas DataFrame
        
        """
        
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], axis=1).copy()
        return concatted


    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
        
        Returns
        ----------
        self : object
        """
        
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self

class AddAttrTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that adds an attribute to an DataFrame that has simple columns (i.e., non-MultiIndex columns)
    
    Parameters
    ----------
    dest_attr: string, name of created attribute
    
    """
    
    def __init__(self, dest_attr):
        self.dest_attr = dest_attr


    def transform(self, X, **transform_params):
        """ 
        Adds dest_attr as level 0 of the new MultiIndex for column names
        
        Parameters
        ----------
        X : pandas DataFrame (ignored)
            
        Returns
        ----------
        
        trans : pandas DataFrame that is copy of X but with a MultiIndex for column names

        """

        dest_attr = self.dest_attr
        trans = X.copy()

        # Create tuple for each column: (dest_attr, c)
        cols_w_attr = [ (dest_attr, c) for c in trans.columns] 

        trans.columns = pd.MultiIndex.from_tuples(cols_w_attr)

        return trans

    def fit(self, X, y=None, **fit_params):
        """ 
        Do nothing

        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """

        return self
    

class GenSelectAttrsTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that provides column selection
    
    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.

    Works with DataFrame whose columns are a MultiIndex
    - The columns selected are based on the FIRST level of the MultiIndex
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: [] 
    dropSingle: If True:
        if the DataFrame columns are a MultiIndex, and only a single column is selected: drop level 0 in the columns of the result
    
    """
    def __init__(self, columns, dropSingle=False):
        self.columns    = columns
        self.dropSingle = dropSingle

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains selected columns of X      
        """
            
        if isinstance(X.columns, pd.MultiIndex):
            # NOTE: although we are selecting a sub-set of attributes, ALL the attributes remain, even if they are not used !
            trans = X.loc[:, idx[self.columns,:] ] .copy()

            # Drop the unused attributes
            if pd.__version__ > "0.20.0":
                trans.columns.remove_unused_levels(inplace=True)
            else:
                # Get the active column names (as pairs) and re-create column MultiIndex
                z  = list( zip(trans.columns.get_level_values(0), trans.columns.get_level_values(1)) )
                mi = pd.MultiIndex.from_tuples(z)

                trans.columns = mi
                trans.sortlevel(axis=1, inplace=True)
            

            if (len(self.columns) == 1 and self.dropSingle):
                trans.columns = trans.columns.droplevel(0)
        else:
            trans = X.loc[:, self.columns].copy()
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
                
        
        Returns
        ----------
        self  
        """
        return self


class ShiftTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that shifts the DataFrame
    
    
    Parameters
    ----------
    periods: amount to shift.  df_result.iloc[ i+ periods ] = df.iloc[ i ]

    
    """
    def __init__(self, periods=1):
        self.periods    = periods

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            shifted X
        """
        # Shift the DataFrame
        trans = X.shift( self.periods ).copy()
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
                
        
        Returns
        ----------
        self  
        """
        return self

class GenRenameAttrsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that renames columns
    
    
    Parameters
    ----------
    None
    
    """
    def __init__(self, func, **init_params):
        self.func = func
        self.init_params = init_params
        
        return

    def transform(self, X, **transform_params):
        """ Renames columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains renamed columns of X      
        """

        if self.init_params and "level" in self.init_params:
            if pd.__version__ > "0.20.0":
                trans = X.rename(columns=self.func, **self.init_params).copy()
            else:
                print("transform: pandas version <= 0.20.")
                
                level = self.init_params["level"]
                newColNames = X.columns.levels[level].map(self.func)

                trans = X.copy()
                trans.columns = trans.columns.set_levels(newColNames, **self.init_params)
                
        else:
            trans = X.rename(columns=self.func).copy()
            
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self



class GenPctChangeTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides Percent Change
    
    
    Parameters
    ----------
    None
    
    """
    def __init__(self):
        return

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains Percent Change columns of X      
        """
        
        trans = X.pct_change().copy()
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self



class GenRankTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides Rank
    
    
    Parameters
    ----------
    None
    
    """
    def __init__(self):
        return

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains Rank columns of X      
        """

        def rankFun(s):
            # s_noNa: s with naN's dropped
            # consec_nums: integer range of length size of s_noNa
            # ranks: same size and Index as s_noNa
            s_noNa = s.dropna()
            consec_nums = pd.Series( range(0, s_noNa.size) )

            ranks = pd.Series( np.empty(shape=s_noNa.size), index=s_noNa.index )

            # Assign consecutive integers to positions in the sort order
            sort_idxs = s_noNa.argsort().values
            ranks[ sort_idxs ] = consec_nums

            # Re-index ranks so it has same index as s (NOT s_noNa)
            ranks = ranks.reindex(s.index, fill_value=np.nan)
            
            return ranks
        
        
        trans = X.apply(rankFun, axis=1)
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class GenDataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers
    
    Fit several DataFrame transformers and provides a concatenated
    Data Frame
    
    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers
        
    """ 
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        concatted :  pandas DataFrame
        
        """
        
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                               self.fitted_transformers_], axis=1, keys=self.fitted_transformers_names_).copy()
        return concatted


    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
        
        Returns
        ----------
        self : object
        """
        
        self.fitted_transformers_ = []
        self.fitted_transformers_names_ = []
                
        for (name, transformer) in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
            self.fitted_transformers_names_.append(name)
        return self


class RestrictToCalendarColTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that restricts index to those rows in which the entry in a "calendar column" is non-null
    
    
    Parameters
    ----------
    None
    
    """
    def __init__(self, calCol):
        self.calCol = calCol
        return

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
        """

        calVals = X.loc[:, self.calCol ]
        nonNull = ~calVals.isnull()
        trans = X.loc[ nonNull ]
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class RestrictToNonNullTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that restricts rows to those that are non-null:
    Wrapper around pd.dropna
    
    
    Parameters
    ----------
    how: {"any", "all}
    - "any": eliminate row if ANY element is null
    - "all": eliminate row if ALL elements are null
    
    """
    def __init__(self, how="all"):
        self.how = how
        return

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
        """

        trans = X.dropna(axis=0, how = self.how)
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class FillNullTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that fills null values
    Wrapper around pd.fillna
    
    
    Parameters
    ----------
    value: value or dict (see pd.fillna)
    method: {"ffill", "bfill"}
    - ffill: forward fill
    - bfill: backwards fill
    
    """
    def __init__(self, value=None, method=None):
        self.value = value
        self.method = method
        return

    def transform(self, X, **transform_params):
        """ Fills columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
        """

        trans = X.fillna(value=self.value, method=self.method)
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class DatetimeIndexTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that changes the index to be of type Datetime, given that index is currently a string Date
    
    
    Parameters
    ----------
    dtCol:   name of new index column, containing the Datetime version of the dateCol column
    
    """
    def __init__(self, dtCol):
        self.dtCol = dtCol
        return

    def transform(self, X, **transform_params):
        """ Creates a new index, with name self.dtCol, that is a Datetime version of the current index, which is a string Dte
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
        """

        trans = X.copy()

        # Create column named self.dtCol, which is the Datetime version of the index
        trans[ self.dtCol ] = trans.index.map( lambda  s: pd.to_datetime(s, infer_datetime_format=True))
        trans = trans.set_index( self.dtCol )

        # Always a good idea to sort after altering an index (either axis)
        trans.sortlevel(axis=0, inplace=True)
        trans.sortlevel(axis=1, inplace=True)
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class DataFrameConcat(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that concatenates this dataframe to the pipeline output

    It differs from the Union transformers in that it takes DataFrames, NOT Pipelines, as arguments
    
    
    Parameters
    ----------
    list_of_dfs : list of DataFrames
        
    """ 
    def __init__(self, list_of_dfs, df_keys=None):
        self.list_of_dfs = list_of_dfs
        self.df_keys = df_keys
        
    def transform(self, X, **transformparamn): 
        """
        Concatenates the DataFrames in the list to df X
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        concatted :  pandas DataFrame
        
        """

        list_of_dfs = self.fitted_transformers_

        # Prepend X to list, if it is non-empty
        if (X.shape[0] > 0):
            list_of_dfs.insert(0, X.copy())

        # Concatenate the dataframes
        if self.df_keys:
            concatted = pd.concat(list_of_dfs, axis=1, keys=self.df_keys).copy()
        else:
            concatted = pd.concat(list_of_dfs, axis=1).copy()


        # Always a good idea to sort after altering an index (either axis)
        concatted.sortlevel(axis=0, inplace=True)
        concatted.sortlevel(axis=1, inplace=True)

        return concatted


    def fit(self, X, y=None, **fitparams):
        """ Adds DataFrame Transformers
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
        
        Returns
        ----------
        self : object
        """
        
        self.fitted_transformers_ = []
        for df in self.list_of_dfs:
            self.fitted_transformers_.append(df.copy())
        return self


rankTrans = DataFrameFunctionTransformer(func = lambda s: s.rank(method="first"), axis=1)
pctTrans  = DataFrameFunctionTransformer(func = lambda s: s.pct_change())

class GenRetAttrTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that provides relative changes in an Attribute
    
    Parameters
    ----------
    src_attr:  string, name of attribute containing source data (src)
    period:    integer, amount to shift source data
    src_shift_attr: string, name of created attribute that will contain the source data, shifted by period (src_shift)
    dest_attr: string, name of created attribute that will contain the relative data: (src/src_shift - 1)
    
    """
    
    def __init__(self, src_attr, src_shift_attr, dest_attr, periods=1):
        self.src_attr = src_attr
        self.src_shift_attr = src_shift_attr
        self.dest_attr = dest_attr
        self.periods = periods

    def transform(self, X, **transform_params):
        """ 
        Performs the relative calculation  (src/src_shift -1)
        
        Parameters
        ----------
        X : pandas DataFrame (ignored)
            
        Returns
        ----------
        
        trans : pandas DataFrame containing two attributes: src_shift_attr and dest_attr
          dest_attr contains (src/src_shift -1)

        """
        
        src_df, src_shift_df = self.src_df, self.src_shift_df
        src_shift_attr, dest_attr = self.src_shift_attr, self.dest_attr

        
        dest_df = (src_df/src_shift_df -1)
        concat_pl = DataFrameConcat( [ src_shift_df, dest_df ], df_keys = [ src_shift_attr, dest_attr ] )
        trans = concat_pl.fit_transform( pd.DataFrame() )

        return trans

    def fit(self, X, y=None, **fit_params):
        """ 
        Store source data (src_df) and source data shifted by period (src_shift_df) in self
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """

        src_attr = self.src_attr
        
        src_pl       = make_pipeline( GenSelectAttrsTransformer ( [ src_attr ],
                                                                  dropSingle=True
                                                                  )
                                      )

        src_shift_pl = make_pipeline( GenSelectAttrsTransformer ( [ src_attr ],
                                                                  dropSingle=True
                                                                  ),
                                      ShiftTransformer( self.periods )
                                      )


        src_df       = src_pl.fit_transform(X)
        src_shift_df = src_shift_pl.fit_transform( X )

        self.src_df   = src_df
        self.src_shift_df = src_shift_df
        
        return self
    



class GenRankEndOfPeriodAttrTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that provides a ranking -- at each end-of-period -- across an attribute
    e.g., rank cross-section of monthly returns
    
    Parameters
    ----------
    src_attr:  string, name of attribute containing source data that will be used in ranking
    univ:      array,  names of columns (level 1) that are to be included in the ranking universe
    dest_attr: string, name of created attribute that will contain the ranks to use
    eop_df:    DataFrame.  Frequency is only end of ranking period (e.g., end of month).  Ranking is done according to eop_df.loc[:, src_attr]

    NOTE: Ranks are computed at the date corresponding to end of period.  However, the computed ranks are applied FORWARD, to each DAY of the following period
          e.g., the rank computed at end of month M is used for each day of month M+1
    
    """
    
    def __init__(self, eop_df, src_attr, univ, dest_attr):
        self.eop_df = eop_df
        self.univ = univ
        self.src_attr = src_attr
        self.dest_attr = dest_attr

    def transform(self, X, **transform_params):
        """ 
        Takes the prior period rank and applies it forward to each day of following period
        
        Parameters
        ----------
        X : pandas DataFrame (only X.index is used: to give the result the same index as X)
            
        Returns
        ----------
        
        trans : pandas DataFrame containing two attributes: src_shift_attr and dest_attr
          dest_attr contains (src/src_shift -1)

        """

        dest_attr = self.dest_attr
        rank_df = self.rank_df

        # The ranks computed at end of period are used for the FOLLOWING period. Shift forward one period
        shift_pl = make_pipeline( ShiftTransformer(1) )
        rank_shift_df = shift_pl.fit_transform( rank_df )

        # Convert the end of period frequency to the same index as X (usually, at daily frequency instead of end of period frequency)
        rank_shift_daily_df = pd.concat(  [rank_shift_df,  pd.DataFrame(index=X.index) ], axis=1 )

        # Always a good idea to sort after altering an index (either axis)
        rank_shift_daily_df.sortlevel(axis=0, inplace=True)
        rank_shift_daily_df.sortlevel(axis=1, inplace=True)


        # We now have the ranks, pushed forward to end of following period.  Backfill daily for the entire month. Result attribute is dest_attr
        bfill_pl = make_pipeline( FillNullTransformer(method="bfill"),
                                  AddAttrTransformer( dest_attr )
                                  )

        rank_shift_daily_df = bfill_pl.fit_transform( rank_shift_daily_df )

        trans = rank_shift_daily_df

        return trans

    def fit(self, X, y=None, **fit_params):
        """ 
        Computes the end of period rank (of the src_attr, over the universe univ), using the end of period DataFrame eop_df
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """

        univ = self.univ
        src_attr = self.src_attr
        dest_attr = self.dest_attr
        eop_df   = self.eop_df
        

        # Rank the end of period dataframe, using the src_attr
        # For ranking: select only the attribute src_attr of the tickers in the universe univ
        rank_pl = make_pipeline( GenSelectAttrsTransformer( [ src_attr ], dropSingle=True ),
                                 SelectColumnsTransformer(univ),
                                 rankTrans
                                 )

        ## The alternate way of selecting only the universe columns for src_attr is:
        ##   univ_cols = [ (src_attr, t) for t in univ ]
        ##   SelectColumnsTransformer( univ_cols )

        rank_df = rank_pl.fit_transform( eop_df )
        self.rank_df = rank_df
        
        return self
    



class GenRankToPortRetTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that computes a portfolio return based on ranks

    We will compute a weights attribute wt_attr, based on the ranks (rank_attr of the rank_df DataFrame: rank_df.loc[:, idx[rank_attr,:] ])
    The weights will multiply the data (src_attr of the transform source DataFrame X: X.loc[:, idx[src_attr,:] ] )
    The portfolio return (weighted sum of products) will be stored in column dest_col with attribute src_attr

    The weights are computed from the ranks by the function rank_to_wt_func,
    e.g., lambda s: (s >= hi_rank) * 1.0 + (s <= lo_rank) * -1.0
    
    Parameters
    ----------
    src_attr: string, the source attribute in transform source DataFrame X

    rank_df:  DataFrame, containing the ranks.  Level 1 columns of src_df and rank_df should be consistent
    rank_attr:string, attribute of the rank, in rank_df, i.e., rank is rank_df[:, idx[rank_attr,:] ]

    wt_attr:  string, the name of the new "weight" atttribute that will be created
    dest_col:string,  the name of the column that will be added (with attribute src_attr) that contains the portfolio return

    Returns:
    trans: DataFrame with attributes (wt_attr, src_attr)
      trans[:, idx[wt_attr,;]]  will contain the weights, based on ranks
      trans[:, idx[src_attr,:]] will contain the weighted returns, plus the sum of the weighted returns
        trans[:, idx[src_attr, dest_col] ]: simple sum of weighted returns.  Note the weights don't necessary sum up to anything in particular (e.g., neither 100 nor 0)
        trans[:, idx[src_attr, dest_col + " > 0"] ]: sum of weighted returns, where weights are > 0, divided by sum(weights > 0). This is a true return for an equally weighted long portfolio
        trans[:, idx[src_attr, dest_col + " < 0"] ]: sum of weighted returns, where weights are < 0, divided by sum(weights < 0). This is a true return for an equally weighted short portfolio
        trans[:, idx[src_attr, dest_col + " net"] ]: sum of the long and short returns = trans[:, idx[src_attr, dest_col + " > 0"] ] + trans[:, idx[src_attr, dest_col + " < 0"] ]
          This is the difference between a long portfolio worth 1 and short portfoli worth 1


 at trans[:, idx[src_attr, dest_col]]
      trans[:, idx[src_attr,:]] will contain the weighted returns, plus the sum of the weighted returns at trans[:, idx[src_attr, dest_col]]
    
    """
    
    def __init__(self, src_attr, rank_df, rank_attr, rank_to_wt_func, wt_attr, dest_col):
        self.src_attr = src_attr
        self.rank_df, self.rank_attr, self.rank_to_wt_func = rank_df, rank_attr, rank_to_wt_func
        
        self.wt_attr = wt_attr
        self.dest_col = dest_col

    def transform(self, X, **transform_params):
        """ 
        Takes the prior period rank and applies it forward to each day of following period
        
        Parameters
        ----------
        X : pandas DataFrame (ignored)
            
        Returns
        ----------
        
        trans : DataFrame
        wt

        """

        src_attr, dest_col =  self.src_attr, self.dest_col
        wt_attr = self.wt_attr
        wt_df = self.wt_df

        # Select the source attribute src_attr from the source DataFrame src_df
        ret_pl = make_pipeline( GenSelectAttrsTransformer( [ src_attr ], dropSingle=True ) )
        ret_df = ret_pl.fit_transform( X )

        # Compute number of positively/negatively weighted items
        pos_cnt_series = (wt_df > 0).sum(axis=1)
        neg_cnt_series = (wt_df < 0).sum(axis=1)

        # Compute weighted returns
        wt_ret_df  = wt_df * ret_df
        sum_series =  wt_ret_df.sum(axis=1)

        # NOTE: Number of positively and negatively weighted items may not be the same, so net_ret_series NOT necessarily same as sum_series
        pos_ret_series = ((wt_df > 0) * wt_ret_df).sum(axis=1)/pos_cnt_series
        neg_ret_series = ((wt_df < 0) * wt_ret_df).sum(axis=1)/neg_cnt_series

        # Net is addition (since neg_ret_series has negative weights)
        net_ret_series = pos_ret_series + neg_ret_series


        # Add the Portfolio Return to the weighted returns df
        wt_ret_df.loc[:, dest_col ] = sum_series
        wt_ret_df.loc[:, dest_col + " > 0"] = pos_ret_series
        wt_ret_df.loc[:, dest_col + " < 0"] = neg_ret_series
        wt_ret_df.loc[:, dest_col + " net"] = net_ret_series

        # Return df:
        trans_pl = DataFrameConcat( [ wt_df, wt_ret_df ], df_keys=[ wt_attr, src_attr ] )
        trans    = trans_pl.fit_transform( pd.DataFrame() )
        
        return trans

    def fit(self, X, y=None, **fit_params):
        """ 
        Computes the end of "weight" attribute, based on the rank_attr of the rank_df
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """

        rank_df, rank_attr, rank_to_wt_func = self.rank_df, self.rank_attr, self.rank_to_wt_func
        
        hi_rank, lo_rank = 5, 1
        # hmlWtTrans = DataFrameFunctionTransformer(func = lambda s: (s >= hi_rank) * 1.0 + (s <= lo_rank) * -1.0)
        hmlWtTrans = DataFrameFunctionTransformer(func = rank_to_wt_func)
        
        wt_pl = make_pipeline( GenSelectAttrsTransformer( [ rank_attr ], dropSingle=True),
                               hmlWtTrans
                               )

        wt_df = wt_pl.fit_transform(rank_df)

        self.wt_df = wt_df
        
        return self
    
class GetDataTransformer(BaseEstimator, TransformerMixin):
    """ 
    A DataFrame transformer that gets raw data
    
    Parameters
    ----------
    tickers: list of tickers to get
    cal_ticker: ticker whose dates will be used as the common "calendar"

    Returns:
    trans: DataFrame, with index the same as cal_ticker

    
    """
    
    def __init__(self, tickers, cal_ticker=None):
        # Use GetData to obtain the data
        gd = GetData()
        self.gd = gd
        
        # Default for tickers are all existing files
        if not tickers:
            tickers = gd.existing()

        # Make sure cal_ticker is in tickers
        if not cal_ticker in tickers:
            tickers.insert(0, cal_ticker)
            
        self.tickers, self.cal_ticker = tickers, cal_ticker


    def transform(self, X, **transform_params):
        """ 
        Gets the data for all tickers, converts index to DateTime, restricts index to those dates present in cal_ticker
        
        Parameters
        ----------
        X : pandas DataFrame (ignored)
            
        Returns
        ----------
        
        trans : DataFrame

        """

        cal_ticker =self.cal_ticker

        # Pipeline to convert index to DateTime and restrict it to the dates that cal_ticker has data
        pipe_d = make_pipeline( DatetimeIndexTransformer("Dt"),
                                RestrictToCalendarColTransformer( ("Adj Close", cal_ticker)),
                                RestrictToNonNullTransformer("all")
                                )

        trans = pipe_d.fit_transform( self.df )
                
        return trans
        
 
    def fit(self, X, y=None, **fit_params):
        """ 
        Get the data for the tickers in self.tickers
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """

        gd = self.gd
        df = gd.combine_data( self.tickers )

        self.df = df
        
        return self
    
