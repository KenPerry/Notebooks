import numpy as np
import pandas as pd

import datetime as dt
import os

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline

import re

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



class GenSelectAttrsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
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
            trans = X.loc[:, idx[self.columns,:] ] .copy()

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
    """ A DataFrame transformer that concatenates this dataframe to the pipeline output
    
    
    Parameters
    ----------
    list_of_dfs : list of DataFrames
        
    """ 
    def __init__(self, list_of_dfs):
        self.list_of_dfs = list_of_dfs
        
    def transform(self, X, **transformparamn): 
        """ Concatenates the DataFrames in the list to df X
        
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

