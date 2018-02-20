
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# From https://pythonprogramming.net/combining-stock-prices-into-one-dataframe-python-programming-for-finance/

# In[ ]:




# In[2]:

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web

import numpy as np

import pickle
import requests

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline

import re

idx = pd.IndexSlice


# In[3]:

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers


# In[4]:

def get_data_from_yahoo(reload_sp500=False):
    
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
            
    tickers.remove('BRK.B')
    tickers.remove('BF.B')
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2017, 12, 31)
    
    for ticker in tickers:

        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


# In[5]:

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')



# In[191]:

sp500_tickers = save_sp500_tickers()


# In[114]:

import re
[ ticker for ticker in sp500_tickers if re.match('.*\.B$', ticker)]


# In[47]:

sp500_tickers


# In[6]:

get_data_from_yahoo()


# In[193]:

compile_data()


# In[6]:

df = pd.read_csv('sp500_joined_closes.csv')


# In[7]:

df.shape


# In[8]:

df.columns


# In[7]:

def compile_data_all(tickers=None):
    if (tickers is None):
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    main_df = pd.DataFrame()
    dfs = []
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        
        dfs.append(df)

        if count % 10 == 0:
            print(count)
            
    df_big = pd.concat( dfs, axis=1, keys=tickers)
    df_big.index.name = "Date"
    
    return df_big



# In[ ]:

dfA = pd.read_csv('stock_dfs/{}.csv'.format('AAPL'))
dfA = dfA.set_index('Date')
dfB = pd.read_csv('stock_dfs/{}.csv'.format('GOOG'))
dfB = dfB.set_index('Date')


# In[54]:

data = dict((sym, web.DataReader(sym, "yahoo"))
          for sym in ['AAPL', 'GOOG'])


# In[58]:

dfA = data['AAPL']
dfB = data[ 'GOOG']


# In[60]:

pd.concat( [ dfA, dfB], axis=1, keys=['A', 'B'], names=["ticker", "attr"]).head()
                
                  


# In[8]:

all = compile_data_all()


# In[12]:

with open("sp500_allAttrs.pickle","wb") as fp:
        pickle.dump(all,fp)


# In[14]:

with open("sp500_allAttrs.pickle", "rb") as fp:
    all = pickle.load(fp)


# In[15]:

all.shape
all.columns.values
all.columns.get_level_values(0).unique()
all.columns.get_level_values(1).unique()


# In[ ]:




# In[84]:

all.tail()


# In[81]:

all.shape
all.columns
all.index
all.tail()


# In[90]:

all.loc['2017-12-22': '2017-12-27']


# In[97]:

all.loc['2017-12-22': '2017-12-27', idx[:,['Close', 'Adj Close'] ] ]


# In[140]:

a = all.loc['2017-12-22': '2017-12-27', idx['AAPL',['Close', 'Adj Close'] ] ]
b = all.loc['2017-12-22': '2017-12-27', idx['GOOG',['Close', 'Adj Close'] ] ]
a
b
c = pd.concat([a,b], axis=1)
c
c.columns

d = all.loc['2017-12-22': '2017-12-27', idx[:,['Close', 'Adj Close'] ] ]
d
d.columns


# In[21]:

singleAttr = all.loc['2017-12-22': '2017-12-27', idx[['AAPL', 'GOOG'], ['Adj Close'] ] ]
singleAttr
type(singleAttr.columns)

singleAttr.columns = singleAttr.columns.droplevel(1)
singleAttr
type(singleAttr.columns)


# In[98]:

all.tail(10).to_csv('all.csv')


# In[99]:

pd.read_csv('all.csv')


# In[16]:

class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):
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
    


# In[17]:

class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    
    Parameters
    ----------
    impute : Boolean, default False
        
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise 
        an array of the form [n_elements, 1]
    
    """
    
    def __init__(self, func, impute = False):
        self.func = func
        self.impute = impute
        self.series = pd.Series() 

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
            trans = pd.DataFrame(X).apply(self.func).copy()
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
            self.series = pd.DataFrame(X).apply(self.func).copy()
        return self
    
    


# In[18]:

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


# In[68]:

class GenSelectAttrTransfomer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: [] 
    
    """
    def __init__(self, columns):
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
        colType = type(X.columns).__name__
        if (re.match('MultiIndex$', colType)):
            trans = X.loc[:, idx[:, self.columns] ] .copy()
            if (len(self.columns) == 1):
                trans.columns = trans.columns.droplevel(1)
        else:
            trans = X.loc[:, self.column].copy()
        
        return trans

    def fit(self, X, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
                
        
        Returns
        ----------
        self  
        """
        return self


# In[82]:

class GenPctChangeTransfomer(BaseEstimator, TransformerMixin):
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


# In[83]:

single_pipe = make_pipeline(GenSelectAttrTransfomer(['Adj Close'] ),
                            GenPctChangeTransfomer()
                           )
s = single_pipe.fit_transform(all)
s.tail()


# In[72]:

s.loc['2017-12-22':'2017-12-27'].pct_change().rank(axis=1)


# In[66]:

s.loc['2017-12-22':'2017-12-27'].rank(axis=1)


# In[120]:

singleAttr.head()


# In[124]:

singleAttr.apply(np.log, axis=0)


# In[126]:

ticker_pipeline = make_pipeline(  
        SelectColumnsTransfomer(['GOOG'])
)


# In[134]:

ticker_pipeline.fit_transform(singleAttr).pct_change()


# In[133]:

p_1 = make_pipeline( DataFrameFunctionTransformer(func = lambda x: x.pct_change()) )
p_1.fit_transform(singleAttr)


# In[136]:

p_2 = make_pipeline( DataFrameFunctionTransformer( func = lambda df: df.pct_change()))
p_2.fit_transform(all['2017-12-22': '2017-12-27'])

