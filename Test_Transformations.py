
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

import pandas as pd
idx = pd.IndexSlice

from sklearn.pipeline import Pipeline, make_pipeline


# In[3]:

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')


# In[4]:

from trans.data import GetData as gd
from trans.gtrans import *


# In[ ]:

sp500 = gd.save_sp500_tickers()


# In[ ]:

len(sp500)


# In[ ]:

all = gd.compile_data_all(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG'])


# In[ ]:

all.loc['2017-12-22':'2017-12-27']


# In[ ]:

gd.save_data(all, "test_all.pkl")


# In[5]:

all = gd.load_data("test_all.pkl")


# In[6]:

ids = gd.get_data_from_yahoo2([ 'SPY'])
ids_all = gd.compile_data_all(['SPY'])
gd.save_data(ids_all, "indexes_all.pkl")


# In[6]:

ids_all = gd.load_data("indexes_all.pkl")


# ## Rank: if there are n items, the sum of positions is sum from 1 to n, equals (n/2 * (1+n)).  The sum of the ranks obeys this too, with the caveat that multiple items having the same value have the same fractional rank, e.g., if there are 3 items, they will have the same rank r such tat 3*r = sum of positions of the three elements in the list

# In[19]:

f = pd.DataFrame({ 'A': [ 20, 10, 20, 20], 'B': [20, 30, 10, 10]})
f
f.rank()
f.rank().astype(int)
f.apply(np.argsort)


# In[8]:

from trans.gtrans import DataFrameFunctionTransformer 

pctTrans = DataFrameFunctionTransformer(func = lambda s: s.pct_change())
rankTrans = DataFrameFunctionTransformer(pd.Series.rank, axis=1)


# In[64]:

from trans.gtrans import GenSelectAttrTransformer
from trans.gtrans import GenRenameAttrTransformer
from trans.gtrans import GenRankTransformer

pipe_1 = make_pipeline(GenSelectAttrTransformer(['Adj Close'] ), 
                       pctTrans,
                       GenRenameAttrTransformer(lambda col: col + ' pct')
                      )

pipe_2 = make_pipeline(GenSelectAttrTransformer(['Adj Close'] ), 
                       pctTrans, 
                       # rankTrans,
                       GenRankTransformer(),
                       GenRenameAttrTransformer(lambda col: col + ' rank')
                          )

featU = DataFrameFeatureUnion([ pipe_1, pipe_2 ])
u = featU.fit_transform(all.head())
u.head()
                    


# In[65]:

from trans.gtrans import GenDataFrameFeatureUnion

pipe_1 = make_pipeline(GenSelectAttrTransformer(['Adj Close'] ), 
                       pctTrans,
                       # GenRenameAttrTransformer(lambda col: col + ' pct')
                      )
pipe_2 = make_pipeline(GenSelectAttrTransformer(['Adj Close'] ), 
                       pctTrans,
                       # DataFrameFunctionTransformer(pd.Series.rank, axis=1)
                        GenRankTransformer(),
                       # GenRenameAttrTransformer(lambda col: col + ' rank')
                      )

featUn =  GenDataFrameFeatureUnion( [ ("Pct", pipe_1), ("Rank", pipe_2)])
un = featUn.fit_transform(all['2000-01-19':'2000-01-27'])
un.tail()


# In[ ]:

u.columns
u.tail().loc[:, 'FB']

