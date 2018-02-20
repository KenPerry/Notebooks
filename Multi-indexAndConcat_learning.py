
# coding: utf-8

# In[1]:

from pandas.io.data import DataReader
# from pandas_datareader import data, wb
import pandas as pd
import numpy as np


# In[2]:

symbols = ['MSFT', 'GOOG', 'AAPL']

import datetime
start = datetime.datetime(2016, 12, 31)
end = datetime.datetime(2017, 1, 10)


# ## Data from Yahoo

# In[3]:

data = dict((sym, DataReader(sym, "yahoo", start, end))
          for sym in symbols)


# In[4]:

data['AAPL'].index


# In[5]:

dfa = data['AAPL']
dfa['New']  = np.arange( dfa.shape[1] )

dfg = data['GOOG']
dfg['New'] = np.arange( dfg. shape[1] )[::-1]


# # At the end of the day, Pandas are 2D
# ## even with Multi-Index
# ##    pct_change() works along row dimension (and is NOT affected by multi-index

# In[6]:

c = pd.concat([ data['AAPL'], data['GOOG']], axis=0, keys=['AAPL', 'GOOG'])


# In[7]:

c


# ## Notice below the bad value in GOOG Open on first date
# ## it is the change from the row above, which is AAPL last date

# In[8]:

c.pct_change() * 100


# ## Concat along columns

# ## The column is now multi-index

# In[9]:

d=pd.concat( [data['AAPL'], data['GOOG']], axis=1, keys=['AAPL', 'GOOG'], names=["ticker", "attr"])


# In[10]:

d


# In[11]:

d.pct_change()


# # To illustrate point that Pandas are 2D, try a different concat
# # where Date is left-most part of multi-index rather than right-most

# ## move the ticker from the column multi-index to a row multi-index

# In[12]:

e = d.stack(level=0)


# In[13]:

e


# ## NOTICE below: the bad pct change for GOOG Open on first day
# ### This is the pct change from AAPL first day Open (115.8) to GOOG first day Open (778.8)
# ###   This is b/c they are consecutive rows (i.e., multi-index is just a fancy way of labelling the row, it has not import for computation)

# In[14]:

e.pct_change() * 100


# ## The below won't work either for pct_chg() b/c row order not consistent with what we want

# In[15]:

f = e.swaplevel(i=0,j=1)


# In[16]:

f


# In[42]:

bydate = c.groupby(level=1)
for name, group in bydate:
        print("Group name: {gname}, group:\n\t{ggroup}\n\n".format(gname=name, ggroup=group))


# In[40]:

byticker = c.groupby(level=0)
for name, group in byticker:
        print("Group name: {gname}, group:\n\t{ggroup}\n\n".format(gname=name, ggroup=group))

