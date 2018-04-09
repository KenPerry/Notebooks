
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

import trans.datastore.odo as odb
get_ipython().magic('aimport trans.datastore.odo')

import datetime as dt
import dateutil.parser as dup


# In[3]:

create = False


# In[4]:

dburl = 'sqlite:////tmp/foo.db'
db = odb.ODO(dburl, echo=True)


# In[5]:

if create:
    db.setup_database()


# In[6]:

Memorialize = False


# In[7]:

today = dt.datetime.combine( dt.date.today(), dt.time.min)
if Memorialize:
    today = dup.parse("03/09/2018")
    
start = dup.parse("01/01/2018")
end   = today

status, df_go = db.get_one("AAPL", start, end)


# ## Create a full load

# In[8]:

dburl = 'sqlite:////tmp/full.db'
idb = odb.ODO(dburl)

# idb.setup_database()

idb.existing()
from datetime import timedelta

today = dt.datetime.combine( dt.date.today(), dt.time.min)
start=dup.parse("01/01/2000")
end  = today

from trans.data import GetData
gd = GetData()
tickers = gd.existing()
tickers.sort()
print( len(tickers) )

changed = idb.get_data(tickers, start, end)


# In[21]:

dfi = idb.combine_data(tickers = [ "FB", "AAPL", "AMZN", "NFLX", "GOOG" ], start="2018-03-01")
dfi.shape
dfi.head()


# In[19]:

dfi.info()


# In[118]:

dfi.columns.get_level_values(1).unique()


# In[104]:

tickers = [ "FB", "AAPL", "AMZN" ]
db.get_data(tickers, start, end)


# In[105]:

df = db.combine_data(["A", "AA"])
df.shape


# In[106]:

df = db.combine_data([ "AAPL", "AMZN"]) # , dup.parse("03/02/2018"), dup.parse("03/09/2018"))


# In[107]:

df.shape
df.columns
df.tail()


# In[69]:

from trans.data import GetData
gd = GetData()
f_df = gd.load_data("verify_mom_raw_df.pkl")


# In[54]:

f_df.index.min(), f_df.index.max()
df.index.min(), df.index.max()
df.columns


# In[74]:

from trans.verify_tools import *
verify_df( df. loc[:, "Adj Close"], f_df.loc[:, ["AAPL", "AMZN"]])


# In[82]:


import pandas as pd
idx = pd.IndexSlice
abs(df.loc["2018-01-03":,   idx["Adj Close"]] -f_df.loc["2018-01-03":,   ["AAPL", "AMZN"]])


