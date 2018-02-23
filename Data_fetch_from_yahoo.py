
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[10]:

import pandas as pd
import os
import datetime as dt
from datetime import date


# In[3]:

get_ipython().magic('aimport trans.data')
from trans.data import GetData

start = dt.datetime(2000, 1, 1)


# In[4]:

gd = GetData()


# ## Issue: webreader returns index as datetime; writing to csv converts it to object, so when concatenatin the two we get datetime

# In[5]:

existing_tickers = gd.existing()
len(existing_tickers)


# In[6]:

existing_tickers.sort()
len(existing_tickers)


# In[7]:

today = dt.datetime.combine( date.today(), dt.time.min)
today


# ## In case we have damaged the files with duplicates: clean them up

# In[15]:

from trans.data import GetData
get_ipython().magic('aimport trans.data')
gd = GetData()
cleaned = gd.clean_data( existing_tickers )


# In[45]:

len(cleaned)


# In[14]:

changed_tickers = gd.get_data( existing_tickers, start, today )


# In[80]:

len(changed_tickers)

