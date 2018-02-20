
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.reg')
get_ipython().magic('aimport trans.date_manip')


import pandas as pd
idx = pd.IndexSlice

from trans.data import GetData
from trans.gtrans import *
from trans.reg import Reg
import datetime as dt
import dateutil.parser as dup
import datedelta

from trans.date_manip import Date_Manipulator

gd = GetData()


# In[3]:


sp500_univ = gd.get_sp500_tickers()
len(sp500_univ)

univ = sp500_univ


# In[4]:

get_ipython().magic('aimport trans.data')
price_df = gd.combine_data( univ )
price_df.shape

