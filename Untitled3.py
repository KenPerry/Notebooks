
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

import datetime as dt
from datetime import date
from datetime import timedelta
import dateutil.parser as dup

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.reg')
get_ipython().magic('aimport trans.regpipe')
get_ipython().magic('aimport trans.qfactors')

from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

import trans.qfactors as qf
from trans.date_manip import Date_Manipulator


# In[14]:

start = dup.parse("01/01/2017")
#end =   dup.parse("03/15/2018")

today = dt.datetime.combine( date.today(), dt.time.min)
end = today
start, end

refreshData = False


# In[7]:

existing_tickers = gd.existing()
len(existing_tickers)


# In[15]:

existing_tickers.sort()
changed_tickers = []
if refreshData:
    changed_tickers = gd.get_data( existing_tickers, start, end)


# In[21]:

cleaned = gd.clean_data(existing_tickers)


# In[22]:

universe = gd.get_r1000_tickers()
len(universe)

missing_universe = list(set(universe) - set(existing_tickers))
missing_universe.sort()
print("No data for following R1000 names: {}".format(", ".join(missing_universe)))

universe = list(set(universe) - set(missing_universe))
universe.sort()
print("Available universe has {} tickers".format(len(universe)))

mom = qf.MomentumPipe(universe)
price_df = mom.load_prices( start, end)


# In[23]:

price_df.shape


# In[24]:

dm = Date_Manipulator( mom.price_df.index )
eom_in_idx = dm.periodic_in_idx_end_of_month(end)
mom.set_endDates( eom_in_idx )


# In[25]:

eom_in_idx


# In[ ]:

price_attr = "Adj Close"
ret_attr = "Ret"
daily_ret_df = mom.create_dailyReturns(price_attr, ret_attr )


# In[ ]:

period_ret_attr = ret_attr + " yearly"
yearly_ret_df = mom.create_periodReturns(price_attr, period_ret_attr, periods=12 )


# In[ ]:

daily_rank_df = mom.create_ranks()


# In[ ]:

factor_ret_attr = ret_attr + " Factor"
factor_df = mom.create_factor()

