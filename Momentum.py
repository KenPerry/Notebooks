
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


# In[3]:

start = dup.parse("01/01/2017")
end =   dup.parse("03/15/2018")
#end = date.today()
start, end


# In[4]:

import trans.qfactors as qf

from trans.date_manip import Date_Manipulator


# ## Define the universe

# In[5]:

universe = gd.get_r1000_tickers()
len(universe)
universe = [ "FB", "AAPL", "AMZN", "NFLX", "GOOG"]


# In[6]:

mom = qf.MomentumPipe(universe)


# ## Memorialize: save intermediate data as "blessed" output

# In[7]:

memorialize = False
#memorialize = True


# ## Load prices

# In[8]:

price_df = mom.load_prices( start, end)


# In[9]:

if memorialize:
    gd.save_data(price_df.loc[:, "Adj Close"], "verify_mom_raw_df.pkl")
    


# ## Set period-end dates
# ### end-of-month dates, subject to those dates being in price_df.index

# In[10]:

dm = Date_Manipulator( mom.price_df.index )


# In[11]:

eom_in_idx = dm.periodic_in_idx_end_of_month(end)
mom.set_endDates( eom_in_idx )


# ## Create daily returns
# ### Needed to create the daily factor series

# In[12]:

price_attr = "Adj Close"
ret_attr = "Ret"
daily_ret_df = mom.create_dailyReturns(price_attr, ret_attr )


# ## Create period returns
# ### Needed to create the ranks

# In[13]:

period_ret_attr = ret_attr + " yearly"
yearly_ret_df = mom.create_periodReturns(price_attr, period_ret_attr, periods=12 )


# In[14]:

if memorialize:
    gd.save_data(daily_ret_df,  "verify_mom_daily_ret_df.pkl")
    gd.save_data(yearly_ret_df, "verify_mom_yearly_ret_df.pkl")
    


# ## Create ranks
# ### Ranks are based on self.period_ret_df, calculated on each period-end date
# ### These ranks are then applied in the subsequent period, on a daily basis
# ### i.e.,  shifted one-day forward  and then forward filled dail
# 

# In[15]:

daily_rank_df = mom.create_ranks()


# In[16]:

factor_ret_attr = ret_attr + " Factor"
factor_df = mom.create_factor()


# In[17]:

if memorialize:
    gd.save_data(daily_rank_df, "verify_mom_daily_rank_df.pkl")
    gd.save_data(factor_df,     "verify_mom_factor_df.pkl")


# In[18]:

factor_df.tail()

