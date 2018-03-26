
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

from trans.verify_tools import *
get_ipython().magic('aimport trans.verify_tools')

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

import trans.qfactors as qf

from trans.date_manip import Date_Manipulator


# ## Create momentum object; define universe and time range

# In[4]:

universe = [ "FB", "AAPL", "AMZN", "NFLX", "GOOG"]

mom = qf.MomentumPipe(universe)

start = dup.parse("01/01/2017")
end =   dup.parse("03/15/2018")
#end = date.today()
start, end

price_attr = "Adj Close"


# ## Load prices

# In[5]:

price_df = mom.load_prices( start, end)


# In[6]:

f_df = gd.load_data("verify_mom_raw_df.pkl")


# In[7]:

f_df.shape
price_df.shape


# In[8]:

from trans.verify_tools import *
verify_file( price_df. loc[:, "Adj Close"], "verify_mom_raw_df.pkl")


# In[9]:

dm = Date_Manipulator( mom.price_df.index )
eom_in_idx = dm.periodic_in_idx_end_of_month(end)
mom.set_endDates( eom_in_idx )


# ## Create daily returns (needed to construct daily factor series)

# In[10]:

ret_attr = "Ret"
daily_ret_df = mom.create_dailyReturns(price_attr, ret_attr )


# ## Create period returns (this is what is used for ranking)

# In[11]:

period_ret_attr = ret_attr + " yearly"
yearly_ret_df = mom.create_periodReturns(price_attr, period_ret_attr, periods=12 )


# In[12]:

verify_file(daily_ret_df, "verify_mom_daily_ret_df.pkl")
verify_file(yearly_ret_df, "verify_mom_yearly_ret_df.pkl")
    


# ## Create ranks
# - first create end-of-period ranks, using period returns
# - change to daily frequency.  Push the end-of-period ranks forward one day
# - forward fill the ranks daily so perior period end-of-period rank is pushed to all days of subsequent period

# In[13]:

daily_rank_df = mom.create_ranks()


# ## Create the factor returns, using the daily ranks

# In[14]:

factor_ret_attr = ret_attr + " Factor"
factor_df = mom.create_factor()


# In[15]:

verify_file(daily_rank_df, "verify_mom_daily_rank_df.pkl")
verify_file(factor_df, "verify_mom_factor_df.pkl")
    

