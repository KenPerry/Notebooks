
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

# df = gd.load_data("ret_and_beta_df.pkl")
ret_df = gd.load_data("ret_and_rolled_beta_df.pkl")
ret_df.shape
ret_df.tail()


# In[4]:

i = ret_df.index
i.min()
i.max()


# In[5]:

from trans.date_manip import Date_Manipulator
dm = Date_Manipulator( ret_df.index )
monthly = dm.periodic( dup.parse("01/01/2018"), datedelta.datedelta(months=1) )

end_of_monthly =  dm.periodic_end_of_month( dup.parse("12/15/2017"))
end_of_monthly[-10:]


# In[6]:

eom_in_idx = dm.periodic_in_idx_end_of_month( dup.parse("12/15/2017"))
eom_in_idx[-10:]


# In[7]:

month_r = dm.range_in_index( eom_in_idx)
month_r[-10:]


# In[8]:

price_monthly_df = ret_df.loc[ eom_in_idx,: ]
price_monthly_df.shape


# ## Create monthly returns

# In[9]:

price_attr = "Adj Close"
price_attr_shift_1m = price_attr + " -1m"

ret_attr = "Pct"
monthly_ret_attr = "Month " + ret_attr
monthly_rank_attr = monthly_ret_attr + ' rank'


# In[10]:

monthly_ret_pl = GenRetAttrTransformer( price_attr, price_attr_shift_1m, monthly_ret_attr, 1 )
monthly_ret_df = monthly_ret_pl.fit_transform( price_monthly_df )
monthly_ret_df.tail()


# ## Create end-of-month rank (based on monthly return), and use them as rank for each day of following month

# ## Rank ONLY the tickers in the universe (e.g., NOT SPY)

# In[11]:

univ = ["FB", "AAPL", "AMZN", "NFLX", "GOOG"]


# In[12]:


next_period_rank_pl = GenRankEndOfPeriodAttrTransformer(
    monthly_ret_df,
    monthly_ret_attr,
    univ,
    monthly_rank_attr
)
next_period_rank_df = next_period_rank_pl.fit_transform( ret_df )

next_period_rank_df.loc[ "11/25/2017":"12/01/2017"]


# ## Create HML portfolio return

# In[41]:

wt_attr = "HML weight"
portret_attr = "Port"

portret_pl = make_pipeline(  GenRankToPortRetTransformer(
    ret_attr,
    next_period_rank_df,
    monthly_rank_attr,
    wt_attr,
    portret_attr
)
                          )

port_ret_df = portret_pl.fit_transform( ret_df )

port_ret_df.loc[ "11/25/2017":"12/01/2017"]

