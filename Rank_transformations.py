
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

price_df = gd.load_data("price.pkl")


# In[4]:

price_df.index


# In[5]:

from trans.date_manip import Date_Manipulator
dm = Date_Manipulator( price_df.index )
monthly = dm.periodic( dup.parse("02/16/2018"), datedelta.datedelta(months=1) )

end_of_monthly =  dm.periodic_end_of_month( dup.parse("01/15/2018"))
end_of_monthly[-10:]


# In[6]:

eom_in_idx = dm.periodic_in_idx_end_of_month( dup.parse("12/15/2017"))
eom_in_idx[-10:]


# In[7]:

month_r = dm.range_in_index( eom_in_idx)
month_r[-10:]


# In[8]:

price_monthly_df = price_df.loc[ eom_in_idx,: ]
price_monthly_df.shape


# ## Create monthly returns

# In[9]:

price_attr = "Adj Close"
price_attr_shift_1d = price_attr + " -1d"
price_attr_shift_1m = price_attr + " -1m"

ret_attr = "Pct"
monthly_ret_attr = "Month " + ret_attr
monthly_rank_attr = monthly_ret_attr + ' rank'


# In[10]:

monthly_ret_pl = GenRetAttrTransformer( price_attr, price_attr_shift_1m, monthly_ret_attr, 1 )
monthly_ret_df = monthly_ret_pl.fit_transform( price_monthly_df )
monthly_ret_df.tail()


# ## Create daily returns (need to construct portfolio)

# In[11]:

daily_ret_pl = GenRetAttrTransformer( price_attr, price_attr_shift_1d, ret_attr, 1 )
daily_ret_df = daily_ret_pl.fit_transform( price_df )
daily_ret_df.tail()


# ## Create end-of-month rank (based on monthly return), and use them as rank for each day of following month

# ## Rank ONLY the tickers in the universe (e.g., NOT SPY)

# In[12]:

univ = ["FB", "AAPL", "AMZN", "NFLX", "GOOG"]


# In[13]:

non_univ = ["SPY"]
univ = daily_ret_df.loc[:, idx[ret_attr,:]].columns.get_level_values(1).unique().tolist()

len(univ)
univ = list( set(univ) - set(non_univ) )
univ.sort()
len(univ)


# In[14]:

next_period_rank_pl = GenRankEndOfPeriodAttrTransformer(
    monthly_ret_df,
    monthly_ret_attr,
    univ,
    monthly_rank_attr
)
next_period_rank_df = next_period_rank_pl.fit_transform( daily_ret_df )

next_period_rank_df.loc[ "11/25/2017":"12/01/2017"]


# ## Create HML portfolio return

# In[15]:

wt_attr = "HML weight"
portret_col = "Port"
hi_rank, lo_rank = 401, 100

portret_pl = make_pipeline(  GenRankToPortRetTransformer(
    ret_attr,
    next_period_rank_df,
    monthly_rank_attr,
    lambda s: (s >= hi_rank) * 1.0 + (s <= lo_rank) * -1.0,
    wt_attr,
    portret_col
)
                          )

port_ret_df = portret_pl.fit_transform( daily_ret_df )

port_ret_df.loc[ "11/25/2017":"12/01/2017"]


# ## Find the col.s with the different type of Port returns
# ### Port: sum of weights * returns.  Weights don't sum up to anything in particular so results can be large
# ### Port > 0: sum of weights * returns/count, all conditional on weights > 0
# ### Port < 0: sum of weights * returns/count, all conditional on weights < 0
# ### Port net: sum of (Port > 0) + (Port < 0).  Sum of the return of a long portfolio and a short portfolio.  Differs from Port in that each side can have a different count as well

# In[16]:

cols = port_ret_df.columns.get_level_values(1).unique().tolist()
port_cols = [ c for c in cols if re.search(portret_col,c)]
port_cols

port_ret_df.loc[ "11/25/2017":"12/01/2017", idx[ret_attr, port_cols] ]


# In[17]:

port_ret_df.shape

