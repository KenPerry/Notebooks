
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


# In[3]:

start = dup.parse("01/01/2015")
#end =   dup.parse("03/15/2018")

today = dt.datetime.combine( date.today(), dt.time.min)
# today = dup.parse( dt.datetime.today().strftime("%m/%d/%Y"))
end = today
start, end

refreshData = False


# In[4]:

existing_tickers = gd.existing()
len(existing_tickers)


# In[5]:

existing_tickers.sort()
changed_tickers = []
if refreshData:
    changed_tickers = gd.get_data( existing_tickers, start, end)
    cleaned = gd.clean_data(existing_tickers)


# In[6]:

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


# In[7]:

price_df.shape


# In[8]:

dm = Date_Manipulator( mom.price_df.index )
eom_in_idx = dm.periodic_in_idx_end_of_month(end)
mom.set_endDates( eom_in_idx )


# In[9]:

price_attr = "Adj Close"
ret_attr = "Ret"
daily_ret_df = mom.create_dailyReturns(price_attr, ret_attr )


# ## 12m1m: 11 month return beginning 12m ago and ending 1m ago
# - so 11 month return, shift forward 1 mo

# In[10]:

period_ret_attr = ret_attr + " yearly"
yearly_ret_df = mom.create_periodReturns(price_attr, period_ret_attr, periods=11)

yearly_ret_df = mom.period_ret_df
shifted_yearly_ret_df = ShiftTransformer(1).fit_transform(yearly_ret_df)
mom.period_ret_df = shifted_yearly_ret_df


# In[11]:


yearly_ret_df.loc["2017-12-01":"2018-03-20", idx[ period_ret_attr, "NFLX"]]
shifted_yearly_ret_df.loc["2017-12-01":"2018-03-20", idx[ period_ret_attr, "NFLX"]]


# In[12]:

daily_rank_df = mom.create_ranks()


# In[37]:

s = daily_rank_df.loc["2018-03-01", idx["Rank",:]]
s_defined = s[ s.isnull() == False ]
size = s_defined.size
size


# In[25]:

factor_ret_attr = ret_attr + " Factor"
factor_df = mom.create_factor()


# In[26]:

f_ret = factor_df.loc[:, idx["Ret", "Port net"]]
f_ret.loc["2018-01-01":]


# ## Standard deviation computed over last year

# In[27]:

std_dev = f_ret.loc[end - dt.timedelta(days=365) :].std()


# In[28]:

get_ipython().magic('matplotlib inline')
f_ret_recent = f_ret.loc["2018-01-01":]
f_ret_recent.plot()
f_ret_recent_z = f_ret_recent/std_dev

f_ret_recent_z.plot()


# In[29]:

gd.save_data(factor_df, "mom_r1000_12m1m.pkl")


# In[30]:

factor_df.to_csv("mom_r1000_12m1m.csv")


# In[31]:

f_ret_recent_z.tail()


# In[39]:

s = daily_rank_df.loc["2018-03-01", idx["Rank",:]]
s_defined = s[ s.isnull() == False ]
size = s_defined.size
size

row = factor_df.loc["2018-03-01", idx[ "weight",:]]

size
size * 0.20
row[ row < 0].sum()
row[ row > 0].sum()

