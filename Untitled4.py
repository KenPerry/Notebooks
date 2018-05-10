
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[49]:

import pandas as pd
idx = pd.IndexSlice

import datetime as dt
from datetime import date
from datetime import timedelta
import dateutil.parser as dup

from trans.date_manip import Date_Manipulator


# In[3]:

Memorialize = False


# In[4]:

today = dt.datetime.combine( dt.date.today(), dt.time.min)
if Memorialize:
    today = dup.parse("03/09/2018")
    
start = dup.parse("01/01/2018")
end   = today



# In[5]:

import trans.dataprovider.alphavantage as aa
get_ipython().magic('aimport trans.dataprovider.alphavantage')

aa_dp = aa.Alphavantage()


# In[33]:

import trans.dataprovider.odo as odo_reader
get_ipython().magic('aimport trans.dataprovider.odo')

from sqlalchemy.ext.declarative import declarative_base

dburl="sqlite:////tmp/full.db"
decBase_r = declarative_base()

odr = odo_reader.ODO(dburl, declarative_base=decBase_r, provider=aa_dp)

df_aa = odr.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG" ], start="2018-03-01")
df_aa.shape


# In[7]:

from trans.data import GetData
gd = GetData()

universe = gd.get_r1000_tickers()
universe.sort()
len(universe)


# In[121]:

import trans.quantfactor.volatility as vf
get_ipython().magic('aimport trans.quantfactor.volatility')

get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.quantfactor.base')

v = vf.Volatility(universe=["FB", "AAPL", "AMZN", "NFLX", "GOOG" ], dataProvider=odr)


# In[122]:

daily_price_df = v.load_prices(start=start, end=end)


# In[62]:

daily_price_df.shape


# In[44]:

existing_tickers = list(daily_price_df.columns.get_level_values(1).unique())
existing_tickers.sort()
existing_tickers


# In[123]:

price_attr = "Adj Close"
ret_attr = "Ret"
daily_ret_df = v.create_dailyReturns(price_attr, ret_attr )


# In[64]:

daily_ret_df.shape


# In[125]:

dm = Date_Manipulator( v.price_df.index )
eom_in_idx = dm.periodic_in_idx_end_of_month(end)
v.set_endDates( eom_in_idx )


# In[126]:

s = v.daily_ret_df.loc[:, idx[ret_attr,:]].std()
v.daily_ret_df.loc[:, idx[ret_attr,:]].columns.get_level_values(1).tolist()
for (k,val) in s.iteritems():
    print("k {}, v {}".format(k,val))
s.get_values()


# In[136]:

from datetime import timedelta
vol_window = timedelta(days=30)
 
v.create_period_attr( v.daily_ret_df.loc[:, idx[ret_attr,:]], start, end, vol_window, [ "Volaitlity" ])


# In[45]:

missing_universe = list(set(universe) - set(existing_tickers))
missing_universe.sort()
print("No data for following R1000 names: {}".format(", ".join(missing_universe)))

universe = list(set(universe) - set(missing_universe))
universe.sort()
print("Available universe has {} tickers".format(len(universe)))

