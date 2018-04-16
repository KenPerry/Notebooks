
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

import pandas_datareader.data as web
import pandas_datareader as dr
dr.__version__


# In[3]:

ticker = "SPY"
today = dt.datetime.combine( dt.date.today(), dt.time.min)

start = dup.parse("01/01/2018")
end   = today
from contextlib import suppress
from pandas_datareader.exceptions import DEP_ERROR_MSG, ImmediateDeprecationError
    
with suppress(ImmediateDeprecationError):
    df = web.DataReader(ticker, "yahoo", start, end)
    # df = web.get_data_yahoo(ticker, start, end)
df.shape


# In[4]:

try:
    df = web.DataReader(ticker, "yahoo", start, end)
    # df = web.get_data_yahoo(ticker, start, end)
except ImmediateDeprecationError as e:
    print("Exception: {}".format(e))
    pass


# In[5]:

df.shape
df.head()


# In[33]:

tingo_ak_file="/home/ubuntu/Notebooks/tiingo_apkey.txt"
with open(tingo_ak_file, "r") as fp:
    ak_t = fp.read().rstrip()


# In[39]:

df_t = web.DataReader(ticker, "tiingo", start, end, access_key=ak_t)
df_t.shape
# df_t.loc["2018-03-10":].head()
df_t.index = df_t.index.droplevel(0)
df_t.loc["2018-03-10":, ["close", "adjClose"]].head()
df.loc["2018-03-10":, ["Close", "Adj Close"]].head()

df.loc["2018-03-10":, "Close"] - df_t.loc["2018-03-10":,"close"]
df.loc["2018-03-10":, "Adj Close"] - df_t.loc["2018-03-10":,"adjClose"]
df_t.loc["2018-03-10":,"divCash"] == 0


# In[8]:

from trans.dataprovider.PDR.yahoo import Yahoo
get_ipython().magic('aimport trans.dataprovider.PDR.yahoo')

yh = Yahoo()
df_yp = yh.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[9]:

df_yp.info()
df_yp.tail()


# In[42]:

from trans.dataprovider.PDR.tiingo import Tiingo
get_ipython().magic('aimport trans.dataprovider.PDR.base')
get_ipython().magic('aimport trans.dataprovider.PDR.tiingo')

tg = Tiingo(access_key=ak_t)
# pyh.source = "yahoo"
df_tg = tg.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[43]:

df_tg.info()
df_tg.tail()


# In[12]:

from trans.dataprovider.odo import ODO
get_ipython().magic('aimport trans.dataprovider.odo')

dburl="sqlite:///full.db"

odr = ODO(dbURL=dburl)
# pyh.source = "yahoo"
df_od = odr.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[13]:

df_od.info()


# In[14]:

from trans.data import GetData
gd = GetData()


# In[16]:

status, df_gd = gd.get_one("FB", start, end)
df_gd.info()


# In[31]:

import pandas as pd
idx = pd.IndexSlice
df_yp_fb = df_yp.loc[:, idx[:,"FB"]]
df_yp_fb.columns = df_yp_fb.columns.droplevel(level=1)
df_yp_fb.info()

abs(df_yp_fb - df_gd).max()
df_yp_fb.tail()
df_gd.tail()

