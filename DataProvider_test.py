
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[ ]:

import trans.datastore.odo as odb
get_ipython().magic('aimport trans.datastore.odo')


# In[2]:

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


# In[ ]:

try:
    df = web.DataReader(ticker, "yahoo", start, end)
    # df = web.get_data_yahoo(ticker, start, end)
except ImmediateDeprecationError as e:
    print("Exception: {}".format(e))
    pass


# In[ ]:

df.shape
df.head()


# In[ ]:

tingo_ak_file="/home/ubuntu/Notebooks/tiingo_apkey.txt"
with open(tingo_ak_file, "r") as fp:
    ak_t = fp.read().rstrip()


# In[ ]:

df_t = web.DataReader(ticker, "tiingo", start, end, access_key=ak_t)
df_t.shape
# df_t.loc["2018-03-10":].head()
df_t.index = df_t.index.droplevel(0)
df_t.loc["2018-03-10":, ["close", "adjClose"]].head()
df.loc["2018-03-10":, ["Close", "Adj Close"]].head()

df.loc["2018-03-10":, "Close"] - df_t.loc["2018-03-10":,"close"]
df.loc["2018-03-10":, "Adj Close"] - df_t.loc["2018-03-10":,"adjClose"]
df_t.loc["2018-03-10":,"divCash"] == 0


# In[ ]:

av_ak_file="/home/ubuntu/Notebooks/alphavantage_apkey.txt"
with open(av_ak_file, "r") as fp:
    av_ak_t = fp.read().rstrip()


# In[ ]:

import pandas as pd
import requests
url="https://alphavantage.co/query"
av_func= "TIME_SERIES_DAILY"
av_args = { "apikey": av_ak_t,
           "symbol": ticker,
           "function": av_func
          }

url_str = "https://www.alphavantage.co/query?function={f}&symbol={s}&apikey={ak}".format(f="TIME_SERIES_DAILY", s=ticker, ak=av_ak_t)
#url_str = url_str + "&outputsize=full"
url_str_csv = url_str + "&datatype=csv"
print(url_str)
print(url_str_csv)

df_c = pd.read_csv(url_str_csv)
df_c.info()
df_c.head()
df_c[ "Date"] = df_c["timestamp"].map( lambda  s: pd.to_datetime(s, infer_datetime_format=True))
df_c.set_index("Date")   
df_c.tail()

pages= requests.get(url_str)

type(pages)
dictionary = pages.json()
keys = list( dictionary.keys())
print(keys)
series = keys[1]
df_av = pd.DataFrame.from_dict( dictionary[series], orient="index")
df_av = df_av.astype(float)
df_av.info()





# In[ ]:

df_av.tail()


# In[4]:

from trans.dataprovider.alphavantage import Alphavantage
get_ipython().magic('aimport trans.dataprovider.alphavantage')
aa = Alphavantage()


# In[ ]:

df_a = aa.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[ ]:

df_a.tail()
df_a.sort_index(axis=1, level=0, inplace=True)
df_a.sort_index(axis=1, level=1, inplace=True)
df_a.tail()


# In[ ]:

df_a.columns


# In[ ]:

from trans.dataprovider.PDR.yahoo import Yahoo
get_ipython().magic('aimport trans.dataprovider.PDR.yahoo')

yh = Yahoo()
df_yp = yh.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[ ]:

df_yp.info()
df_yp.tail()


# In[ ]:

from trans.dataprovider.PDR.tiingo import Tiingo
get_ipython().magic('aimport trans.dataprovider.PDR.base')
get_ipython().magic('aimport trans.dataprovider.PDR.tiingo')

tg = Tiingo(access_key=ak_t)
# pyh.source = "yahoo"
df_tg = tg.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[ ]:

df_tg.info()
df_tg.tail()


# In[10]:

from trans.dataprovider.odo import ODO
get_ipython().magic('aimport trans.dataprovider.odo')
from sqlalchemy.ext.declarative import declarative_base

dburl="sqlite:////tmp/full.db"
decBase = declarative_base()

odr = ODO(dbURL=dburl, declarative_base=decBase, provider=aa)
# pyh.source = "yahoo"
df_od = odr.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG"], start=start, end=end)


# In[9]:

df_od.info()
df_od.tail()


# In[ ]:

from trans.data import GetData
gd = GetData()


# In[ ]:

status, df_gd = gd.get_one("FB", start, end)
df_gd.info()


# In[ ]:

import pandas as pd
idx = pd.IndexSlice
df_yp_fb = df_yp.loc[:, idx[:,"FB"]]
df_yp_fb.columns = df_yp_fb.columns.droplevel(level=1)
df_yp_fb.info()

abs(df_yp_fb - df_gd).max()
df_yp_fb.tail()
df_gd.tail()

