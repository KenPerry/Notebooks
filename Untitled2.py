
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[16]:

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

from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe



# In[19]:

today = dt.datetime.combine( date.today(), dt.time.min)
today

start = dup.parse("01/01/2000")
start


# In[20]:

gd = GetData()
univ = gd.existing()
univ.sort()

len(univ)


# In[21]:

sectors =  { 
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial": "XLF",
    "Health": "XLV",
    "Industrial": "XLI", 
    "Materials" : "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK", 
    "Telecom": "XTL",
    "Utilities": "XLU"
}
   


# In[22]:

sector_tickers = list( sectors.values() )


# In[28]:

changed_tickers = gd.get_data( sector_tickers, start, today )


# In[29]:

len(sector_tickers)
len(changed_tickers)
list( set(sector_tickers) - set(changed_tickers))


# In[30]:

price_df = GetDataTransformer(sector_tickers, cal_ticker="SPY").fit_transform( pd.DataFrame())
price_df.shape


# In[ ]:

get_ipython().magic('aimport trans.data')
raw_df = gd.combine_data(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
raw_df.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

sector_tickers =  { 
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial": "XLF",
    "Health": "XLV"
    "Industrial": "XLI", 
    "Materials" : "XLB",
    "Real Estate"; "XLRE",
    "Technology": "XLK", 
    "Telecom": "XTL",
    "Utilities": "XLU"
   

