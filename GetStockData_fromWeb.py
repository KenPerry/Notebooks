
# coding: utf-8

# ## From
# https://chrisconlan.com/download-historical-stock-data-google-r-python/

# In[52]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## Demonstrates how to use Web request to fetch data from URL into
# ## a Panda Dataframe

# In[53]:

import pandas as pd
import io
import requests
import time
 


# In[2]:

def google_stocks(symbol, startdate = (1, 1, 2005), enddate = None):
 
    startdate = str(startdate[0]) + '+' + str(startdate[1]) + '+' + str(startdate[2])
 
    if not enddate:
        enddate = time.strftime("%m+%d+%Y")
    else:
        enddate = str(enddate[0]) + '+' + str(enddate[1]) + '+' + str(enddate[2])
 
    stock_url = "http://www.google.com/finance/historical?q=" + symbol +                 "&startdate=" + startdate + "&enddate=" + enddate + "&output=csv"
 
    raw_response = requests.get(stock_url).content
 
    stock_data = pd.read_csv(io.StringIO(raw_response.decode('utf-8')))
 
    return stock_data
 
 


# In[9]:

apple_data = google_stocks('AAPL')
# apple_data.info
apple_data.columns
apple_data.index


# In[4]:

apple_data = google_stocks('AAPL')
print(apple_data)
 


# In[10]:

apple_truncated = google_stocks('AAPL', enddate = (1, 1, 2006))
print(apple_truncated)


# ## Use Quandl
# 
# See https://www.quandl.com/tools/python

# In[41]:

import quandl

with open("quandl_apkey.txt", "r") as keyfile:
    key = keyfile.read()
    
quandl_key = key.rstrip()
quandl.ApiConfig.api_key = quandl_key # "evWfebtKvTVN_dxvWqau"


# In[42]:

mydata = quandl.get("FRED/GDP")

mydata.head()


# In[51]:

data = quandl.get_table('ZACKS/FC', paginate=True, 
                        ticker=['AAPL', 'MSFT'], 
                        per_end_date={'gte': '2015-01-01'}, 
                        qopts={'columns':['ticker', 'per_end_date']}
                       )

data['ticker'].value_counts()
data.head()


# ## Use Datareader
# ### n.b., uses Panel, which is deprecated ?

# In[11]:

from pandas_datareader import data
import pandas as pd


# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
tickers = ['AAPL', 'MSFT', 'SPY']

# Define which online source one should use
data_source = 'google'

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader(tickers, data_source, start_date, end_date)


# In[17]:

type(panel_data)
panel_data.ix['Close'].head()


# In[19]:

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data.ix['Close']

# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

close.tail(10)


# In[21]:

df = panel_data.to_frame()
df.head()


# ## Fama French

# In[23]:

from pandas_datareader.famafrench import get_available_datasets

import pandas_datareader.data as web

len(get_available_datasets())
ds = web.DataReader("5_Industry_Portfolios", "famafrench")

print(ds['DESCR'])


# In[26]:

type(ds)
ds.keys()


# In[29]:

type(ds[4])
ds[4].head()


# ## NASDAQ Symbols

# In[31]:

from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()
print(symbols.ix['IBM'])

