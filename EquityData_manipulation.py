
# coding: utf-8

# In[101]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[102]:

from pandas.io.data import DataReader
# from pandas_datareader import data, wb
import pandas as pd
import numpy as np


# In[103]:

import datetime
start = datetime.datetime(2016, 12, 31)
end = datetime.datetime(2017, 1, 10)


# In[104]:

# For Yahoo, replace leading '.' by leading '^'
symbols = [ 'SPY', 'MSFT', 'GOOG', 'AAPL' ] #, '^GSPC', 'INDEXSP:.INX', 'INDEXCBOE:SPX']
source = "yahoo"
source = "google"

data = DataReader(symbols, source, start, end).to_frame()
data.index = data.index.set_names(["Date", "Ticker"])
data


# In[105]:

data_ts = data.unstack(level=1)
data_ts.columns = data_ts.columns.set_names(["Attr", "Ticker"])
data_ts


# In[106]:

data_ts_pct = data_ts.pct_change()
data_ts_pct


# In[107]:

rename_dict = dict( (c, c + " Pct") for c in data_ts_pct.columns.levels[0] )
rename_dict
data_ts_pct = data_ts_pct.rename(columns=rename_dict)
data_ts_pct


# In[108]:

data_ts = pd.concat([ data_ts, data_ts_pct], axis=1)
data_ts


# In[111]:

data = data_ts.stack(level=1)
data


# In[140]:

data.loc[:, 'Clost Pct Rank'] = data.loc[:,'Close Pct'].groupby(axis=0, level=0).rank()
data


# In[156]:

import quandl
quandl.ApiConfig.api_key ='evWfebtKvTVN_dxvWqau'

symbols = [ 'MSFT', 'GOOG', 'AAPL' ]
t = [ "WIKI/{sym}.{num:d}".format(sym=s, num=i)  for s in symbols for i in np.arange(5)]
t
mydata = quandl.get(t, start_date=start, end_date=end)
mydata

