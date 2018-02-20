
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[29]:

import os
import datetime as dt
from datetime import date


# In[26]:

get_ipython().magic('aimport trans.data')
from trans.data import GetData

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 12, 1)

end2 = dt.datetime(2018, 1, 8)


# In[61]:

gd = GetData()


# In[9]:

sp500_tickers = gd.get_sp500_tickers()
len(sp500_tickers)


# ## Issue: webreader returns index as datetime; writing to csv converts it to object, so when concatenatin the two we get datetime

# In[10]:

df= gd.get_one("MTUM", start, end)
df.index

df.tail()


# In[15]:

changed, dfs = gd.extend("MTUM", start, end2 )
changed


# In[12]:

dfs.tail()
type(dfs.index)


# In[13]:

tickers = [ "MTUM" ]
file = 'stock_dfs/{}.csv'.format(tickers[0])

if (os.path.exists(file)):
    print("{} exists before fetch".format(file))
    
gd.get_data( tickers, start, end )

if (os.path.exists(file)):
    print("{} exists after fetch".format(file))
    


# In[14]:

gd.get_data(tickers, start, end2)


# In[28]:

existing_tickers = gd.existing()
len(existing_tickers)


# In[58]:

existing_tickers.sort()


# In[45]:

today = dt.datetime.combine( date.today(), dt.time.min)
today


# In[51]:

dt.datetime.strftime(today, "%m/%d/%Y")


# In[65]:

changed_tickers = gd.get_data( existing_tickers, start, today )


# In[66]:

len(changed_tickers)

