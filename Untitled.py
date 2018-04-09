
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

import dateutil.parser as dup


# In[3]:

dburl = 'sqlite:////tmp/foo.db'
db = odb.ODO(dburl, echo=True)


# In[4]:

db.setup_database()


# In[5]:

session = db.createSession()


# In[6]:

start = dup.parse("03/01/2018")
end   = dup.parse("03/10/2018")

db.get_one("AAPL", start, end)


# In[7]:

tickers = [ "FB", "AAPL", "AMZN" ]
db.get_data(tickers, start, end)


# In[ ]:



