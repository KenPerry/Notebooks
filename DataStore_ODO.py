
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


# In[3]:

create = False


# ## Create a Yahoo data provider to retrieve data from Yahoo

# In[4]:

import trans.dataprovider.PDR.yahoo as yh
yh_dp = yh.Yahoo()


# ## Create a Tiingo data provider

# In[5]:

import trans.dataprovider.PDR.tiingo as tg
get_ipython().magic('aimport trans.dataprovider.PDR.tiingo')

tingo_ak_file="/home/ubuntu/Notebooks/tiingo_apkey.txt"
with open(tingo_ak_file, "r") as fp:
    ak_t = fp.read().rstrip()
    
tg_dp = tg.Tiingo(access_key=ak_t)


# In[19]:

dburl = 'sqlite:////tmp/foo.db'
db = odb.ODO(dburl, provider=tg_dp, echo=True)


# In[20]:

db.setup_database()


# In[21]:

if create:
    db.setup_database()


# In[9]:

Memorialize = False


# In[10]:

today = dt.datetime.combine( dt.date.today(), dt.time.min)
if Memorialize:
    today = dup.parse("03/09/2018")
    
start = dup.parse("01/01/2018")
end   = today

status, df_go = db.get_one("AAPL", start, end)


# ## Create a small load

# In[23]:

get_ipython().magic('aimport trans.dataprovider.PDR.tiingo')
import trans.dataprovider.PDR.tiingo
tickers = [ "FB", "AAPL", "AMZN", "NFLX", "GOOG"]
changed_tickers = db.get_data(tickers, start, end)


# ## Get info about existing table in db

# In[12]:

from sqlalchemy import Table, MetaData
meta = MetaData()
p = Table("prices", meta, autoload=True, autoload_with=db.engine)


# ## Enumerate the columns

# In[13]:

[ c.name for c in p.columns ]


# ## On load, catch the events and name attributes based on physical table column names.
# These attributes DON'T affect the table (or even create a view), they are just attributes of the Table object and control the name of it's attributes

# In[9]:

from sqlalchemy import Table, event 
@event.listens_for(Table, "column_reflect")
def column_reflect(inspector, table, column_info):
    print("reflect")
    column_info["key"] = "test_{}".format(column_info["name"])


# In[12]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float,create_engine, bindparam
from sqlalchemy.orm import Session

from sqlalchemy.sql import and_, or_, not_, func

from sqlalchemy import Table, MetaData
meta = MetaData()

Base = declarative_base()
class TestClass(Base):
    __table__ = Table("prices", meta, autoload=True, autoload_with=db.engine)
   


# In[31]:

import pandas as pd
sess = db.session
q = sess.query(TestClass, TestClass.test_Ticker, TestClass.test_Close).filter(TestClass.test_Ticker == "AAPL")
for r in q.all():
    (t, c) = r.test_Ticker, r.test_Close
    print("Read {} at close {}".format(t,c))
# df_tc = pd.read_sql(q.statement, sess.bind)


# In[18]:

df_tc.shape


# ## Create a full load

# In[25]:

get_ipython().magic('aimport trans.dataprovider.PDR.tiingo')
import trans.dataprovider.PDR.tiingo

dburl = 'sqlite:////tmp/full.db'
idb = odb.ODO(dburl, provider=tg_dp)

idb.setup_database()

idb.existing()
from datetime import timedelta

today = dt.datetime.combine( dt.date.today(), dt.time.min)
start=dup.parse("01/01/2000")
end  = today

from trans.data import GetData
gd = GetData()
tickers = gd.existing()
tickers.sort()
print( len(tickers) )


# In[27]:

changed = idb.get_data(tickers, start, end)


# In[ ]:

dfi = idb.combine_data(tickers = [ "FB", "AAPL", "AMZN", "NFLX", "GOOG" ], start="2018-03-01")
dfi.shape
dfi.head()


# In[ ]:

dfi.info()


# In[ ]:

dfi.columns.get_level_values(1).unique()


# In[ ]:

tickers = [ "FB", "AAPL", "AMZN" ]
db.get_data(tickers, start, end)


# In[10]:

df = db.combine_data(["A", "AA"])
df.shape


# In[ ]:

df = db.combine_data([ "AAPL", "AMZN"]) # , dup.parse("03/02/2018"), dup.parse("03/09/2018"))


# In[ ]:

df.shape
df.columns
df.tail()


# In[ ]:

get_ipython().magic('pdb')
import pickle
with open("verify_mom_raw_df.pkl","rb") as fp:
    df = pickle.load(fp)


# In[ ]:

from trans.data import GetData
gd = GetData()
f_df = gd.load_data("verify_mom_raw_df.pkl")


# In[ ]:

f_df.index.min(), f_df.index.max()
df.index.min(), df.index.max()
df.columns


# In[ ]:

from trans.verify_tools import *
verify_df( df. loc[:, "Adj Close"], f_df.loc[:, ["AAPL", "AMZN"]])


# In[ ]:


import pandas as pd
idx = pd.IndexSlice
abs(df.loc["2018-01-03":,   idx["Adj Close"]] -f_df.loc["2018-01-03":,   ["AAPL", "AMZN"]])


