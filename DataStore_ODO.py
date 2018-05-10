
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

# In[ ]:

import trans.dataprovider.PDR.tiingo as tg
get_ipython().magic('aimport trans.dataprovider.PDR.tiingo')

tingo_ak_file="/home/ubuntu/Notebooks/tiingo_apkey.txt"
with open(tingo_ak_file, "r") as fp:
    ak_t = fp.read().rstrip()
    
tg_dp = tg.Tiingo(access_key=ak_t)


# ## Create an Alphvantage provider

# In[5]:

import trans.dataprovider.alphavantage as aa
get_ipython().magic('aimport trans.dataprovider.alphavantage')

aa_dp = aa.Alphavantage()


# In[ ]:

dburl = 'sqlite:////tmp/foo.db'
db = odb.ODO(dburl, provider=aa_dp, echo=True)


# In[ ]:

db.setup_database()


# In[ ]:

if create:
    db.setup_database()


# In[6]:

Memorialize = False


# In[ ]:

today = dt.datetime.combine( dt.date.today(), dt.time.min)
if Memorialize:
    today = dup.parse("03/09/2018")
    
start = dup.parse("01/01/2018")
end   = today


# In[ ]:

status, df_go = db.get_one("AAPL", start, end)


# In[ ]:

df_go.tail()


# ## Create a small load

# In[ ]:


tickers = [ "FB", "AAPL", "AMZN", "NFLX", "GOOG"]
changed_tickers = db.get_data(tickers, start, end)


# ## Get info about existing table in db

# In[ ]:

from sqlalchemy import Table, MetaData
meta = MetaData()
p = Table("prices", meta, autoload=True, autoload_with=db.engine)


# ## Enumerate the columns

# In[ ]:

[ c.name for c in p.columns ]


# ## On load, catch the events and name attributes based on physical table column names.
# These attributes DON'T affect the table (or even create a view), they are just attributes of the Table object and control the name of it's attributes

# In[ ]:

from sqlalchemy import Table, event 
@event.listens_for(Table, "column_reflect")
def column_reflect(inspector, table, column_info):
    print("reflect")
    column_info["key"] = "test_{}".format(column_info["name"])


# In[ ]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float,create_engine, bindparam
from sqlalchemy.orm import Session

from sqlalchemy.sql import and_, or_, not_, func

from sqlalchemy import Table, MetaData
meta = MetaData()

Base = declarative_base()
class TestClass(Base):
    __table__ = Table("prices", meta, autoload=True, autoload_with=db.engine)
   


# In[ ]:

import pandas as pd
sess = db.session
q = sess.query(TestClass, TestClass.test_Ticker, TestClass.test_Close).filter(TestClass.test_Ticker == "AAPL")
for r in q.all():
    (t, c) = r.test_Ticker, r.test_Close
    print("Read {} at close {}".format(t,c))
# df_tc = pd.read_sql(q.statement, sess.bind)


# In[ ]:

df_tc.shape


# ## Create a full load

# In[13]:

dburl = 'sqlite:////tmp/full.db'
from sqlalchemy.ext.declarative import declarative_base

dburl="sqlite:////tmp/full.db"
decBase = declarative_base()

idb = odb.ODO(dburl, declarative_base=decBase, provider=aa_dp)

if create:
    idb.setup_database()

idb_tickers = idb.existing()
from datetime import timedelta

today = dt.datetime.combine( dt.date.today(), dt.time.min)
start=dup.parse("01/01/2000")
end  = today

from trans.data import GetData
gd = GetData()
tickers = gd.existing()

tickers = idb_tickers
tickers.sort()
print( len(tickers) )


# In[45]:

changed = idb.get_data(tickers, start, end)


# In[46]:

len(changed)


# ## Read back some data.  Use the DataProvider for ODO to read, NOT the DataStore for ODO!
# ## Must pass both the dbURL and AA DataProvider to the ODO DataProvider.  
# ### The AA DataProvider providd to the DataProvider (reader) should be the same as was used by the DataStore (writer) and is needed to know the format of the data stored in ODO

# In[11]:

import trans.dataprovider.odo as odo_reader
get_ipython().magic('aimport trans.dataprovider.odo')

from sqlalchemy.ext.declarative import declarative_base

dburl="sqlite:////tmp/full.db"
decBase_r = declarative_base()

odr = odo_reader.ODO(dburl, declarative_base=decBase_r, provider=aa_dp)

df_aa = odr.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG" ], start="2018-03-01")
df_aa.shape


# ## combine_data is deprecated ! Don't use writer (DataProvider) to read !

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


# In[ ]:

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


