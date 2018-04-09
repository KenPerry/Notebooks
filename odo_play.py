
# coding: utf-8

# http://docs.sqlalchemy.org/en/latest/_modules/examples/performance/bulk_inserts.html

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

import pandas as pd
idx = pd.IndexSlice


# In[3]:

from odo import *


# In[4]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, bindparam
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


# In[5]:

dburl = 'sqlite:////tmp/foo.db'


# In[6]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float,create_engine, bindparam
from sqlalchemy.orm import Session


# In[7]:

Base = declarative_base()

class Price(Base):
    __tablename__ = "prices"
    Ticker = Column(String(255), primary_key=True)
    Date   = Column(Date, primary_key=True)
    AdjClose = Column(Float)
    Close   = Column(Float)
    High    = Column(Float)
    Low     = Column(Float)
    Open    = Column(Float)
    Volume  = Column(Float)
    


# In[8]:

def setup_database(dburl, echo, num):
    engine = create_engine(dburl, echo=echo)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    return engine


# In[9]:

engine = setup_database(dburl, True, 0)


# In[10]:

dfs = []

for t in ("a", "b"):
    df = pd.read_csv("/tmp/" + t + ".csv")
    df["Ticker"] = t
    df = df.rename( columns={ "Adj Close": "AdjClose"} )
    
    df["Date"] = pd.to_datetime( df["Date"])
    
    dfs.append(df)              
                         
df_big_thin = pd.concat( dfs )


# In[11]:

df_big_thin.shape


# In[12]:

from sqlalchemy.sql import and_, or_, not_, func




# In[13]:

session_add = Session(bind=engine)
rows = df_big_thin.to_dict("records")

for row in rows:
    rec = Price(**row)
    
    session_add.add(rec)

try:
    session_add.flush()
except SQLAlchemyError as e:
    print("Flush error: {}".format(e))
    
session_add.commit()    


# In[14]:

rsession = Session(bind=engine)


# ## Find max date per ticker

# In[15]:

# q = session_add.query(Price).filter( and_(Price.Ticker=="a", Price.Date>="2018-03-27") )
q = rsession.query(Price, Price.Ticker, func.max(Price.Date).label("max_date")).group_by(Price.Ticker)
l = q.all()
maxDate = {}
for a in l:
    maxDate[a.Ticker] = a.max_date
    print(a.Ticker, a.max_date)

maxDate


# In[16]:

ticker_up = "a"
df_up = pd.read_csv("/tmp/" + ticker_up + "_up.csv")


# In[ ]:

dates_up = df_up["Date"].tolist()
dates_up


# In[17]:

q = rsession.query(Price).filter( and_(Price.Ticker == ticker_up, Price.Date.in_(dates_up)) )
l = q.all()
for a in l:
    print(a.Ticker, a.Date)


# ## Delete rows

# In[ ]:

q  = rsession.query(Price).filter(and_(Price.Ticker == ticker_up, Price.Date.in_(dates_up)))
print(q.all())
num_rows = q.delete(synchronize_session="fetch") # or = False
print("Deleted {} rows".format(num_rows))


# In[ ]:

q = rsession.query(Price)
resps = q.all()
for r in resps[0:10]:
    print(r.Ticker, r.Date)


# In[ ]:

type(q)
l = q.all()
len(l)
type(l)
l[0]


# ## Read from db; form wide df

# In[ ]:

df_r = pd.read_sql(rsession.query(Price).statement, rsession.bind)
df_r[ "Dt" ] = df_r.loc[:,"Date"].map( lambda  s: pd.to_datetime(s, infer_datetime_format=True) )
df_r = df_r.drop("Date", axis=1)
df_r.set_index(["Dt", "Ticker"], inplace=True)
df_w = df_r.unstack(level=1)
df_w.tail()

df_w.loc["2018-03-29", idx[ "AdjClose",:]]


# ## Convert wide df to think table to store in db

# In[ ]:

df_t = df_w.stack(level=1)
df_t.reset_index(inplace=True)
df_t.tail()
df_t.columns


# In[ ]:




# In[ ]:

def test_flush_no_pk(n):
    """Individual INSERT statements via the ORM, calling upon last row id"""
    session = Session(bind=engine)
    for chunk in range(0, n, 1000):
        session.add_all([
            Customer(
                name='customer name %d' % i,
                description='customer description %d' % i)
            for i in range(chunk, chunk + 1000)
        ])
        session.flush()
    session.commit()


# In[ ]:

fname = "/tmp/a.csv"
ds = discover( resource(fname) )
ds

