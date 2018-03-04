
# coding: utf-8

# http://docs.sqlalchemy.org/en/latest/_modules/examples/performance/bulk_inserts.html

# In[1]:


from odo import *


# In[16]:

fname = "/tmp/a.csv"
ds = discover( resource(fname) )


# In[18]:

ds


# In[52]:

df = pd.read_csv("/tmp/a.csv")
df.columns
df.rename( columns={ "Adj Close": "AdjClose"}, inplace=True)
df.columns


# In[3]:

dburl_table = 'sqlite:////tmp/foo.db::bozo'
#engine = create_engine()


# In[51]:

t = odo(fname, dburl_table, dshape=ds )


# In[36]:

type(t)


# In[101]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, bindparam
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


# In[5]:

dburl = 'sqlite:////tmp/foo.db'


# In[102]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float,create_engine, bindparam
from sqlalchemy.orm import Session

Base = declarative_base()
engine = create_engine(dburl, echo=True)



# In[106]:

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
    


# In[77]:

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


# In[109]:

session = Session(bind=engine)


# In[110]:


for t in ("a", "b"):
    print("Reading {}".format(t))
    
    df = pd.read_csv("/tmp/" + t +  ".csv")
    df = df.rename( columns={ "Adj Close": "AdjClose"} )
    
    df["Date"] = pd.to_datetime( df["Date"])
    
    rows = df.to_dict("records")
   
    for row in rows:
        row["Ticker"] = t
        # print ("Row: {}".format(row))
        rec= Price(**row)
        
        session.add(rec)
 
    try:
        session.flush()
    except SQLAlchemyError as e:
        print("Flush error: {}".format(e))
    
session.commit()
            
                        


# In[103]:

rsession = Session(bind=engine)


# In[122]:

q = rsession.query(Price)
for r in q.all():
    print(r.Ticker, r.Date)


# In[111]:

df_r = pd.read_sql(session.query(Price).statement, rsession.bind)
df_r.head()


# In[105]:

df_r_b = pd.read_sql( rsession.query(Price).filter(Price.Ticker== "b").statement, rsession.bind)
df_r_b.head()


# In[8]:

def setup_database(dburl, echo, num):
    engine = create_engine(dburl, echo=echo)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    return engine



# In[9]:

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



