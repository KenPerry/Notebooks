
# coding: utf-8

# http://docs.sqlalchemy.org/en/latest/_modules/examples/performance/bulk_inserts.html

# In[4]:


from odo import *


# In[24]:

fname = "/tmp/t.csv"
ds = discover( resource(fname) )


# In[25]:

type(ds)


# In[50]:

dburl_table = 'sqlite:////tmp/foo.db::bozo'
#engine = create_engine()


# In[51]:

t = odo(fname, dburl_table, dshape=ds )


# In[36]:

type(t)


# In[45]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, bindparam
from sqlalchemy.orm import Session

Base = declarative_base()


# In[49]:

dburl = 'sqlite:////tmp/foo.db'


# In[60]:

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float,create_engine, bindparam
from sqlalchemy.orm import Session

Base = declarative_base()
engine = create_engine(dburl)



# In[61]:

class Price(Base):
    __tablename__ = "prices"
    ticker = Column(String(255), primary_key=True)
    date   = Column(Date, primary_key=True)
    adjClose = Column(Float)
    close   = Column(Float)
    high    = Column(Float)
    low     = Column(Float)
    open    = Column(Float)
    volume  = Column(Float)
    


# In[75]:




# In[92]:

session = Session(bind=engine)

def f_clean(f):
    f = f.replace(" ", "",1)
    f = f[0].lower() + f[1:]
    return f

for t in ("a", "b"):
    with open("/tmp/" + t + ".csv") as fp:
        print("Reading {}".format(t))
        first = fp.readline()
        first = first.rstrip("\n")
        
        cols =  first.split(",")
        cols = list ( map( lambda f: f_clean(f), cols) )
       
        print("Cols: ", "|".join(cols))

        line = fp.readline()
        while line:
            line = line.rstrip("\n")
            fields = line.split(",")
            
            d = dict( zip(cols, fields))
            d["ticker"] = t
            
            print("d: {}".format(d))
            rec = Price(**d)
            
            session.add(rec)
            
            line=fp.readline()
            
    session.flush()
    
session.commit()
            
                        


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



