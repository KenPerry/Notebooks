
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

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



# In[3]:

today = dt.datetime.combine( date.today(), dt.time.min)
today

start = dup.parse("01/01/2000")
start


# In[4]:

gd = GetData()
univ = gd.existing()
univ.sort()

len(univ)


# In[5]:

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
   


# In[6]:

sector_tickers = list( sectors.values() )


# In[28]:

changed_tickers = gd.get_data( sector_tickers, start, today )


# In[29]:

len(sector_tickers)
len(changed_tickers)
list( set(sector_tickers) - set(changed_tickers))


# In[7]:

price_df = GetDataTransformer(sector_tickers, cal_ticker="SPY").fit_transform( pd.DataFrame())
price_df.shape


# ## Compute returns

# In[30]:

type(price_df.index)


# In[39]:

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=False),
                           # RestrictToCalendarColTransformer( ("Adj Close", "SPY") ),
                           pctTrans,
                           # DatetimeIndexTransformer("Dt"),
                           GenRenameAttrsTransformer(lambda col: "Pct", level=0)
                          )
pct_df = pipe_pct.fit_transform(price_df)
pct_df.columns
pct_df.tail()


# In[40]:

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True), 
                           # RestrictToCalendarColTransformer( "SPY" ),
                           pctTrans,
                           # DatetimeIndexTransformer("Dt"),
                           # RestrictToNonNullTransformer("all"),
                           AddAttrTransformer('Pct')
                      )
pct_df = pipe_pct.fit_transform(price_df)
pct_df.shape
pct_df.tail()


# In[10]:

import dateutil.parser as dup
import dateutil.relativedelta as rd

regWindow = rd.relativedelta(months=+6)
regStep   = rd.relativedelta(weeks=+4)

regStart = dup.parse("01/01/2000")
regEnd   = dup.parse("12/29/2017")


# ## Compute the model: 
# $Return_{sector ticker} = \beta_0 + \beta * Return_{SPY} + \epsilon$

# In[11]:

rp = RegPipe( pct_df )
rp.indCols( [ idx["Pct", "SPY"] ] )
rp.regress( regStart, regEnd, regWindow, regStep)


# In[12]:

rp.beta_df.tail()


# ## Compute residuals:
#  - For residual, don't roll beta: the date of the beta is the last date of the regression window
#  - Fill the beta backwards, so the in-sample beta is applied

# In[15]:

rollAmount = 0
fillMethod = "bfill"

rp.attrib_setup(pct_df, rp.beta_df, rollAmount, fillMethod)


# In[17]:

rp.attrib()
rp.retAttr_df.loc[:"2017-12-29",:].tail()

