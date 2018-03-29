
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


# ## Memorialize: a switch that causes repeatability by fixing the end date, etc.  
# ## It then writes is output to a "verify_" file for regression testing
# 

# In[3]:

Memorialize = False


# In[4]:

today = dt.datetime.combine( date.today(), dt.time.min)
if Memorialize:
    today = dup.parse("03/09/2018")
    
end_fixed = today
today

start = dup.parse("01/01/2000")
start


# In[5]:

gd = GetData()
univ = gd.existing()
univ.sort()

len(univ)


# In[6]:

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
   


# In[7]:

sector_tickers = list( sectors.values() )


# In[8]:

sector_tickers


# ## Download data

# In[9]:

get = False
if get:
    changed_tickers = gd.get_data( sector_tickers, start, today )
    len(changed_tickers)
    list( set(sector_tickers) - set(changed_tickers))


# ## Assemble data (already downloaded) into DataFrame
# - Note: The index will be a DateTime already, no need to convert from string. No need for DatetimeIndexTransformer
# - Note: the index will be restricted to dates from SPY, no need for RestrictToCalendarColTransformer

# In[10]:

price_df = GetDataTransformer(sector_tickers, cal_ticker="SPY").fit_transform( pd.DataFrame())
if Memorialize:
    price_df = price_df.loc[:end_fixed,:]
    
price_df.shape


# In[11]:

price_df.index.min()
price_df.index.max()
price_df.loc[:, idx["Adj Close",:]].shape

if Memorialize:
    gd.save_data( price_df.loc[:, idx["Adj Close",:]], "verify_sectors_raw_df.pkl")


# ## Compute returns

# In[12]:

type(price_df.index)


# In[13]:

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=False),
                           pctTrans,
                           GenRenameAttrsTransformer(lambda col: "Pct", level=0)
                          )
pct_df = pipe_pct.fit_transform(price_df)
pct_df.tail()


# In[14]:

if Memorialize:
    gd.save_data( pct_df, "verify_sectors_pct_df.pkl")


# ## Alternate way of creating Returns: drop attribute and re-add

# In[15]:

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


# In[16]:

import dateutil.parser as dup
import dateutil.relativedelta as rd

regWindow = rd.relativedelta(months=+6)
regStep   = rd.relativedelta(weeks=+4)

regStart = dup.parse("01/01/2000")
regEnd   = dup.parse("12/29/2017")
# regEnd   = dup.parse("02/28/2018")


# In[17]:

regParams = { "start": regStart, "end": regEnd, "window": regWindow, "step": regStep }
if Memorialize:
    gd.save_data( regParams, "verify_regParams.pkl")


# ## Compute the model: 
# $Return_{sector ticker} = \beta_0 + \beta * Return_{SPY} + \epsilon$

# In[18]:

rp = RegPipe( pct_df )
rp.indCols( [ idx["Pct", "SPY"] ] )
rp.regress( regStart, regEnd, regWindow, regStep)


# In[19]:

rp.beta_df.shape
rp.beta_df.tail()


# ## Compute residuals:
#  - For residual, don't roll beta: the date of the beta is the last date of the regression window
#  - Fill the beta backwards, so the in-sample beta is applied

# In[20]:

rollAmount = 0
fillMethod = "bfill"

rp.attrib_setup(pct_df, rp.beta_df, rollAmount, fillMethod)


# In[21]:

rp.attrib()

rp.retAttr_df.shape
rp.retAttr_df.loc[:"2017-12-29",:].tail()


# ## Demonstrate a non-rolling

# In[22]:

regStarts = regEnd - regWindow + timedelta(days=1)

pct_dfs = pct_df.loc[ regStarts:regEnd,:]

rps = RegPipe( pct_dfs )
rps.indCols( [ idx["Pct", "SPY"] ] )
rps.regressSingle()

rps.beta_df.shape
rps.beta_df.tail()


# In[23]:

if Memorialize:
    gd.save_data( rps.beta_df, "verify_beta_df.pkl")


# In[24]:

rollAmount = 0
fillMethod = "bfill"

rps.attrib_setup(pct_dfs, rps.beta_df, rollAmount, fillMethod)
rps.attrib()

rps.retAttr_df.shape
rps.retAttr_df.loc[:"2017-12-29",:].tail()


# In[25]:

sector_residuals = rps.retAttr_df.loc[:, idx["Error",:]]
sector_residuals.tail()


# In[26]:

if Memorialize:
    gd.save_data(sector_residuals, "sector_residuals.pkl")


# In[27]:

resStart = dup.parse("01/01/2016")


# ## OBSOLETE, replaced by trans.stack_residual

# In[28]:

from trans.stack import Stack
get_ipython().magic('aimport trans.stack')

s = Stack(pct_df)
stack = s.repeated(resStart, regEnd, regWindow, regStep)


# In[29]:

for stk in stack :
    suffix = stk[0].strftime("%Y%m%d")
    data = stk[1]
    
    if Memorialize:
        gd.save_data(data, "sector_residuals_{}.pkl".format(suffix))
    
             


# ## Residual stack

# In[30]:

get_ipython().magic('aimport trans.stacked.residual')
from trans.stacked.residual import Residual

rstack = Residual(debug=True)
rstack.init(df=pct_df, start=resStart, end=regEnd, window=regWindow, step=regStep)
resid_stack = rstack.repeated()
rstack.done()


# In[31]:

if Memorialize:
    gd.save_data( resid_stack, "verify_resid_stack.pkl")


# In[32]:

for stk in resid_stack :
    suffix = stk[0].strftime("%Y%m%d")
    data = stk[1]
    
    print("Stack {} shape: {}".format(stk[0], stk[1].shape))
    #gd.save_data(data, "sector_residuals_{}.pkl".format(suffix))         


# ## PCA stack

# In[33]:

get_ipython().magic('aimport trans.stacked.pca')

from trans.stacked.pca import PCA_stack

pstack = PCA_stack(debug=True)
pstack.init(stack=resid_stack)
pca_stack = pstack.repeated()
pstack.done()


# In[34]:

for stk in pca_stack :
    suffix = stk[0].strftime("%Y%m%d")
    data = stk[1]
    
    print("Stack {} shape: {}".format(stk[0], stk[1].shape))
    #gd.save_data(data, "sector_residuals_{}.pkl".format(suffix))
    


# ## Composed (residual, PCA) stack

# In[35]:

get_ipython().magic('aimport trans.stacked.pipeline')

from trans.stacked.pipeline import Pipeline_stack

resid_obj = Residual()
pca_obj   = PCA_stack()

plstack = Pipeline_stack([ resid_obj, pca_obj ], debug=True)

## Inelegant: manuallly init one member of pipe
resid_obj.init(df=pct_df, start=resStart, end=regEnd, window=regWindow, step=regStep)
#plstack.init(stack=resid_stack)
pl_stack = plstack.repeated()
plstack.done()


# In[36]:

for stk in pl_stack :
    suffix = stk[0].strftime("%Y%m%d")
    data = stk[1]
    
    print("Stack {} shape: {}".format(stk[0], stk[1].shape))
    #gd.save_data(data, "sector_residuals_{}.pkl".format(suffix))
    


# In[37]:



