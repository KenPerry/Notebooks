
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import os
import datetime as dt
from datetime import date
from datetime import timedelta
import dateutil.parser as dup
import dateutil.relativedelta as rd


# In[3]:

get_ipython().magic('aimport trans.data')
from trans.data import GetData

start = dt.datetime(2000, 1, 1)
today = dt.datetime.combine( date.today(), dt.time.min)
today
gd = GetData()


# In[4]:


from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe


# ## Usually the Adj Close and Close fields are close in value in Yahoo
# ### However, around dividends, the fields diverge backwards in time from the payment date
# ### It seems that eventually Yahoo repopulates the data and the difference disappears again
# ### There is a SPY divident of 1.09 on 3/16/2018 that is causing a big divergence.
# ### The percent changes are close EXCEPT on the start of the week of the dividend, i.e., 03/09/2018
# ## Althoug Adj Close is more correct (since two tickers pay dividends at different times), we may need to use Close (which is what PX_LAST on Bloomber is) to eliminate one big returns

# In[5]:

close_field = "Adj Close" # "Close"


# In[6]:

existing_tickers = gd.existing()
existing_tickers.sort()
len(existing_tickers)


# In[7]:

req_tickers = [ "FB", "NFLX", "BA", "AVGO", "MCHP", "AKAM", "MRVL", "EA", "MSFT"]
exp_tickers = list( set(existing_tickers).union( req_tickers ) )
exp_tickers.sort()
len(exp_tickers)


# In[8]:

cleaned = gd.clean_data( exp_tickers )
cleaned


# In[9]:

update_data = False


# In[10]:

if update_data:
    changed_tickers = gd.get_data( exp_tickers, start, today )
    print("Number of tickers updated: {}", len(changed_tickers))
        


# ## Create sector residuals

# In[11]:

sector_tickers = ['SPY',
 'XLY',
 'XLP',
 'XLE',
 'XLF',
 'XLV',
 'XLI',
 'XLB',
 'XLRE',
 'XLK',
 'XTL',
 'XLU']

price_df = GetDataTransformer(sector_tickers, cal_ticker="SPY").fit_transform( pd.DataFrame())


# In[12]:

pipe_pct   = make_pipeline(GenSelectAttrsTransformer([ close_field], dropSingle=False),
                           pctTrans,
                           GenRenameAttrsTransformer(lambda col: "Pct", level=0)
                          )
pct_df = pipe_pct.fit_transform(price_df)


# ## NOTE: we are NOT using the Residual stack b/c that object only gives residuals, and not access to the betas (which we want to observe)

# In[13]:

end = dup.parse("03/16/2018")
window = rd.relativedelta(months=+6)

regStarts = end - window + timedelta(days=1)
regStarts, end
pct_dfs = pct_df.loc[ regStarts:end,:]

rps = RegPipe( pct_dfs )
rps.indCols( [ idx["Pct", "SPY"] ] )
rps.regressSingle()

rps.beta_df.shape


# In[14]:

rps.beta_df
rps.beta_df.to_csv("sector_residuals_beta_03162018.csv")


# In[15]:

rollAmount = 0
fillMethod = "bfill"

rps.attrib_setup(pct_dfs, rps.beta_df, rollAmount, fillMethod)
rps.attrib()

rps.retAttr_df.shape

sector_residuals = rps.retAttr_df.loc[:, idx["Error",:]]


# In[16]:

sector_residuals.to_csv("sector_residuals_03162018.csv")


# ## Rename residuals attribute from Error to Tilt

# In[17]:

pipe_rename   = make_pipeline(GenSelectAttrsTransformer(['Error'], dropSingle=False),
                              GenRenameAttrsTransformer(lambda col: "Tilt", level=0)
                          )
sector_residuals = pipe_rename.fit_transform(sector_residuals)


# ## Retrieve the non-tilt tickers and compute returns

# In[18]:

price_df = GetDataTransformer(req_tickers, cal_ticker="SPY").fit_transform( pd.DataFrame())
pct_df = pipe_pct.fit_transform(price_df)


# ## Create single DataFrame with ticker and tilt returns

# In[19]:

pct_df.index.min(), pct_df.index.max()
sector_residuals.index.min(), sector_residuals.index.max()


# In[20]:

reg_df = DataFrameConcat( [ pct_df.loc[ sector_residuals.index.min():sector_residuals.index.max(),:], sector_residuals]).fit_transform( pd.DataFrame())


# ## Do the regression for the Info Tech sector

# In[22]:

xlk = tuple(("Tilt", "XLK"))


# In[23]:

cols = list( (map(lambda t: ("Pct", t), req_tickers)) )
cols.append( ("Tilt", "XLK"))
cols

# cols = [ ("Pct", "BA"), ("Pct", "SPY") , ("Tilt", "XLK")]


# In[24]:

end = dup.parse("03/16/2018")
window = rd.relativedelta(months=+6)

regStarts = end - window + timedelta(days=1)
regStarts, end
reg_dfs = reg_df.loc[ regStarts:end,cols]

rps = RegPipe( reg_dfs )
rps.indCols( [ idx["Pct", "SPY"],
               idx["Tilt", "XLK"] ] )
rps.regressSingle()

rps.beta_df.shape


# In[25]:

reg_dfs.to_csv("ticker_returns_03162018.csv")
reg_dfs.tail()


# In[42]:

tech_tickers = [ "AKAM", "AVGO", "EA", "FB", "MCHP", "MRVL", "MSFT", "NFLX"]


# In[43]:

rps.beta_df.loc[:, idx["Beta 1", tech_tickers]]
rps.beta_df.loc[:, idx["Beta 2", tech_tickers]]
rps.beta_df.to_csv("ticker_betas_03162018.csv")


# ## Do the attribution

# In[27]:

rollAmount = 0
fillMethod = "bfill"

rps.attrib_setup(reg_dfs, rps.beta_df, rollAmount, fillMethod)
rps.attrib()

rps.retAttr_df.shape
rps.retAttr_df.to_csv("ticker_attr_03162018.csv")
ticker_alpha = rps.retAttr_df.loc[:, idx["Error",:]]


# In[28]:

10000* rps.retAttr_df.loc[:, idx["Contrib from SPY",:]].tail()


# In[29]:

10000* rps.retAttr_df.loc[:, idx["Contrib from XLK",:]].tail()


# In[30]:

10000* rps.retAttr_df.loc[:, idx["Error",:]].tail()


# ## Combine the returns and attribution DataFrames to ease presentation

# In[31]:

comb_df = DataFrameConcat( [ reg_dfs, rps.retAttr_df]).fit_transform( pd.DataFrame())


# In[78]:

get_ipython().magic('aimport trans.gtrans')
from trans.gtrans import *

# PnL attribution period: should end no later than regression end "end"ullpass

last  = end #"2018-03-16"
first = "2018-03-12"

# Create cumulative returns
cumret_pl = make_pipeline( CumRetTransformer() )
cumret_df = cumret_pl.fit_transform( comb_df.loc[ first:last,:]) 

# Create attribution, denominated in percent of ticker retrun
attrs_pct = []
attrs = []
for t in tech_tickers:
    ret_per = cumret_df.loc[ last, idx[:,t]]
    # ret_per
    ticker_ret_per = cumret_df.loc[last, idx["Pct", t]]

    attr_for_t_pct = 100 *ret_per/ticker_ret_per
    attrs_pct.append(attr_for_t_pct.unstack(level=1))
    attrs.append(10000 * ret_per.unstack(level=1))

attr_for_t.unstack(level=1)

# Glue together the ticker attributes horizontally
summary     = pd.concat( attrs, axis=1)
summary
summary_pct = pd.concat( attrs_pct, axis=1)
summary_pct
summary_pct.to_csv("summary_pct_03162018.csv")


# In[33]:

sector_residuals.loc["2018-01-02":"2018-03-16", idx["Tilt", [ "XLI", "XLK"]]].plot()


# ## Verify one regression

# In[34]:

from sklearn import linear_model
lm = linear_model.LinearRegression()

reg_dfs.shape
ind_df = reg_dfs.loc[:, [ ("Pct", "SPY"), ("Tilt", "XLK")]]
dep_df = reg_dfs.loc[:, [ ("Pct", "AKAM")]]
model = lm.fit(ind_df, dep_df)
model.intercept_
model.coef_

