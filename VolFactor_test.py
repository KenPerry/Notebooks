
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
import dateutil.relativedelta as rd
import dateutil.parser as dup

from trans.date_manip import Date_Manipulator


# In[3]:

Memorialize = False


# In[4]:

today = dt.datetime.combine( dt.date.today(), dt.time.min)
if Memorialize:
    today = dup.parse("03/09/2018")
    
start = dup.parse("01/01/2018")
end   = today



# In[5]:

import trans.dataprovider.alphavantage as aa
get_ipython().magic('aimport trans.dataprovider.alphavantage')

aa_dp = aa.Alphavantage()


# In[6]:

import trans.dataprovider.odo as odo_reader
get_ipython().magic('aimport trans.dataprovider.odo')

from sqlalchemy.ext.declarative import declarative_base

dburl="sqlite:////tmp/full.db"
decBase_r = declarative_base()

odr = odo_reader.ODO(dburl, declarative_base=decBase_r, provider=aa_dp)

df_aa = odr.get(tickers=["FB", "AAPL", "AMZN", "NFLX", "GOOG" ], start="2018-03-01")
df_aa.shape


# In[7]:

from trans.data import GetData
gd = GetData()

universe = gd.get_r1000_tickers()
universe.sort()
len(universe)


# In[8]:

if Memorialize:
    universe = ["FB", "AAPL", "AMZN", "NFLX", "GOOG" ]


# In[9]:

import trans.quantfactor.volatility as vf
get_ipython().magic('aimport trans.quantfactor.volatility')

get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.quantfactor.base')

v = vf.Volatility(universe=universe, dataProvider=odr)


# In[10]:

daily_price_df = v.load_prices(start=start, end=end)


# In[11]:

daily_price_df.shape


# In[12]:

if Memorialize:
    gd.save_data(price_df.loc[:, "Adj Close"], "verify_volat_raw_df.pkl")


# In[13]:

existing_tickers = list(daily_price_df.columns.get_level_values(1).unique())
existing_tickers.sort()
len(existing_tickers)


# In[14]:

price_attr = "Adj Close"
ret_attr = "Pct"
daily_ret_df = v.create_dailyReturns(price_attr, ret_attr )


# In[15]:

daily_ret_df.shape


# In[16]:

dm = Date_Manipulator( v.price_df.index )
eom_in_idx = dm.periodic_in_idx_end_of_month(end)
v.set_endDates( eom_in_idx )


# In[17]:

from datetime import timedelta
vol_window = timedelta(days=30)
 
v.create_period_attr( v.daily_ret_df.loc[:, idx[ret_attr,:]], start, end, vol_window, "Volatility")


# In[18]:

daily_rank_df = v.create_ranks()


# In[19]:

if Memorialize:
    daily_rank_df.loc[ "2018-01-31":"2018-03-02"]


# In[20]:

factor_df = v.create_factor()


# In[21]:

factor_df.tail()


# ## Simplified version of factor creation: combine all steps into one method "create"

# In[24]:

v2 = vf.Volatility(universe=universe, dataProvider=odr)
factor_df2 = v2.create(start=start, end=end,
               price_attr="Adj Close", ret_attr="Ret", rank_attr="Factor",
               window=timedelta(days=30)
             )
factor_df2.tail()


# ## Create residuals

# ### Create DataFrame for residual computation

# In[ ]:

resid_input_df =pd.concat( [ factor_df.loc[:, idx[ret_attr, "Port net"]],
    v.daily_ret_df.loc[:, idx[ret_attr,"SPY"]]
           ], axis=1
         )


# In[ ]:

get_ipython().magic('aimport trans.stacked.residual')
from trans.stacked.residual import Residual

(resStart, resEnd) = (resid_input_df.index.min(), resid_input_df.index.max())
regWindow = rd.relativedelta(months=+2)
regStep   = rd.relativedelta(weeks=+4)


# In[ ]:

ret_attr


# In[ ]:

get_ipython().magic('aimport trans.reg')
get_ipython().magic('aimport trans.regpipe')


# In[ ]:

rstack = Residual(indCols=[ idx[ret_attr, "SPY"] ], debug=True)
rstack.init(df=resid_input_df, start=resStart, end=v.endDates[-1], window=regWindow, step=regStep)
resid_stack = rstack.repeated()
rstack.done()


# In[ ]:

(l, resid_last_df) = resid_stack[0]
resid_last_df.columns 

for r in resid_stack:
    (label, df) = r
    print("{l}: from {s} to {e}".format(l=label, s=df.index.min(), e=df.index.max()))


# ## Find the sensitivity of selected names

# In[ ]:

names = [ "BA", "PAGS", "CTXS", "TSLA", "NFLX"< "MSFT", "EA", "BABA", "AYI", "SABR", "TXN" ]


# In[ ]:

names = existing_tickers


# ## Get returns of members of the universe

# In[ ]:

from trans.gtrans import *
# Get the data for the tickers in self.universe
price_df = GetDataProviderTransformer(names, cal_ticker="SPY", dataProvider=odr).fit_transform( pd.DataFrame())

# Limit the output to date range from start to end
price_df = price_df.loc[ start:, ]
        
price_df = price_df.loc[ :end, ]
price_attr = "Adj Close"
price_shifted_attr = price_attr + " prior"

daily_ret_pl = GenRetAttrTransformer( price_attr, price_shifted_attr, ret_attr, 1 )
daily_ret_df = daily_ret_pl.fit_transform( price_df )


# ## Compute betas wrt vol factor

# ### Rename the "Error" attribute to ret_attr

# In[ ]:

volat_resid_df = resid_last_df.loc[:, idx["Error",:]]
volat_resid_df.tail()
lev0 =  volat_resid_df.columns.levels[0].tolist()
lev0[ lev0.index("Error")] = ret_attr


# ### Rename the ticker to "Volat factor", and the attibute to ret_attr

# In[ ]:

volat_resid_df.columns.set_levels(["Volat factor"], level=1, inplace=True)
volat_resid_df.columns.set_levels(lev0, level=0, inplace=True)

volat_resid_df.tail()


# ## Compute the std dev. of the Volatility factor (outperformance)

# In[ ]:

volat_resid_df.loc[ v.endDates[-1] -rd.relativedelta(months=+1):, :].std()


# ## Put the returns of the universe and the Volatlity factor in same DataFrame

# In[ ]:

reg_input_df = DataFrameConcat( [ daily_ret_df.loc[:, idx[ret_attr,:]], volat_resid_df] ).fit_transform(pd.DataFrame())


# In[ ]:

reg_input_df.columns
reg_input_df.loc[ volat_resid_df.index.max() - rd.relativedelta(months=+1):]


# ## Perform rolling regression of each member of universe vs. Volat factor

# In[ ]:

from trans.regpipe import RegPipe
rp = RegPipe( reg_input_df )
rp.indCols( [ idx[ret_attr, "Volat factor"] ] )
rp.regress( resStart, v.endDates[-1], regWindow, regStep)


# In[ ]:

rp.beta_df


# In[ ]:

gd.save_data(rp.beta_df, "/tmp/beta_wrt_volfactor_04302018.pkl")


# In[ ]:

rp.beta_df.loc[ rp.beta_df.index.max(), idx["Beta 1",:]].to_csv("/tmp/beta_wrt_volfactor_04302018.csv")


# In[ ]:

missing_universe = list(set(universe) - set(existing_tickers))
missing_universe.sort()
print("No data for following R1000 names: {}".format(", ".join(missing_universe)))

universe = list(set(universe) - set(missing_universe))
universe.sort()
print("Available universe has {} tickers".format(len(universe)))

