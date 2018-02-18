
# coding: utf-8

# In[57]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[58]:

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.reg')
get_ipython().magic('aimport trans.date_manip')


import pandas as pd
idx = pd.IndexSlice

from trans.data import GetData
from trans.gtrans import *
from trans.reg import Reg
import datetime as dt
import dateutil.parser as dup
import datedelta

from trans.date_manip import Date_Manipulator

gd = GetData()


# In[59]:

# df = gd.load_data("ret_and_beta_df.pkl")
ret_df = gd.load_data("ret_and_rolled_beta_df.pkl")
ret_df.shape
ret_df.tail()


# In[60]:

i = ret_df.index
i.min()
i.max()


# In[61]:

from trans.date_manip import Date_Manipulator
dm = Date_Manipulator( ret_df.index )
monthly = dm.periodic( dup.parse("01/01/2018"), datedelta.datedelta(months=1) )

end_of_monthly =  dm.periodic_end_of_month( dup.parse("12/15/2017"))
end_of_monthly[-10:]


# In[62]:

eom_in_idx = dm.periodic_in_idx_end_of_month( dup.parse("12/15/2017"))
eom_in_idx[-10:]


# In[63]:

month_r = dm.range_in_index( eom_in_idx)
month_r[-10:]


# In[64]:

price_monthly_df = ret_df.loc[ eom_in_idx,: ]
price_monthly_df.shape


# ## Create price and shifted price

# In[81]:

from trans.gtrans import *


# In[82]:

monthly_ret_pl = GenRetAttrTransformer( price_attr, price_attr_shift_1m, monthly_ret_attr, 1 )
monthly_ret_df = m_ret_pl.fit_transform( price_monthly_df )
monthly_ret_df.tail()


# ## Rank the monthly returns

# In[83]:

univ = ["FB", "AAPL", "AMZN", "NFLX", "GOOG"]


# ## Use the monthly_ret_attr but ONLY for tickers in the universe !

# In[88]:

rank_pl = make_pipeline( GenSelectAttrsTransformer( [ monthly_ret_attr ],
                                                    dropSingle=True
                                                  ),
                        SelectColumnsTransformer( univ ),
                        rankTrans,
                        AddAttrTransformer( monthly_ret_attr + " rank")
                       )
rank_df = rank_pl.fit_transform( monthly_ret_df )
rank_df.tail()
                 


# In[96]:

pd.concat(  [rank_df,  pd.DataFrame(index=ret_df.index) ], axis=1 ).loc[ "11/25/2017":"12/01/2017"]


# ### Alternate way:
# ### Eliminate the non-universe tickers (SPY) from ranking

# In[90]:


univ_cols = [ (monthly_ret_attr, t) for t in univ ]
univ_cols


# In[91]:

rank_pl = make_pipeline( 
                        SelectColumnsTransformer(univ_cols),
                        rankTrans,
                        GenRenameAttrsTransformer(lambda col: col + ' rank', level=0)
                       )
rank_df = rank_pl.fit_transform( monthly_ret_df )
rank_df.tail()


# ## Add rank to month pct

# In[12]:

ret_and_rank_pl = DataFrameConcat( [ monthly_ret_df, rank_df])
ret_and_rank_df = ret_and_rank_pl.fit_transform( pd.DataFrame() )
ret_and_rank_df.tail()


# ## Shift the monthly returns fwd, as they are used for following month selection

# In[13]:

fwd_pl = make_pipeline( 
                        ShiftTransformer(1),
                        GenRenameAttrsTransformer(lambda col: "Prior " + col, level=0)
                         )
monthly_ret_rolled_df = fwd_pl.fit_transform(ret_and_rank_df)
monthly_ret_rolled_df.tail()


# In[14]:

monthly_ret_and_rolled_ret_pl = DataFrameConcat( [ monthly_ret_df, monthly_ret_rolled_df])
monthly_ret_and_rolled_ret_df = monthly_ret_and_rolled_ret_pl.fit_transform( pd.DataFrame() )
monthly_ret_and_rolled_ret_df.tail()


# ## Join the monthly ranks to the daily series

# In[15]:

ret_and_rank_pl = DataFrameConcat( [ ret_df, monthly_ret_and_rolled_ret_df])
ret_and_rank_df = ret_and_rank_pl.fit_transform( pd.DataFrame())
ret_and_rank_df.loc[ "11/25/2017":"12/01/2017"]


# ## Fill the daily ranks but rolling the monthly ranks backwards
# ### They were computed at end of previous month, pushed forward one month
# ### so they apply for the entirety of the month on which their month-end is date
# ### (hence, roll backward to each day)

# In[16]:


ret_and_rank_attrs = ret_and_rank_df.columns.get_level_values(0).unique().tolist()

pat = "^Prior Month Pct"
monthly_attrs = [ attr for attr in ret_and_rank_attrs if re.search(pat, attr) ]
monthly_attrs


# In[17]:

dfwd_pl = make_pipeline(GenSelectAttrsTransformer(monthly_attrs),
                        FillNullTransformer(method="bfill"),
                        GenRenameAttrsTransformer(lambda col: col + ' filled', level=0)
                         )
monthly_ret_rolled_df = dfwd_pl.fit_transform(ret_and_rank_df)
monthly_ret_rolled_df.loc[ "11/25/2017":"12/01/2017"]


# ## Add the filled daily ranks to the big dataframe

# In[101]:

monthly_rank_attr = monthly_ret_attr + ' rank'
next_period_rank_pl = GenRankEndOfPeriodAttrTransformer(
    monthly_ret_df,
    monthly_ret_attr,
    univ,
    monthly_rank_attr
)
next_period_rank_df = next_period_rank_pl.fit_transform( ret_df )

next_period_rank_df.loc[ "11/25/2017":"12/01/2017"]


# In[18]:

ret_and_rank_2_pl = DataFrameConcat( [ ret_and_rank_df, monthly_ret_rolled_df])
ret_and_rank_2_df = ret_and_rank_2_pl.fit_transform( pd.DataFrame())
ret_and_rank_2_df.loc[ "11/25/2017":"12/01/2017"]


# ## Use the universe to select both 
# ### 1. the data source ("Pct": daily returns)
# ### 2. the weight source ("Prior month rank filled")
# ###
# ### Need to use same universe to make sure columns are aligned
# ### NOTE: alignment of columns guaranteed if multipld DataFrames, NOT is convert to  NumPy ndarray
# ### Need to use universe b/c Pct attribute may have non-universe tickers

# In[19]:


pref = "Prior Month Pct"
ret_attr = "Pct"
src_attr = pref + " rank filled"

univ_ret_cols = [ (ret_attr, t) for t in univ ]
univ_ret_cols

univ_src_cols = [ (src_attr, t) for t in univ ]
univ_src_cols


# In[20]:

hi_rank, lo_rank = 5, 1
hmlWtTrans = DataFrameFunctionTransformer(func = lambda s: (s >= hi_rank) * 1.0 + (s <= lo_rank) * -1.0)


# In[21]:

wt_pl = make_pipeline( SelectColumnsTransformer(univ_src_cols),
                       hmlWtTrans,
                       GenRenameAttrsTransformer(lambda col: col + ' wt', level=0)
                     )
wt_df = wt_pl.fit_transform(ret_and_rank_2_df)

wt_df.loc[ "11/25/2017":"12/01/2017"]


# In[22]:

dret_pl = make_pipeline( SelectColumnsTransformer(univ_ret_cols),
                     )
dret_df = dret_pl.fit_transform(ret_and_rank_2_df)

dret_df.loc[ "11/25/2017":"12/01/2017"]


# ### Create the HML returns by multiplying Data Source by Weights
# ### Since Level 0 are different attributes, have to drop in order to multiply

# In[23]:

wt_df.columns   = wt_df.columns.droplevel(0)
dret_df.columns = dret_df.columns.droplevel(0)


# In[24]:

wted_ret_df = wt_df * dret_df
wted_ret_df.loc[ "11/25/2017":"12/01/2017"]

wted_ret_df.sum(axis=1).loc[ "11/25/2017":"12/01/2017"]

