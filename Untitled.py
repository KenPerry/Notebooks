
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

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


# In[15]:

# df = gd.load_data("ret_and_beta_df.pkl")
ret_df = gd.load_data("ret_and_rolled_beta_df.pkl")
ret_df.shape
ret_df.tail()


# In[16]:

i = ret_df.index
i.min()
i.max()


# In[17]:

from trans.date_manip import Date_Manipulator
dm = Date_Manipulator( ret_df.index )
monthly = dm.periodic( dup.parse("01/01/2018"), datedelta.datedelta(months=1) )

end_of_monthly =  dm.periodic_end_of_month( dup.parse("12/15/2017"))
end_of_monthly[-10:]


# In[6]:

eom_in_idx = dm.periodic_in_idx_end_of_month( dup.parse("12/15/2017"))
eom_in_idx[-10:]


# In[7]:

month_r = dm.range_in_index( eom_in_idx)
month_r[-10:]


# In[19]:



ret_df.loc[ eom_in_idx,:].shape


# ## Add Monthly Pct return by
# ### feeding in df at monthly frequency (end-of-month)

# In[20]:

monthly_ret_attr = "Monthly Pct"

pipe_close = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )
                      )   

pipe_pct = make_pipeline(GenSelectAttrsTransformer(['Adj Close']), 
                         pctTrans,
                         GenRenameAttrsTransformer(lambda col:monthly_ret_attr, level=0)
                        )

monthly_ret_df = pipe_pct.fit_transform( ret_df.loc[ eom_in_idx,: ])
monthly_ret_df.tail()


# ## Rank the monthly returns
# ### Eliminate the non-universe tickers (SPY) from ranking

# In[10]:

univ_cols = [ (monthly_ret_attr, t) for t in ["FB", "AAPL", "AMZN", "NFLX", "GOOG"] ]
univ_cols


# In[11]:

rank_pl = make_pipeline( # GenSelectAttrsTransformer(univ_cols),
                        SelectColumnsTransformer(univ_cols),
                        rankTrans,
                        GenRenameAttrsTransformer(lambda col: col + ' rank', level=0)
                       )
rank_df = rank_pl.fit_transform( monthly_ret_df )
rank_df.tail()


# ## Add rank to df

# In[30]:

ret_and_rank_pl = DataFrameConcat( [ monthly_ret_df, rank_df])
ret_and_rank_df = ret_and_rank_pl.fit_transform( pd.DataFrame() )
ret_and_rank_df.tail()


# ## Shift the monthly returns fwd, as they are used for following month selection

# In[34]:

fwd_pl = make_pipeline( 
                        ShiftTransformer(1),
                        GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
monthly_ret_rolled_df = fwd_pl.fit_transform(ret_and_rank_df)
monthly_ret_rolled_df.tail()


# In[21]:

monthly_ret_and_rolled_ret_pl = DataFrameConcat( [ monthly_ret_df, monthly_ret_rolled_df])
monthly_ret_and_rolled_ret_df = ret_and_rolled_ret_pl.fit_transform( pd.DataFrame() )
monthly_ret_and_rolled_ret_df.tail()


# ## Join the monthly ranks to the daily series

# In[44]:

ret_and_rank_pl = DataFrameConcat( [ ret_df, monthly_ret_and_rolled_ret_df])
ret_and_rank_df = ret_and_rank_pl.fit_transform( pd.DataFrame())
ret_and_rank_df.loc[ "11/25/2017":"12/01/2017"]


# ## Roll the monthly ranks/returns backwards
# ### They were computed at end of previous month, pushed forward one month
# ### so they apply for the entirety of the month on which their month-end is date
# ### (hence, roll backward to each day)

# In[48]:


ret_and_rank_attrs = ret_and_rank_df.columns.get_level_values(0).unique().tolist()

pat = "^Monthly Pct .*rolled fwd$"
monthly_attrs = [ attr for attr in ret_and_rank_attrs if re.search(pat, attr) ]
monthly_attrs


# In[53]:

dfwd_pl = make_pipeline(GenSelectAttrsTransformer(monthly_attrs),
                        FillNullTransformer(method="bfill"),
                        GenRenameAttrsTransformer(lambda col: col + ' rolled fwd/back', level=0)
                         )
monthly_ret_rolled_df = dfwd_pl.fit_transform(ret_and_rank_df)
monthly_ret_rolled_df.loc[ "11/25/2017":"12/01/2017"]

