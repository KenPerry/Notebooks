
# coding: utf-8

# In[282]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[283]:

from pandas.io.data import DataReader
# from pandas_datareader import data, wb
import pandas as pd
import numpy as np


# In[284]:

symbols = ['MSFT', 'GOOG', 'AAPL']

import datetime
start = datetime.datetime(2016, 12, 31)
end = datetime.datetime(2017, 1, 10)


# # Get data from Yahoo

# In[285]:

data = dict((sym, DataReader(sym, "yahoo", start, end))
          for sym in symbols)



# # data is a hash, keyed by ticker, whose elements are pandas
# ## Add a column named 'New' to the panda for each ticker

# In[286]:

dfa = data['AAPL']
dfa['New']  = np.arange( dfa.shape[1] )

dfg = data['GOOG']
dfg['New'] = np.arange( dfg. shape[1] )[::-1]


# # Concat the pandas for each ticker into one panda
# ##  Dimension 1 is date: this allows date-oriented manipulation on all attributes for all tickers
# ## Dimension 2 is indexed by a MultiIndex: (ticker, attribute) pair

# In[287]:

d=pd.concat( [ dfa, dfg], axis=1, keys=['AAPL', 'GOOG'], names=["ticker", "attr"])
d


# # Perform a date-oriented operation: percent change on all columns

# In[288]:

d_pct = d.pct_change()
d_pct


# ## Rename the columns of the percent change panda with suffice "Pct"

# ### First, create a map from old column name to new columns name

# In[289]:

renameDict = dict( [ (c, c + ' Pct') for c in list( d_pct.columns.levels[1] ) ] )
renameDict


# In[290]:

d_pct_renamed = d_pct.rename(columns= renameDict)
d_pct_renamed


# ## A digression: bug in Pandas ?
# ## NO: was a problem only when rename used the "idx=string" arg !
# ### 
# ### Note the type of the index of d_pct before and after the renaming
# #### d_pct.index are Dates, but d_pct_renamed.index are strings

# In[291]:

d_pct.index
d_pct_renamed.index
list(d_pct.index) == list(d_pct_renamed.index)


# ### So the fact that d_pct_renamed.index is not same as d_pct.index means that
# ### join operations (like concat) won't find common keys
# ### Need to fix this by changing d_pct_renamed.index back to Date

# In[292]:

# d_pct_renamed.index = d_pct.index
# list(d_pct.index) == list(d_pct_renamed.index)


# # We can now modify Panda d to have Pct columns by joining it with d_pct_renamed

# In[293]:

d_new = pd.concat( [ d, d_pct_renamed ], axis=1)
d_new


# ## Now let's try a column-oriented (MultiIndex) operation, per ticker

# In[294]:

d_new_grouped = d_new.groupby(axis=1, level=0)

for name, group in d_new_grouped:
    print("Name: {n}, \n\tGroup: {g}\n\n".format(n=name, g=group))

# This is min across all attributes, per ticker
d_new_min_per_ticker = d_new_grouped.min()
d_new_min_per_ticker


# ## We need to add a MultiIndex column name to d_new_min

# In[295]:

min_per_ticker_labels = [ (t, 'Ticker attr rank') for t in d_new_min_per_ticker.columns]
col_multi = pd.MultiIndex.from_tuples(min_labels, names=['t', 'a'])
col_multi


# ### Change the columns via reindex, "broadcasting" (joining) on MultiIndex name 't'
# #### i.e., use the name 't' from the MultiIndex to index into d_new_min.columns

# In[296]:

d_new_min_per_ticker_multi = d_new_min_per_ticker.reindex(columns=col_multi, level='t')
d_new_min_per_ticker_multi.columns
d_new_min_per_ticker_multi


# ### Concatenate and use sortlevel (as usual) to re-order the MultiIndex 
# ### into canonical form

# In[297]:

d_new2 = pd.concat( [ d_new, d_new_min_per_ticker_multi], axis=1).sortlevel(axis=1)
d_new2


# ## Now, let's do a column-oriented operation, per attribute

# In[304]:

d_new_grouped = d_new2.groupby(axis=1, level=1)

#for name, group in d_new_grouped:
#    print("Name: {n}, \n\tGroup: {g}\n\n".format(n=name, g=group))

# This is rank across all tickers, per attribute
d_new_rank_per_attr = d_new_grouped.rank()
d_new_rank_per_attr


# ### Add a suffix to the column names

# In[305]:

rank_labels = [ a + ' Rank' for a in d_new_rank_per_attr.columns.get_level_values(1) ]

# Note: we have to assign back to d_new_rank_per_attr.columns as the set_levels operation returns a 
##  MultiIndex -- i.e., it does not modify the DataFrame
d_new_rank_per_attr.columns = d_new_rank_per_attr.columns.set_levels(rank_labels, level='attr')
d_new_rank_per_attr


# ## Finally: concatenate new columns and sort the MultiIndex

# In[306]:

d_new3 = pd.concat( [ d_new, d_new_rank_per_attr], axis=1).sortlevel(axis=1)
d_new3


# ### Demonstrate that the Close Pct Rank is correct for each day

# In[307]:

d_new3.loc[ :, idx[:, [ 'Close Pct', 'Close Pct Rank'] ] ]


# ## Suppose we want to choose only the highest rank ticker per day

# In[308]:


rank1 = d_new3.loc[:, idx[:,'Close Pct Rank']] == 1
rank1


# In[ ]:

d_new3.loc[:, idx[:,'Close Pct Rank']] * rank1


# ## Might think can multiply 'Close Pct' by rank1 as a way of selecting 
# ## one attribute ('Close Pct') based on a filter of another ('Close Pct Rank')
# ## But since column labels are different, can't multiply.
# ## First need to rename columns of rank1 to match the subset of d_new2

# In[309]:

type(rank1)
d_new3.loc[:, idx[:,'Close Pct']] * rank1

rank1_mod = rank1.rename(columns={ 'Close Pct Rank': 'Close Pct'})
rank1_mod
d_new3.loc[:, idx[:,'Close Pct']] * rank1_mod


# ## This shows that the column oriented approach for data representation
# ## with separate columns per ticker, makes selecting a different
# ## ticker per date (i.e., using rank1) difficult

# ## Row and column indexing don't seem to be on equal footing.
# ## Seems that row index has more flexible syntax for filtering:
# ### Consider: select rows where some column condition is true

# In[310]:

aapl_rank1 =d_new3.loc[:, idx['AAPL', 'Close Pct Rank']] == 1
type(aapl_rank1)
aapl_rank1
d_new3.loc[ aapl_rank1 ]


# ## How would we do equivalent on column filtering ?
# 
# ## Form a mask for each ticker based on 'Close Pct Rank'
# ## Apply the mask for each ticker to it's 'Close Pct' column
# ## This gives a sub-series for each ticker
# ## The sub-series of different tickers are disjoint on date so can concat
# ##  them into a "union" series

# In[442]:

dc = d_new3.loc[:, idx[:, [ 'Close Pct', 'Close Pct Rank'] ] ]

tickers = list(dc.columns.levels[0])

# Form a mask for each ticker's 'Close Pct Rank' indicating whether it is rank 1
r1 = dc.loc[:, idx[:, 'Close Pct Rank']] == 1
r1

# Form a dict mapping ticker to the 'Close Pct' for the ticker
close_pct_dict = dict( [ (t, dc.loc[:, idx[t, 'Close Pct']]) for t in tickers ] )

print("Close Pct Dict:\n")
close_pct_dict

# Iterate thru the tickers, apply the mask for the ticker to the 'Close Pct'
# Result is an array of Series, one per ticker
s = [ close_pct_dict[t][ r1.loc[:, idx[t, 'Close Pct Rank'] ] ] for t in tickers ]

# Join the series for each ticker (which are disjoint on dates) into one union series
pd.concat(s)


# In[595]:

ticker_groups = d_new3.groupby(axis=1,level=0)

def foo(df):
    # print("foo df:\n", df)
    df2 = df.copy()
    df2.columns = df.columns.droplevel(0)
    # print("foo df2:\n", df2)
    
    close_pct = df2.loc[:, 'Close Pct']
    
    close_pct_rank = df2.loc[:,'Close Pct Rank']
    # print("close_pct rank: ", close_pct_rank)
    rank1 = (close_pct_rank == 1)
    # print("rank1: ", rank1)
    s = close_pct.loc[ rank1 ]
    df3 = pd.DataFrame(s)
    print("df3: ", df3)
    
   
    return df3
    
  
closePct_by_tg = ticker_groups.apply(foo)
closePct_by_tg
closePct_by_tg.columns = closePct_by_tg.columns.set_names(['ticker', 'foo'])
closePct_by_tg.columns = closePct_by_tg.columns.droplevel(1)
closePct_by_tg

# Use a non-group version of apply to iterate over the rows
# NOTE: df (the arg to lambda) still has an index (i.e., the ticker); need to drop it via list()
closePct_by_tg.apply(lambda df: pd.Series(list(df.dropna())), axis=1)


# In[378]:

def foo(df):
    print("Df: {d}\n\t".format(d=df))
   
    return df

dx = d_new3.loc[ :, idx[:, [ 'Open', 'Close']]] 
dx


dx.apply( foo )
type(dx)

dx.groupby(axis=1, level=0).apply(foo)


# ## Maybe, try an alternative data representation
# ## The Quantopian approach:
# ## Row index is MultiIndex (date, ticker) and columns are attributes
# ## Then, filtering rows amounts to choosing a subset of tickers per date
# ## Downside: have to do groupby('ticker') to do timeseries operations

# # Alternative data representation:
# ## Dimension 1 is index by a MultiIndex: (date, ticker) pair

# In[549]:

d
dd = d.stack(level=0)
dd


# ## Now, can't do dimension 1 oriented operations like pct_change
# ## because row after  (date1, ticker1) is (date1, ticker2) NOT (date2, ticker1)
# ## Solve this by grouping by ticker and THEN peforming dimension 1 operation

# ### Let's see how the groupby works:

# In[ ]:

dd_grouped = dd.groupby(level=1)
for name, group in dd_grouped:
    print("Name: {n}, \n\tGroup: {g}\n\n".format(n=name, g=group))


# In[ ]:

dd_pct = dd.groupby(level=1).pct_change()
dd_pct


# ## Rename the columns of dd_pct with the "Pct" suffix, as for the first data representation

# In[ ]:

dd_pct_renamed = dd_pct.rename(index=str, columns= renameDict)
dd_pct_renamed


# ## Again, rename as corrupted the type of the index
# ### i.e., dd_pct_renamed.index is now String, not Date
# # NOTE: it is "corrupted" only is we use the 'index=str' arg in rename
# ## We preserve the error (and the "fix" to illustrate)

# In[ ]:

dd_pct.index
dd_pct_renamed.index
list(dd_pct.index) == list(dd_pct_renamed.index)


# ## We fix this by copying the original index, as in the first data representation

# In[ ]:

dd_pct_renamed.index = dd_pct.index
dd_new = pd.concat( [ dd, dd_pct_renamed], axis=1)
dd_new


# ## Let's try accessing the new columns
# ### We will discover that some indexing features, like IndexSlice,
# ### work ONLY if the index or MultiIndex is properly sorted
# ### The .loc operation below won't work until we sort the MultiIndex
# ### See: http://pandas.pydata.org/pandas-docs/stable/advanced.html#sorting-a-multiindex
# ###
# ### Note: this is true for slicing in general (not just MultiIndex): indices need to be sorted
# ### either via sort_index or sortlevel(axis=1)
# ###
# ### Se also: http://www.somebits.com/~nelson/pandas-multiindex-slice-demo.html
# ### for why not sorting gives you errors like:
# 'MultiIndex Slicing requires the index to be fully lexsorted tuple len (2), lexsort depth (1)'

# In[ ]:

idx = pd.IndexSlice
d_new.columns

# The line below will give an error because the MultiIndex is not sorted
#d_new.loc[ :, idx[:, 'New Pct']]

# So sort the MultiIndex first and it will work
d_new.sortlevel(axis=1).columns
d_new.sortlevel(axis=1).loc[ :, idx[:, 'New Pct']]

# Sort it once and for all
d_new.sortlevel(axis=1, inplace=True)
d_new.loc[ :, idx[:, 'New Pct']]


# # Just for fun: let's turn dd_new into dd

# In[548]:

dd_new
dd_new_unstack = dd_new.unstack()
dd_new_unstack
dd_new_unstack2  = dd_new_unstack.swaplevel(axis=1, i=1, j=0)
dd_new_unstack2


# ### Fix the MultiIndex by sorting, as we did for the first data representation

# In[ ]:

dd_new_unstack2.sortlevel(axis=1, inplace=True)


# In[ ]:

d_new.index
dd_new_unstack2.index
d_new.index == dd_new_unstack2.index

d_new.columns
dd_new_unstack2.columns
d_new.columns == dd_new_unstack2.columns


# In[ ]:

d_new == dd_new_unstack2

