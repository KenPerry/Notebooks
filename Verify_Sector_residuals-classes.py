
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


# ## Verify prices (GetDataTransformer)

# In[3]:

def verify_df(df, v_df, cols=None, debug=False, **params):
    (min_d, max_d) = (v_df.index.min(), v_df.index.max())
    if debug:
        print("Verified df ({}, {}), shape {}".format(min_d, max_d, v_df.shape))
        print(df.columns)
        print(v_df.columns)
    
    # Output the verified df to a csv for hand-verification
    v_df.to_csv("/tmp/verify.csv")
    
    if (not cols == None):
        return df.loc[ min_d:max_d, cols].equals( v_df.loc[:, cols])
    else:
        return df.loc[ min_d:max_d, v_df.columns].equals( v_df.loc[:,:])

    
def verify_file(df, verified_df_file, cols=None, debug=False,**params):
    """
    Compare DataFrame to one that is stored in a file
    
    Parameters:
    --------------
    df: DataFrame
    verified_df_file: string. Name of pkl file containing verified DataFrame
    
    Returns
    --------
    Boolean
    """                   
    v_df = gd.load_data(verified_df_file)
    return verify_df(df, v_df)
   


# In[46]:

regParams = gd.load_data("verify_regParams.pkl")
(start, end, step, window) = list( map( lambda c: regParams[c], [ "start", "end", "step", "window" ]) )

start, end


# In[4]:

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


# In[20]:

df = gd.load_data("verify_sectors_raw_df.pkl")


# In[44]:

start, end


# In[34]:

price_df.shape
p = price_df.loc[ start:end, idx["Adj Close", :]]
p.shape
p.columns

d = df.loc[ start:end, idx["Adj Close",:]]
d.shape
df.index.min(), df.index.max()
df.columns

verify_df( p.loc[s:], d)


# In[43]:

p.index.min(), d.index.min()
s = "2012-01-01"
p.loc[s: ].equals(d.loc[s:])


# In[5]:

verify_file( price_df, "verify_sectors_raw_df.pkl")


# ## Verify returns (pctTrans)

# In[6]:

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=False),
                           pctTrans,
                           GenRenameAttrsTransformer(lambda col: "Pct", level=0)
                          )
pct_df = pipe_pct.fit_transform(price_df)


# In[7]:

verify_file(pct_df, "verify_sectors_pct_df.pkl")


# ## Verify single regression

# In[8]:




# In[9]:

regStarts = end - window + timedelta(days=1)
regStarts, end
pct_dfs = pct_df.loc[ regStarts:end,:]

rps = RegPipe( pct_dfs )
rps.indCols( [ idx["Pct", "SPY"] ] )
rps.regressSingle()

rps.beta_df.shape


# In[10]:

verify_file( rps.beta_df, "verify_beta_df.pkl")


# ## Continuation: Verify residuals of single regression

# In[11]:

rollAmount = 0
fillMethod = "bfill"

rps.attrib_setup(pct_dfs, rps.beta_df, rollAmount, fillMethod)
rps.attrib()

rps.retAttr_df.shape

sector_residuals = rps.retAttr_df.loc[:, idx["Error",:]]


# In[12]:

verify_file( sector_residuals, "sector_residuals.pkl")


# ## Verify stacked residual

# In[13]:

from trans.stacked.residual import Residual

resStart = dup.parse("01/01/2016")
rstack = Residual(debug=True)
rstack.init(df=pct_df, start=resStart, end=end, window=window, step=step)
resid_stack = rstack.repeated()
rstack.done()


# In[14]:

v_stack = gd.load_data("verify_resid_stack.pkl")


# ### Verify single regression matches first element of stack

# In[15]:

(v_label, v_df) = v_stack[0]
verify_df(sector_residuals, v_df)


# ## Verify first element of stack

# In[16]:

(label, df) = resid_stack[0]
verify_df(df, v_df)


# ## Verify second element of stack

# In[17]:

(v_label, v_df) = v_stack[1]
(label, df) = resid_stack[1]
verify_df(df, v_df)


# ### Manually carry out second single regression so can compare beta0, beta1, by hand with spreadsheet

# In[18]:

end2 = end - step
regStarts2 = end2 - window + timedelta(days=1)
regStarts2, end2


# In[19]:


pct_dfs2 = pct_df.loc[ regStarts2:end2,:]

rps2 = RegPipe( pct_dfs2 )
rps2.indCols( [ idx["Pct", "SPY"] ] )
rps2.regressSingle()

