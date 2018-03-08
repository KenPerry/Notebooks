
# coding: utf-8

# In[ ]:




# In[194]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[195]:

import pandas as pd
idx = pd.IndexSlice

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression

from datetime import timedelta

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.reg')

from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr

pctOnlyTrans = GenSelectAttrsTransformer(['Pct'], dropSingle=False )


# ## Get the raw data

# In[196]:

get_ipython().magic('aimport trans.data')
raw_df = gd.combine_data(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
raw_df.head()


# ## Define featUn transformer: compute Pct and append to Adj Close

# In[230]:

raw_df.to_csv("/tmp/raw_df.csv")


# In[197]:

pipe_close = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )
                      )   

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True ), 
                         pctTrans,
                      )

featUn = GenDataFrameFeatureUnion( [ ("Adj Close", pipe_close),
                                    ("Pct", pipe_pct)
                                   ] )


# In[198]:

pipe_pct_only   = make_pipeline(GenSelectAttrsTransformer(['Pct'], dropSingle=False ) ) 


# ## Create pipeline to prepare data for regression

# In[199]:

pipe_nn = make_pipeline( featUn,
                         DatetimeIndexTransformer("Dt"),
                         pctOnlyTrans,
                         RestrictToCalendarColTransformer( ("Pct", "SPY")),
                         RestrictToNonNullTransformer("all"),
                         # FillNullTransformer(method="bfill")
                       )
pct_df = pipe_nn.fit_transform(raw_df)
pct_df.head()


# ## Do a rolling regression on the dataframe with prepared data

# In[231]:

pct_df.to_csv("/tmp/pct_df.csv")


# In[200]:

ra = Reg(pct_df)
ma = ra.modelCols( [ idx["Pct", "SPY"]])
ma

beta_df = ra.rollingModelAll( *ma, #idx["Pct", "AAPL"],
                 pd.to_datetime("01/01/2000",infer_datetime_format=True),
                 # pd.to_datetime("04/14/2000", infer_datetime_format=True),
                 pd.to_datetime("12/29/2017", infer_datetime_format=True),
                 timedelta(weeks=4)
            )
beta_df.tail()


# ## By concatenating the pct_df and beta_df
# ### we wind up with CONSITENT indexing for returns and sensitivities
# ### BUT these are the UNION of the dates, not the intersecton

# In[232]:

beta_df.loc["2017-11-03":,:].to_csv("/tmp/beta_df.csv")


# In[207]:

concatTrans = DataFrameConcat( [ pct_df, beta_df ])
ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())
ret_and_beta_df.loc[:"2017-12-29",:].tail()
ret_and_beta_df.shape


# ## Append the rolling betas to the prepared data

# In[233]:

ret_and_beta_df.loc["2017-11-03":,:].to_csv("/tmp/ret_and_beta_df.csv")


# In[208]:

get_ipython().magic('aimport trans.regpipe')
from trans.regpipe import RegPipe


# In[209]:

start = pd.to_datetime("01/01/2000",infer_datetime_format=True)
end   = pd.to_datetime("12/29/2017", infer_datetime_format=True)
window = 0
step   = timedelta(weeks=4)


# In[210]:

rp = RegPipe( pct_df )
rp.indCol( idx["Pct", "SPY"] )
rp.regress( start, end, window, step)


# In[204]:

rp.beta_df.tail()


# In[211]:

rp.beta_df.shape


# ## NEED to decide how to conform the index of pct_df and beta_df
# ### For backwards compatiblity testing, we will use the UNION and force them
# ### back into the object.  NEED a better idea
# 
# ## NOTES:
# -  pct_df includes ALL data; beta_df is limited to date range of regression
#     - beta_df ends at end of 2017; pct_df goes into 2018 and continually grows
# 

# In[212]:

concatTrans = DataFrameConcat( [ pct_df, rp.beta_df ])
rp_ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())
rp_ret_and_beta_df.loc[:"2017-12-29",:].tail()
rp_ret_and_beta_df.shape


# In[237]:

rp_ret_and_beta_df.loc["2015-05-05":,:].to_csv("/tmp/rp_ret_and_beta_df.csv")


# In[213]:

rp_pct_df = rp_ret_and_beta_df.loc[:, idx["Pct",:]]
rp_pct_df.shape


# In[214]:

sensAttrs = beta_df.columns.get_level_values(0).unique().tolist()
sensAttrs


# In[215]:

rp_beta_df = rp_ret_and_beta_df.loc[:, idx[sensAttrs,:] ]
rp_beta_df.shape


# In[223]:

rp.reg.data = rp_pct_df
rp.regAttr.data = rp_pct_df
rp.beta_df  = rp_beta_df


# In[115]:

# rp.indexUnionBeta(pct_df.index)


# In[224]:

rp.rollBeta(1, "ffill")
rp.beta_rolled_df.loc[:"2017-12-29",:].tail()


# In[225]:


beta_rolled_df = rp.beta_rolled_df

sensAttrs = beta_rolled_df.columns.get_level_values(0).unique().tolist()
sensAttrs

beta_rolled_df.columns.get_level_values(1).unique().tolist()

regBeta = RegAttr(pct_df)
regBeta.setSens(beta_rolled_df)

# Problem: The constant is there in beta_rolled_df
depTickers = regBeta.depTickersFromSensAttrs(sensAttrs )
depTickers


# ## NEED to decide how to conform index of pct_df and beta_df:
# ### UNION: gives 4569
# ### Re-indexing: gives 4561
# 
# ### For backward compatibility testing, will force into the larger
# 
# ### n.b., pct_df includes 8 more data points b/c the datafiles for the tickers have been upated from 2/8 to 2/20, not for any other reason
# 

# In[226]:

pct_df.shape
rp.reg.data.shape
ret_and_beta_df.shape

rp.regAttr.data.shape


# In[229]:

rp.attrib()
rp.retAttr_df.loc[:"2018-02-07",:].tail()


# In[ ]:

gd.save_data(pct_df, "ret_df.pkl")
gd.save_data(beta_df, "beta_df.pkl")
gd.save_data(ret_and_beta_df, "ret_and_beta_df.pkl")


# ## Prepare for Return Attribution

# ### Find the attributes with the sensitivities

# In[157]:

rab = Reg(pct_df)
rab.setSens(beta_df)
betaAttrs = rab.sensAttrs('^Beta \d+$')
betaAttrs


# ## Roll the betas forward

# In[158]:

beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaAttrs),
                            ShiftTransformer(1),
                            FillNullTransformer(method="ffill"),
                            GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
beta_rolled_df = beta_r_pl.fit_transform(ret_and_beta_df)
beta_rolled_df.tail()


# ### Append the rolled betas to the regression results

# In[159]:

ret_and_rolled_beta_pl = DataFrameConcat( [ ret_and_beta_df, beta_rolled_df])
ret_and_rolled_beta_df = ret_and_rolled_beta_pl.fit_transform( pd.DataFrame() )
ret_and_rolled_beta_df.tail()


# In[238]:

ret_and_rolled_beta_df.loc["2017-05-05":,:].to_csv("/tmp/ret_and_rolled_beta_df.csv")


# In[160]:

reg = Reg(ret_and_beta_df)
reg.setSens(beta_rolled_df)


# ### Find the columns for: 
# #### independent variables
# #### dependent variables
# #### sensitivities

# In[161]:

indCols = [ ("Pct", "1"), ("Pct", "SPY")]
indCols

sensAttrs = reg.sensAttrs('^Beta \d+ rolled fwd$')
sensAttrs

depTickers = reg.depTickersFromSensAttrs(sensAttrs )
depTickers
depCols = [ ("Pct", t) for t in depTickers ]
depCols



# ## Add constant (for interecept return) column

# In[165]:

reg.addConst(("Pct", "1"), 1)


# ### Perform the return attribution

# In[166]:

retAttr_df =reg.retAttrib(
            indCols,
            depCols, 
            sensAttrs)


# In[167]:

retAttr_df.tail()


# In[240]:

retAttr_df.loc["2017-05-05":,:].to_csv("/tmp/retAttr.csv")


# In[ ]:

gd.save_data(retAttr_df, "retattr_df.pkl")

