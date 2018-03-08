
# coding: utf-8

# In[ ]:




# In[11]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[12]:

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
from trans.reg import Reg


pctTrans     = DataFrameFunctionTransformer(func = lambda s: s.pct_change())
rankTrans    = DataFrameFunctionTransformer(func = lambda s: s.rank(method="first"), axis=1)
pctOnlyTrans = GenSelectAttrsTransformer(['Pct'], dropSingle=False )


# ## Get the raw data

# In[13]:

get_ipython().magic('aimport trans.data')
raw_df = gd.combine_data(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
raw_df.head()


# ## Define featUn transformer: compute Pct and append to Adj Close

# In[14]:

pipe_close = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )
                      )   

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True ), 
                         pctTrans,
                      )

featUn = GenDataFrameFeatureUnion( [ ("Adj Close", pipe_close),
                                    ("Pct", pipe_pct)
                                   ] )


# In[15]:

pipe_pct_only   = make_pipeline(GenSelectAttrsTransformer(['Pct'], dropSingle=False ) ) 


# ## Create pipeline to prepare data for regression

# In[16]:

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

# In[17]:

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


# ## Append the rolling betas to the prepared data

# In[20]:

concatTrans = DataFrameConcat( [ pct_df, beta_df ])
ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())
ret_and_beta_df.loc[:"2017-12-29",:].tail()
ret_and_beta_df.shape


# In[12]:

gd.save_data(pct_df, "ret_df.pkl")
gd.save_data(beta_df, "beta_df.pkl")
gd.save_data(ret_and_beta_df, "ret_and_beta_df.pkl")


# ## Prepare for Return Attribution

# ### Find the attributes with the sensitivities

# In[23]:

rab = RegAttr(ret_and_beta_df)
rab.setSens(beta_df)
betaAttrs = rab.sensAttrs('^Beta \d+$')
betaAttrs


# ## Roll the betas forward

# In[46]:

beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaAttrs),
                            ShiftTransformer(1),
                            FillNullTransformer(method="ffill"),
                            GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
beta_rolled_df = beta_r_pl.fit_transform(ret_and_beta_df)
beta_rolled_df.tail()


# ### Append the rolled betas to the regression results

# In[48]:

ret_and_rolled_beta_pl = DataFrameConcat( [ ret_and_beta_df, beta_rolled_df])
ret_and_rolled_beta_df = ret_and_rolled_beta_pl.fit_transform( pd.DataFrame() )
ret_and_rolled_beta_df.tail()


# In[29]:

reg = Reg(ret_and_rolled_beta_df)


# ### Find the columns for: 
# #### independent variables
# #### dependent variables
# #### sensitivities

# In[49]:

indCols = [ ("Pct", "1"), ("Pct", "SPY")]
indCols

rab2 = RegAttr(ret_and_rolled_beta_df)
rab2.setSens(beta_rolled_df)
sensAttrs = rab2.sensAttrs('^Beta \d+ rolled fwd$')
sensAttrs

depTickers = rab2.depTickersFromSensAttrs(sensAttrs )
depTickers
depCols = [ ("Pct", t) for t in depTickers ]
depCols



# ## Add constant (for interecept return) column

# In[50]:

rab2.addConst(("Pct", "1"), 1)


# ### Perform the return attribution

# In[51]:

retAttr_df =rab2.retAttrib(
            indCols,
            depCols, 
            sensAttrs)


# In[52]:

retAttr_df.loc[:"2018-02-07",:].tail()


# In[37]:

gd.save_data(retAttr_df, "retattr_df.pkl")

