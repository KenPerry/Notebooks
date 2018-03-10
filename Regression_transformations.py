
# coding: utf-8

# In[3]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[4]:

import pandas as pd
idx = pd.IndexSlice

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression

from datetime import timedelta

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.reg')
get_ipython().magic('aimport trans.regpipe')

from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

pctOnlyTrans = GenSelectAttrsTransformer(['Pct'], dropSingle=False )


# ## Get the raw data

# In[5]:

get_ipython().magic('aimport trans.data')
raw_df = gd.combine_data(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
raw_df.head()


# ## Define featUn transformer: compute Pct and append to Adj Close

# In[4]:

raw_df.to_csv("/tmp/raw_df.csv")


# In[6]:

pipe_close = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )
                      )   

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True ), 
                         pctTrans,
                      )

featUn = GenDataFrameFeatureUnion( [ ("Adj Close", pipe_close),
                                    ("Pct", pipe_pct)
                                   ] )


# In[7]:

pipe_pct_only   = make_pipeline(GenSelectAttrsTransformer(['Pct'], dropSingle=False ) ) 


# ## Create pipeline to prepare data for regression
# ### NOTE: should really RestrictToCalendar BEFORE doing pctTrans

# In[8]:

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

# In[8]:

pct_df.to_csv("/tmp/pct_df.csv")


# ## Date arithmetic:
# ### Keep in mind:
# - timedelta (for days)
# - monthdelta (affects months, use last day of month if non-calendar day)
# - dateutils.relativedelta

# ## Test  basic Reg object

# In[9]:

import dateutil.parser as dup
import dateutil.relativedelta as rd

regWindow = rd.relativedelta(months=+6)
regStep   = rd.relativedelta(weeks=+4)

regStart = dup.parse("01/01/2000")
regEnd   = dup.parse("12/29/2017")


# In[10]:

ra = Reg(pct_df)
ma = ra.modelCols( [ idx["Pct", "SPY"]])
ma

beta_df = ra.rollingModelAll( *ma, #idx["Pct", "AAPL"],
                 regStart,
                 # pd.to_datetime("04/14/2000", infer_datetime_format=True),
                 regEnd,
                 #rd.relativedelta(weeks=+4)
                 regWindow,
                 stepTimeDelta=regStep
                 # timedelta(weeks=4)
                             
            )
beta_df.tail()


# ## By concatenating the pct_df and beta_df
# ### we wind up with CONSISTENT indexing for returns and sensitivities
# ### BUT these are the UNION of the dates, not the intersecton
# 
# ## Probably SHOULD do the concatention:
# - the dates of the sensitivities don't HAVE to contain the dates for which we want to do attribution
#  - e.g., we may want to project forward from last regression

# In[38]:

beta_df.loc["2017-11-03":,:].to_csv("/tmp/beta_df.csv")


# In[11]:

concatTrans = DataFrameConcat( [ pct_df, beta_df ])
ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())
ret_and_beta_df.loc[:"2017-12-29",:].tail()
ret_and_beta_df.shape


# ## Append the rolling betas to the prepared data

# In[40]:

ret_and_beta_df.loc["2017-11-03":,:].to_csv("/tmp/ret_and_beta_df.csv")


# ## Test the higher level RegPipe object

# In[12]:

rp = RegPipe( pct_df )
rp.indCols( [ idx["Pct", "SPY"] ] )
rp.regress( regStart, regEnd, regWindow, regStep)


# In[13]:

rp.beta_df.tail()


# In[14]:

rp.beta_df.shape


# ## Set the internal data of RegPipe to sensitivities and returns

# In[16]:

rollAmount = 1
fillMethod = "ffill"

rp.attrib_setup(pct_df, rp.beta_df, rollAmount, fillMethod)


# ## Perform the attribution using the RegPipe object

# In[17]:

rp.attrib()
rp.retAttr_df.loc[:"2018-02-07",:].tail()


# ## Perform the attribution using low-level RegAttr object

# ## Prepare for Return Attribution

# ### Find the attributes with the sensitivities

# In[74]:

rab = RegAttr(pct_df)
rab.setSens(beta_df)
betaAttrs = rab.sensAttrs('^Beta \d+$')
betaAttrs


# ## Roll the betas forward

# ## Concatenate pct_df (high frequency) and beta_df (low frequency)
# ## so that they have same index before rolling the low frequency beta

# In[75]:

concatTrans = DataFrameConcat( [ pct_df, beta_df ])
ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())

ret_and_beta_df.loc[:"2017-12-29",:].tail()
ret_and_beta_df.shape



# In[76]:

beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaAttrs),
                            ShiftTransformer(1),
                            FillNullTransformer(method="ffill"),
                            GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
beta_rolled_df = beta_r_pl.fit_transform(ret_and_beta_df)
beta_rolled_df.tail()


# ### Append the rolled betas to the regression results

# In[77]:

rat = RegAttr(ret_and_beta_df)
rat.setSens(beta_rolled_df)


# ### Find the columns for: 
# #### independent variables
# #### dependent variables
# #### sensitivities

# In[78]:

indCols = [ ("Pct", "1"), ("Pct", "SPY")]
indCols

sensAttrs = rat.sensAttrs('^Beta \d+ rolled fwd$')
sensAttrs

depTickers = rat.depTickersFromSensAttrs(sensAttrs )
depTickers
depCols = [ ("Pct", t) for t in depTickers ]
depCols



# ## Add constant (for interecept return) column

# In[79]:

rat.addConst(("Pct", "1"), 1)


# ### Perform the return attribution

# In[80]:

retAttr_df =rat.retAttrib(
            indCols,
            depCols)


# In[82]:

retAttr_df.loc[:"2018-02-07",:].tail()


# In[24]:

retAttr_df.loc["2017-05-05":,:].to_csv("/tmp/retAttr.csv")


# In[ ]:

gd.save_data(retAttr_df, "retattr_df.pkl")

