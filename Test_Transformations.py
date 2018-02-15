
# coding: utf-8

# In[92]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[93]:

import pandas as pd
idx = pd.IndexSlice

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression

from datetime import timedelta

s = '2017-11-01'
e = '2017-12-20'

get_ipython().magic('aimport trans.data')
get_ipython().magic('aimport trans.gtrans')
get_ipython().magic('aimport trans.reg')

from trans.data import GetData as gd
from trans.gtrans import *

from trans.reg import Reg

from trans.gtrans import DataFrameFunctionTransformer 

pctTrans = DataFrameFunctionTransformer(func = lambda s: s.pct_change())
rankTrans = DataFrameFunctionTransformer(func = lambda s: s.rank(method="first"), axis=1)
pctOnlyTrans = GenSelectAttrsTransformer(['Pct'], dropSingle=False )


# In[94]:

get_ipython().magic('aimport trans.data')
ai = gd.combine_data(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
ai.head()


# In[95]:

#all_ids = pd.concat([ all, ids_all ], axis=1)
#all_ids.head()
all_ids = gd.compile_data_all(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
all_ids.head()
ai = all_ids.swaplevel(axis=1,i=0,j=1)
ai.sortlevel(axis=1, inplace=True)
ai.head()
#ai.loc[:, idx['Adj Close',:]].head()

pipe_close = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )
                      )   

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True ), 
                         pctTrans,
                      )

featUn = GenDataFrameFeatureUnion( [ ("Adj Close", pipe_close),
                                    ("Pct", pipe_pct)
                                   ] )
fud = featUn.fit_transform(ai)
fud.tail()

lr=LinearRegression()

# Can set x_vals either of the two ways below: needed is to be a column, not a row
x_vals = fud.loc[s:e, idx["Pct", "SPY"]].values.reshape(-1,1)
x_vals = np.matrix( np.asarray(fud.loc[s:e, idx["Pct", "SPY"]])).T

y_vals = fud.loc[s:e, idx["Pct", "AAPL"]] 
lr.fit( x_vals,  y_vals)
lr.coef_
lr.intercept_

from trans.gtrans import *
fud2 =  DatetimeIndexTransformer("Dt").transform(fud)
fud2.head()
fud2.index


# In[96]:

from trans.reg import Reg

# Reduce to have only "Pct" attrs since all non-dep columns will become independents
get_ipython().magic('aimport trans.reg')

# pipe_pct_only = make_pipeline(GenSelectAttrsTransformer(['Pct'] )
pipe_pct_only   = make_pipeline(GenSelectAttrsTransformer(['Pct'], dropSingle=False ) )    

fup = pipe_pct_only.fit_transform(fud2)
r = Reg(fup)


# In[97]:

rn = Reg(fup.dropna(axis=0, how="any"))
ma = rn.modelCols( [ idx["Pct", "SPY"]])
(i,d) = ma

(a1, e1) = ("01/01/2000", "04/14/2000")

res_df = rn.rollingModel( i,  d[1], #idx["Pct", "AAPL"],
                 pd.to_datetime(a1,infer_datetime_format=True),
                 pd.to_datetime(e1, infer_datetime_format=True),
                 timedelta(weeks=4)
            )
res_df


# # One big pipeline, starting from raw data

# ## One pipeline to make data

# In[98]:

pipe_close = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )
                      )   

pipe_pct   = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True ), 
                         pctTrans,
                      )

featUn = GenDataFrameFeatureUnion( [ ("Adj Close", GenSelectAttrsTransformer(['Adj Close'], dropSingle=True )),
                                    ("Pct", pipe_pct)
                                   ] )

data_pl = make_pipeline( featUn,
                         DatetimeIndexTransformer("Dt")
                       )

data_df = data_pl.fit_transform(ai)
data_df.head()


# ## Second pipeline to prepare the data for regression

# In[99]:

reg_input_pl = make_pipeline(  pctOnlyTrans,
                         RestrictToCalendarColTransformer( ("Pct", "SPY")),
                         RestrictToNonNullTransformer("all"),
                         # FillNullTransformer(method="bfill")
                       )

so1 = reg_input_pl.fit_transform(data_df)
so1.head()


# ## One big pipeline for both

# In[100]:

pipe_nn = make_pipeline( featUn,
                         DatetimeIndexTransformer("Dt"),
                         pctOnlyTrans,
                         RestrictToCalendarColTransformer( ("Pct", "SPY")),
                         RestrictToNonNullTransformer("all"),
                         # FillNullTransformer(method="bfill")
                       )
so2 = pipe_nn.fit_transform(ai)
so2.head()


# In[101]:


ra = Reg(so1)
ma = ra.modelCols( [ idx["Pct", "SPY"]])
ma

beta_df = ra.rollingModelAll( *ma, #idx["Pct", "AAPL"],
                 pd.to_datetime("01/01/2000",infer_datetime_format=True),
                 # pd.to_datetime("04/14/2000", infer_datetime_format=True),
                 pd.to_datetime("12/29/2017", infer_datetime_format=True),
                 timedelta(weeks=4)
            )
beta_df.tail()


# In[102]:

beta_df.shape


# ## Remember that the index for betas is at a lower frequency than index for data

# In[103]:

data_df.index
beta_df.index


# In[104]:


concatTrans = DataFrameConcat( [ data_df, beta_df ])
ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())
ret_and_beta_df.tail()
ret_and_beta_df.shape


# In[105]:

gd.save_data(data_df, "ret_df.pkl")
gd.save_data(beta_df, "beta_df.pkl")
gd.save_data(ret_and_beta_df, "ret_and_beta_df.pkl")


# In[106]:

ret_df = gd.load_data("ret_df.pkl")
beta_df = gd.load_data("beta_df.pkl")


# In[107]:

from trans.gtrans import *

concatTrans = DataFrameConcat( [ beta_df ])
r_and_b_df = concatTrans.fit_transform(ret_df)
r_and_b_df.tail()
r_and_b_df.shape

concatTrans = DataFrameConcat( [ ret_df, beta_df ])
r_and_b_df = concatTrans.fit_transform(pd.DataFrame())
r_and_b_df.tail( )
r_and_b_df.shape


# In[108]:

ret_and_beta_df = gd.load_data("ret_and_beta_df.pkl")


# In[109]:

cols = beta_df.columns.get_level_values(0).unique().tolist()
betaCols = [ c for c in cols if re.search('^Beta', c) ]

betaCols


# In[110]:

get_ipython().magic('aimport trans.gtrans')
beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaCols),
                            FillNullTransformer(method="ffill"),
                            GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
beta_rolled_df = beta_r_pl.fit_transform(ret_and_beta_df)
beta_rolled_df.tail()


# In[111]:

ret_and_beta_df.shape
beta_rolled_df.shape

ret_and_rolled_beta_pl = DataFrameConcat( [ ret_and_beta_df, beta_rolled_df])
ret_and_rolled_beta_df = ret_and_rolled_beta_pl.fit_transform( pd.DataFrame() )
ret_and_rolled_beta_df.tail()


# In[112]:

ret_and_rolled_beta_df.columns


# ## Really need to select Dep vars, ind vars, not just Pct

# In[113]:

reg = Reg(ret_and_rolled_beta_df)


# In[114]:

sensAttrs = reg.sensAttrs(ret_and_rolled_beta_df, '^Beta \d+ rolled fwd$')
sensAttrs


# In[115]:

depTickers = reg.depTickersFromSensAttrs(ret_and_rolled_beta_df, sensAttrs )
depTickers
depCols = [ ("Pct", t) for t in depTickers ]
depCols


# In[116]:

reg.addConst(ret_and_rolled_beta_df,("Pct", "1"), 1)


# In[117]:

indCols = [ ("Pct", "1"), ("Pct", "SPY")]
indCols


# In[118]:

reg_df =reg.retAttrib(ret_and_rolled_beta_df, 
            indCols,
            depCols, 
            sensAttrs)


# In[119]:

reg_df.tail()


# ## What follows are individual tests for regression

# In[28]:

r = Reg(fup)
ma = r.modelCols( [ idx["Pct", "SPY"]])
ma

res_df = r.rollingModel( i,  d[1], #idx["Pct", "AAPL"],
                 pd.to_datetime("01/01/2017",infer_datetime_format=True),
                 pd.to_datetime("12/15/2017", infer_datetime_format=True),
                 timedelta(weeks=4)
            )
res_df


res_all = r.rollingModelAll( *ma,
                     pd.to_datetime("01/01/2017",infer_datetime_format=True),
                     pd.to_datetime("12/15/2017", infer_datetime_format=True),
                     timedelta(weeks=4)
            )

res_all.tail()


# In[ ]:

sp500 = gd.save_sp500_tickers()


# In[77]:

all = gd.compile_data_all(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG'])


# In[78]:

all.columns


# In[ ]:

gd.save_data(all, "test_all.pkl")


# In[5]:

all = gd.load_data("test_all.pkl")


# In[6]:

ids = gd.get_data_from_yahoo2([ 'SPY'])
ids_all = gd.compile_data_all(['SPY'])
gd.save_data(ids_all, "indexes_all.pkl")


# In[6]:

ids_all = gd.load_data("indexes_all.pkl")


# In[161]:

from trans.gtrans import GenSelectAttrsTransformer
from trans.gtrans import GenRenameAttrsTransformer
from trans.gtrans import GenRankTransformer

pipe_1 = make_pipeline(GenSelectAttrsTransformer(['Adj Close'] ), 
                       pctTrans,
                       GenRenameAttrsTransformer(lambda col: col + ' pct')
                      )

pipe_2 = make_pipeline(GenSelectAttrsTransformer(['Adj Close'] ), 
                       pctTrans, 
                       # rankTrans,
                       GenRankTransformer(),
                       GenRenameAttrsTransformer(lambda col: col + ' rank')
                          )

featU = DataFrameFeatureUnion([ pipe_1, pipe_2 ])
u = featU.fit_transform(ai.head())
u.head()
                    


# In[20]:

from trans.gtrans import GenDataFrameFeatureUnion

pipe_1 = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True ), 
                       pctTrans,
                       # GenRenameAttrTransformer(lambda col: col + ' pct')
                      )
pipe_2 = make_pipeline(GenSelectAttrsTransformer(['Adj Close'], dropSingle=True), 
                       pctTrans,
                       # DataFrameFunctionTransformer(pd.Series.rank, axis=1)
                        GenRankTransformer(),
                       # GenRenameAttrTransformer(lambda col: col + ' rank')
                      )

featUn =  GenDataFrameFeatureUnion( [ ("Pct", pipe_1), ("Rank", pipe_2)])
un = featUn.fit_transform(ai['2000-01-19':'2000-01-27'])
un.tail()


# In[61]:


un.tail().loc[:, idx[:,'AAPL']]


# In[82]:

lrt = LinearRegression()

pipe_reg = make_pipeline( GenSelectAttrsTransformer(['Pct']),
                        lrt
                         # inearRegression()
                        )
pipe_reg.fit(fud.loc[s:e, idx[:,'SPY']], fud.loc[s:e, idx[:,'AAPL']])
lrt.coef_
lrt.intercept_


# ## Predictions (second col) are correct; don't know what first col is
# ## Looks approx like the intercept
# ## Strangely: lrt.coef_  and lrt.intercept_ are length 2, not length 1

# In[83]:

pipe_reg.predict(fud.loc[s:e, idx[:,'SPY']]
                )
pipe_reg.predict(fud.loc[s:e, idx["Pct",'SPY']].to_frame()
                )


# In[84]:

lr.fit( np.matrix( np.asarray( fud.loc[s:e, idx["Pct", "SPY"]])).T,
       y_vals
      )
lr.coef_
lr.intercept_


# In[85]:

fitted_y_vals = lr.predict(x_vals)

x_vals
fitted_y_vals

import matplotlib.pyplot as plt
plt.scatter( x_vals, y_vals, color="blue")
plt.plot( x_vals, fitted_y_vals, color="red")
plt.show()
# pipe_reg.fit_transform( all_ids.loc[:, idx['FB',:]].tail(), un.loc[:, ["Pct", "SPY"]].tail())
                         


# In[86]:

get_ipython().magic('aimport trans.reg')
from trans.reg import Reg

fud.shape
r = Reg(fud2[s:e])

r_df = r.get(  [ idx["Pct", "SPY"] ],  idx["Pct", "AAPL"] )
r.fit( r_df )

#%%debug 
r = Reg(fud2)

r_df  = r.get(  [ idx["Pct", "SPY"] ], idx["Pct", "AAPL"])

res_df = r.rollingFit( r_df, 
                 pd.to_datetime("01/01/2017",infer_datetime_format=True),
                 pd.to_datetime("12/15/2017", infer_datetime_format=True),
                 timedelta(weeks=4)
            )

res_df

r = Reg(fud2)

res_df = r.rollingModel( [ idx["Pct", "SPY"] ],  idx["Pct", "AAPL"],
                 pd.to_datetime("01/01/2017",infer_datetime_format=True),
                 pd.to_datetime("12/15/2017", infer_datetime_format=True),
                 timedelta(weeks=4)
            )
res_df

