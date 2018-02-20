
# coding: utf-8

# In[136]:

# %load /tmp/t.py


# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

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


# In[3]:

get_ipython().magic('aimport trans.data')
ai = gd.combine_data(['FB', 'AAPL', 'AMZN', 
                           'NFLX', 'GOOG', 'SPY'])
ai.head()


# In[4]:

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


# In[5]:

from trans.reg import Reg

# Reduce to have only "Pct" attrs since all non-dep columns will become independents
get_ipython().magic('aimport trans.reg')

# pipe_pct_only = make_pipeline(GenSelectAttrsTransformer(['Pct'] )
pipe_pct_only   = make_pipeline(GenSelectAttrsTransformer(['Pct'], dropSingle=False ) )    

fup = pipe_pct_only.fit_transform(fud2)
r = Reg(fup)


# In[7]:

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

# In[8]:

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

# In[9]:

reg_input_pl = make_pipeline(  pctOnlyTrans,
                         RestrictToCalendarColTransformer( ("Pct", "SPY")),
                         RestrictToNonNullTransformer("all"),
                         # FillNullTransformer(method="bfill")
                       )

so1 = reg_input_pl.fit_transform(data_df)
so1.head()


# ## One big pipeline for both

# In[10]:

pipe_nn = make_pipeline( featUn,
                         DatetimeIndexTransformer("Dt"),
                         pctOnlyTrans,
                         RestrictToCalendarColTransformer( ("Pct", "SPY")),
                         RestrictToNonNullTransformer("all"),
                         # FillNullTransformer(method="bfill")
                       )
so2 = pipe_nn.fit_transform(ai)
so2.head()


# In[19]:


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


# In[12]:

beta_df.shape


# ## Remember that the index for betas is at a lower frequency than index for data

# In[13]:

data_df.index
beta_df.index


# In[20]:


concatTrans = DataFrameConcat( [ data_df, beta_df ])
ret_and_beta_df = concatTrans.fit_transform(pd.DataFrame())
ret_and_beta_df.tail()
ret_and_beta_df.shape


# In[21]:

gd.save_data(data_df, "ret_df.pkl")
gd.save_data(beta_df, "beta_df.pkl")
gd.save_data(ret_and_beta_df, "ret_and_beta_df.pkl")


# In[1]:

ret_df = gd.load_data("ret_df.pkl")
beta_df = gd.load_data("beta_df.pkl")


# In[36]:

from trans.gtrans import *

concatTrans = DataFrameConcat( [ beta_df ])
r_and_b_df = concatTrans.fit_transform(ret_df)
r_and_b_df.tail()
r_and_b_df.shape

concatTrans = DataFrameConcat( [ ret_df, beta_df ])
r_and_b_df = concatTrans.fit_transform(pd.DataFrame())
r_and_b_df.tail( )
r_and_b_df.shape


# In[23]:

ret_and_beta_df = gd.load_data("ret_and_beta_df.pkl")


# In[25]:

cols = beta_df.columns.get_level_values(0).unique().tolist()
betaCols = [ c for c in cols if re.search('^Beta', c) ]

betaCols


# In[26]:

get_ipython().magic('aimport trans.gtrans')
beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaCols),
                            FillNullTransformer(method="ffill"),
                            GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
beta_rolled_df = beta_r_pl.fit_transform(ret_and_beta_df)
beta_rolled_df.tail()


# In[27]:

ret_and_beta_df.shape
beta_rolled_df.shape

ret_and_rolled_beta_pl = DataFrameConcat( [ ret_and_beta_df, beta_rolled_df])
ret_and_rolled_beta_df = ret_and_rolled_beta_pl.fit_transform( pd.DataFrame() )
ret_and_rolled_beta_df.tail()


# In[60]:

ret_and_rolled_beta_df.columns


# ## Really need to select Dep vars, ind vars, not just Pct

# In[29]:

ret_df = GenSelectAttrsTransformer(['Pct']).fit_transform(ret_and_rolled_beta_df)
ret_df.tail()
cols = ret_and_rolled_beta_df.columns.get_level_values(0).unique().tolist()

betaFwdCols = [ c for c in cols if re.search('^Beta \d+ rolled fwd$', c) ]


rolled_beta_df = GenSelectAttrsTransformer(betaFwdCols).fit_transform(ret_and_rolled_beta_df)
rolled_beta_df.tail()



# In[9]:

ret_and_rolled_beta_df.columns


# In[8]:

# Add a constant factor col
ret_and_rolled_beta_df[idx["Pct", "One"]] = 1


# In[27]:

betaFwdAttrs
pr = zip(betaFwdAttrs, [ ("Pct", "One"), ("Pct", "SPY")])
for (a, c) in pr:
    print("Attr {}, col {}".format(a,c))


# In[31]:

depAttr = "Pct"
depTickers = ret_and_rolled_beta_df.loc[:, idx[betaFwdAttrs[0]] ].head().columns.tolist()
depTickers
depCols = [ ("Pct", t) for t in depTickers]
depCols

ret_and_rolled_beta_df.loc[:, idx["Pct", depTickers]].tail()
ret_and_rolled_beta_df.loc[:, depCols].tail()


# In[137]:

get_ipython().magic('aimport trans.reg')
import trans.reg 
sensAttrs = Reg.sensAttrs(ret_and_rolled_beta_df, '^Beta \d+ rolled fwd$')
sensAttrs


# In[138]:

depTickers = Reg.depTickersFromSensAttrs(ret_and_rolled_beta_df, sensAttrs )
depTickers
depCols = [ ("Pct", t) for t in depTickers ]
depCols


# In[140]:

get_ipython().magic('aimport trans.reg')
import trans.reg
Reg.addConst(ret_and_rolled_beta_df,("Pct", "One"), 1)
#ret_and_rolled_beta_df[("Pct", "One")] = 1


# In[141]:

indCols = [ ("Pct", "One"), ("Pct", "SPY")]
indCols


# In[142]:

(df_dep, list_of_ind_dfs, list_of_sens_dfs) = Reg.retAttribSetup(ret_and_rolled_beta_df, 
                       depCols, 
                        indCols, 
                       sensAttrs)


# In[143]:

df_dep.tail()


# In[144]:

len(list_of_ind_dfs)
for df in list_of_ind_dfs:
    print(df.tail())


# In[145]:

len(list_of_sens_dfs)
for df in list_of_sens_dfs:
    print(df.tail())

