
# coding: utf-8

# In[161]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[162]:

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


# In[163]:

ret_df = gd.load_data("ret_df.pkl")
beta_df = gd.load_data("beta_df.pkl")
ret_and_beta_df = gd.load_data("ret_and_beta_df.pkl")


# In[164]:

cols = beta_df.columns.get_level_values(0).unique().tolist()
betaCols = [ c for c in cols if re.search('^Beta', c) ]

betaCols


# In[165]:

beta_r_pl = make_pipeline( GenSelectAttrsTransformer(betaCols),
                            FillNullTransformer(method="ffill"),
                            GenRenameAttrsTransformer(lambda col: col + ' rolled fwd', level=0)
                         )
beta_rolled_df = beta_r_pl.fit_transform(ret_and_beta_df)
beta_rolled_df.tail()


# In[166]:

ret_and_beta_df.shape
beta_rolled_df.shape

ret_and_rolled_beta_pl = DataFrameConcat( [ ret_and_beta_df, beta_rolled_df])
ret_and_rolled_beta_df = ret_and_rolled_beta_pl.fit_transform( pd.DataFrame() )
ret_and_rolled_beta_df.tail()


# In[167]:

reg = Reg(ret_and_rolled_beta_df)
reg.addConst(ret_and_rolled_beta_df,("Pct", "1"), 1)


# In[168]:

gd.save_data(ret_and_rolled_beta_df, "ret_and_rolled_beta_df.pkl")


# In[169]:

sensAttrs = reg.sensAttrs(ret_and_rolled_beta_df, '^Beta \d+ rolled fwd$')
sensAttrs

depTickers = reg.depTickersFromSensAttrs(ret_and_rolled_beta_df, sensAttrs )
depTickers
depCols = [ ("Pct", t) for t in depTickers ]
depCols

indCols = [ ("Pct", "1"), ("Pct", "SPY")]
indCols


# In[170]:

(df_dep, list_of_ind_dfs, list_of_sens_dfs) = reg.retAttrib_setup(ret_and_rolled_beta_df, 
                                                                 indCols,
                                                                 depCols, 
                                                                 sensAttrs)


# In[171]:

get_ipython().magic('aimport trans.reg')
from trans.reg import *
r = reg.retAttrib_to_np(list_of_ind_dfs, df_dep, list_of_sens_dfs)
(indMat, depMat, sensMat) = r


# In[172]:

depMat.shape
indMat.shape
sensMat.shape


# In[173]:

get_ipython().magic('aimport trans.reg')
from trans.reg import *
(contribsMat, predMat, errMat) = reg.retAttrib_np(indMat, depMat, sensMat)


# In[174]:

contribsMat.shape
predMat.shape
errMat.shape


# In[175]:

(list_of_contribs_dfs, predict_df, err_df) = reg.retAttrib_to_df(contribsMat, 
                                                                 predMat, 
                                                                 errMat,
                                                                 ret_and_rolled_beta_df.index,
                                                                 sensAttrs,
                                                                 depTickers
                                                               )


# In[176]:

len(list_of_contribs_dfs)
list_of_contribs_dfs[0].shape
predict_df.shape
err_df.shape


# In[177]:

ret_and_rolled_beta_df.loc[:, idx["Pct",:]].tail()
list_of_contribs_dfs[0].tail()
predict_df.tail()
err_df.tail()


# In[178]:

indTickers = [ col[1] for col in indCols ]
contrib_k = [ "Contrib from " + indTickers[i] for i in range( len(list_of_contribs_dfs ))]


# contrib_k + [ pred_k ] + [ err_k ]
regResTrans = DataFrameConcat( list_of_contribs_dfs + [ predict_df ] + [ err_df ],
                               df_keys= contrib_k + [ "Predicted" ]  + ["Error"]
                             )
regRes_df = regResTrans.fit_transform(pd.DataFrame())


# In[179]:

regRes_df.shape
regRes_df.columns.get_level_values(0).unique()
regRes_df.columns.get_level_values(1).unique()
regRes_df.tail()


# In[180]:

from trans.reg import *
reg_df =reg.retAttrib(ret_and_rolled_beta_df, 
            indCols,
            depCols, 
            sensAttrs)

reg_df.tail()

