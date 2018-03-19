
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
get_ipython().magic('aimport trans.pca')

from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().magic('matplotlib inline')


# In[3]:

residuals_file = "sector_residuals.pkl"
residuals_file = "sector_residuals_{}.pkl".format(dup.parse("02/28/2018").strftime("%Y%m%d"))
residuals_file


# In[4]:

sector_residuals = gd.load_data(residuals_file)


# In[5]:

scale_pl = make_pipeline( SklearnPreproccessingTransformer( StandardScaler() ) )

scaled_df = scale_pl.fit_transform(sector_residuals)
    
scaled_df.head()


# In[6]:

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(scaled_df)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf.head(5)
pca.__dict__
pca.components_.shape


# In[7]:


sectors =  { 
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial": "XLF",
    "Health": "XLV",
    "Industrial": "XLI", 
    "Materials" : "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK", 
    "Telecom": "XTL",
    "Utilities": "XLU"
}

to_label = {}

for key, val in sectors.items():
    to_label[val] = key
    
to_label


# In[8]:

tickers = scaled_df.columns.get_level_values(1).tolist()
tickers = [ to_label[t] for t in tickers]


# In[9]:

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Ticker', fontsize = 15)
ax.set_ylabel('Weight', fontsize = 15)

width = 1
ind = np.arange( len(tickers))
bars_pc0 = ax.bar( 4*ind, pca.components_[0], color="r")
bars_pc1 = ax.bar( 4*ind + width, pca.components_[1], color="b")

ax.set_xticks(4*ind + width / 2)
ax.set_xticklabels(tickers)
plt.xticks(rotation=60)
ax.legend( (bars_pc0, bars_pc1), [ "PC {:d}".format(i) for i in np.arange(2)])


# In[10]:

sector_residuals.head()


# In[11]:

get_ipython().magic('aimport trans.pca')
import trans.pca as pct

pco = pct.PrincipalComp(sector_residuals, )

pca_df = pco.singlePCA()


# In[12]:

pca_df.head()
pca_df.index


# In[13]:

s_df = pca_df.loc[:, idx[ ["PC 0", "PC 1"],:]].stack()
s_df
s_df.plot.bar()


# In[ ]:



