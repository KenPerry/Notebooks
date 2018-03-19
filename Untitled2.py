
# coding: utf-8

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[2]:

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().magic('matplotlib inline')


# In[30]:

dates = [ "12/01/2018", "12/29/2018" ]
row1 = { ("PC1", "T1"): 111, ("PC2", "T1"): 121, ("PC1", "T2"): 211, ("PC2","T2"): 221 }
row2 = { ("PC1", "T1"): 112, ("PC2", "T1"): 122, ("PC1", "T2"): 212, ("PC2","T2"): 222}
rows = { ("PC1", "T1"): [111, 112], ("PC2", "T1"): [121, 122], ("PC1", "T2"): [211, 212], ("PC2","T2"): [221,222] } 
tuples = [ (111, 121, 211, 221), (112, 122, 212, 222)]


# In[48]:

c=pd.MultiIndex.from_tuples([ ("PC1", "T1"), ("PC2", "T1"), ("PC1", "T2"), ("PC2", "T2")])
d = pd.DataFrame(tuples, columns=c)
d.columns
d


# In[28]:

df = pd.DataFrame(rows, index=dates)
df.index
df.columns

