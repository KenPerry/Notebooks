
# coding: utf-8

# From: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb
# 
# Article:
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# In[ ]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Reload all modules imported with %aimport
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')


# In[1]:

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().magic('matplotlib inline')


# In[3]:

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['sepal length','sepal width','petal length','petal width','target'])

df.head()


# In[4]:

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values

y = df.loc[:,['target']].values

x = StandardScaler().fit_transform(x)

pd.DataFrame(data = x, columns = features).head()


# In[5]:

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf.head(5)


# ## Examine attributes of the pca object

# In[28]:

pca.__dict__


# ## If we had retained ALL components, we could calculate
# ## pca.explained_variance_ratio_ ourselves

# In[31]:

pca_full = PCA(n_components=4)
pc_full = pca_full.fit(x)


# In[34]:

pca.explained_variance_/np.sum(pca_full.explained_variance_) - pca.explained_variance_ratio_


# In[35]:

pca_full.components_


# ## Original data matrix X shape (num_samples, num_features)
# $X = U * S * V^T$
# ### U shape is (num_samples,num_features)
# ### S shape  is (num_features, num_components), num_components <= num_features
# ### V^T shape is (num_components, num_features)
# 
# ### So
# $X = (U*S) * V^T$
# 
# ### Multiply both sides by V:
# $ X * V = (U*S) = X_{new} $
# 
# $X_{new}$ = original data X, expressed in new coordinate space
# $X_{new}$ sometimes call "principal components" but they are NOT unit length
# 
# ## Believe $U$ is unit length because S has the eigenvalues ?
# 
# 
# ## Because:
# $X = (U*S) * V^T$
# ## So if (U*S) are the PC's, 
# $V^T$ are the betas of X w.r.t. PC's
# ## and
# $ X * V = (U*S) $
# ## so the PC's (U*S) are the weighted sum of $X$, with row $i$ of $V$ defining how $PC_i$ is obtained by weighting of $X$

# ## pca.components_ == $V^T$

# In[17]:

pca.components_.shape


# ## $x_{new}$ == PC's##

# In[19]:

x_new = pca.transform(x)
x_new.shape


# ## Show $X * V == X_{new}$
# ### because $X = (U*S) * V^T$, multiply both sides by $V$

# In[25]:

np.matmul(x, np.transpose(pca.components_)) - x_new


# ## Show that the columns of principalDf are orthogonal
# ### But they are NOT magnitude 1, their magnitude is S (where X = U * S * V )

# In[44]:

pc1, pc2 = principalDf.iloc[:,0], principalDf.iloc[:,1]


# In[46]:

def mag(vec):
    return np.sum(vec * vec)** 0.5

print("Mag PC1: {}, Mag PC2: {}".format(mag(pc1), mag(pc2)))


# In[81]:

from scipy import linalg
from sklearn.utils.extmath import svd_flip
U, S, V = linalg.svd(x, full_matrices=False)

# Flip eigenvectors signs to enforce deterministic output
U, V = svd_flip(U,V)
S


# In[54]:

print("Shapes: U={}, S ={}, VT={}, U*S={}".format(U.shape, S.shape, V.shape, (U*S).shape))


# In[57]:

x_new[:5]


# ## Show x_new ( = U*S)/S = U
# ### So we can obtain magnitude 1 U, rather than just U*S
# 
# ## Question: to compute sensitivities (of PC wrt X and vice-versa)): do we need to divide by S ?

# In[92]:

x_new/S[:x_new.shape[1]] - U[:,:x_new.shape[1]]


# ## Even though not unit magnitude, pc1 and pc2 are orthogonal

# In[9]:

np.dot(pc1,pc2)


# In[13]:

df[['target']].head()
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)


# ## Plot the data in the new coordinate system defined by PC1 and PC2
# - the pc1, and pc2 ndarrays defined above are the coordinates in the new system
# 

# In[16]:

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

