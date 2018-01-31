
# coding: utf-8

# # From
# https://www.kaggle.com/jankoch/scikit-learn-pipelines-and-pandas/notebook

# In[1]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Typically, when you want to use the standard pandas/sklearn framework to tackle a machine learning or data analysis problem, you will start analysing the dataset using pandas. Once you've gotten some insights, you may derive a transformed data set using pandas and derive a model using a scikit-learn estimator like Linear or Ridge Regression. Typically, you will estimate the performance of such a model using cross validation.
# 
# There are several things I don't like with this approach and want to raise awareness of:
# 1. Typically, notebooks contain a lot of repetitive code
# 2. The pandas and scikit-learn frameworks are somehow separated and a combined pipeline is not used
# 3. Most importantly: As it is explained [here]http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics) the preprocessing step should be tested on a hold out set. This implies, that cross-validation on a dataset requires to perform preprocessing on the respective subsets constructed during cross-validation.
#     
# To solve these issues, we only have to find the answer to the following question: Is there a way to perform scikit-learn and pipeline compatible preprocessing using pandas?
# 
# Luckily, all we need to do is to adhere to the scikit-learn transformers api. A transformer typically contains a transform and a fit method. Using the scikits [TransformerMixin-Class](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin) a fit_transform function is constructed.
# 
# Let us briefly describe how transformers are used:
# 
# If the transformer needs to remember the state of the training data, e.g. the mean of a column, the fit method is used on the training data to store this state. Subsequently, the transform function is used on the train and test data. However, if not state preservation is needed, e.g. in the case of log transforming data, the fit functions may essentially do nothing, and we just use the transform function. Note that the fit function is **never** used on the test data
# 
# So the agenda of this notebooks is as follows:
# 
# 1. Load and read the data
# 2. Define transformer objects (feel free to skip this lengthy paragraph at first)
# 3. Use these transformers to preprocess the data
# 4. Use preprocessing and Ridge regression, gridsearch and crossvalidation to estimate the generalization performance
# 6. Under the assumption, that the gridsearch parameters are stable between cross validation folds, retrain a model using gridsearch on all of the training data
# 7. Further comments on additional transformers, next steps and references

# ## Loading the data

# In[2]:

import pandas as pd
import numpy as np

df_train = pd.read_csv("./pipelines_and_pandas_input/train.csv")
df_test = pd.read_csv("./pipelines_and_pandas_input/test.csv")

y_train = df_train.set_index("Id")["SalePrice"]
X_train = df_train.set_index("Id").iloc[:,:-1]
X_test = df_test.set_index("Id")


# ## Transformer Objects

# In[3]:

from sklearn.base import TransformerMixin, BaseEstimator, clone

class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: [] 
    
    """
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains selected columns of X      
        """
        trans = X[self.columns].copy() 
        return trans

    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self
    

class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    
    Parameters
    ----------
    impute : Boolean, default False
        
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise 
        an array of the form [n_elements, 1]
    
    """
    
    def __init__(self, func, impute = False):
        self.func = func
        self.impute = impute
        self.series = pd.Series() 

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame
        
        Parameters
        ----------
        X : DataFrame
            
        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X 
        """
        
        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
                
        Returns
        ----------
        self  
        """
        
        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func).copy()
        return self
    
    
class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers
    
    Fit several DataFrame transformers and provides a concatenated
    Data Frame
    
    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers
        
    """ 
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        concatted :  pandas DataFrame
        
        """
        
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], axis=1).copy()
        return concatted


    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
        
        Returns
        ----------
        self : object
        """
        
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self
    

class ToDummiesTransformer(BaseEstimator, TransformerMixin):
    """ A Dataframe transformer that provide dummy variable encoding
    """
    
    def transform(self, X, **transformparams):
        """ Returns a dummy variable encoded version of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        trans : pandas DataFrame
        
        """
    
        trans = pd.get_dummies(X).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Do nothing operation
        
        Returns
        ----------
        self : object
        """
        return self


class DropAllZeroTrainColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides dropping all-zero columns
    """

    def transform(self, X, **transformparams):
        """ Drops certain all-zero columns of X
        
        Parameters
        ----------
        X : DataFrame
        
        Returns
        ----------
        trans : DataFrame
        """
        
        trans = X.drop(self.cols_, axis=1).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Determines the all-zero columns of X
        
        Parameters
        ----------
        X : DataFrame
        y : not used
        
        Returns
        ----------
        self : object
        """
        
        self.cols_ = X.columns[(X==0).all()]
        return self


# ## Preprocessing

# In[4]:

from sklearn.pipeline import Pipeline, make_pipeline


# ### Area Columns
# We start with the columns describing some form of area. As  [Alexandre Paipu](https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models) points out, skewed columns should be log transformed. Checking some data, we find that the skewed columns are essentially the area columns. 
# 
# We simply use a regular expression to filter the area columns, then we transform to float, impute by the mean (even if no missing values appear) and perform a log (x+1) transform.

# In[5]:

area_cols = X_train.columns[X_train.columns.str.contains('(?i)area|(?i)porch|(?i)sf')].tolist()

area_cols_pipeline = make_pipeline(  
        SelectColumnsTransfomer(area_cols),
        DataFrameFunctionTransformer(func = lambda x: x.astype(np.float64)),
        DataFrameFunctionTransformer(func = np.mean, impute=True),
        DataFrameFunctionTransformer(func = np.log1p) 
    )


# ### Object Columns
# The object columns are the categorical columns. Reading the data set description, we see that NaN values are allowed for each column. Consequently, we assume that possible levels of the categorical values are known beforehand and that the NaNs are correctly encoded. So though we access the values of the test data, we don't use information not known before the analyzing the data
# 
# So what we will do here is to determine all possible levels across all categories, construct dummy variables for all variables and levels (this is not efficient!) and then drop the combinations that do not occur. Particularly, when fitting the pipeline to the training data sets and applying to the test sets, we will only keep levels contained in the training data sets!
# 
# We simply filter the columns by data type, construct the object levels, impute, convert to dummy notation and drop all zero columns

# In[6]:

object_columns = X_train.columns[X_train.dtypes == object].tolist()
object_levels = np.union1d(X_train[object_columns].fillna('NAN'), X_test[object_columns].fillna('NAN'))

categorical_cols_pipeline = make_pipeline(
        SelectColumnsTransfomer(object_columns),
        DataFrameFunctionTransformer(lambda x:'NAN', impute=True),
        DataFrameFunctionTransformer(lambda x:x.astype('category', categories=object_levels)),
        ToDummiesTransformer(),
        DropAllZeroTrainColumnsTransformer()
    )


# ### Remaining columns
# The remaining columns are mostly integer columns. However, if an integers column in a training set has a missing value in the test set the data type will be float in the test set. So, without actually using information beforehand, we just convert the remaining columns to float, store the mean and impute if necessary on train and test sets. 
# 
# Typically, integer columns provide some form of count data, hence we do not use a log transform here. Note however, that the GarageBltYear is also a remaining column. For simplicity, we treat it like the other count-like columns

# In[7]:

remaining_cols = [x for x in X_train.columns.tolist() if x not in object_columns and x not in area_cols]

remaining_cols_pipeline = make_pipeline(
        SelectColumnsTransfomer(remaining_cols),
        DataFrameFunctionTransformer(func = lambda x: x.astype(np.float64)),
        DataFrameFunctionTransformer(func = np.mean, impute=True)
    )


# ### Uniting the pipelines
# We put the pipelines together using the DataFrameFeatureUnion transformer. To demonstrate that we get a DataFrame we simply use fit_transform on the training set and show the first rows

# In[34]:

print("Training has {} columns:\n".format(len(X_train.columns)))
X_train.columns


# In[42]:

print("There are {} area columns:\n".format(len(area_cols)))
area_cols

play_area_features = DataFrameFeatureUnion([area_cols_pipeline,])

print("\tAfter fitting area columns, result has {} columns".format(
    len(play_area_features.fit_transform(X_train).columns))
)
transformed_area_cols = play_area_features.fit_transform(X_train).columns

transformed_area_cols


# In[43]:

print("There are {} categorical columns:\n".format(len(object_columns)))
object_columns

play_cat_features = DataFrameFeatureUnion([categorical_cols_pipeline,])

print("\tAfter fitting categorical columns, result has {} columns".format(
    len(play_cat_features.fit_transform(X_train).columns))
)
transformed_cat_cols = play_cat_features.fit_transform(X_train).columns
transformed_cat_cols


# In[37]:

print("There are {} other columns:\n".format(len(remaining_cols)))


# In[44]:

print("So, after transformation, result should have {} columns.".format(
       len(transformed_area_cols) + len(transformed_cat_cols) + len(remaining_cols)
    )
     )


# In[8]:

preprocessing_features = DataFrameFeatureUnion([area_cols_pipeline, categorical_cols_pipeline, remaining_cols_pipeline])
preprocessing_features.fit_transform(X_train).head()


# ## Gridsearch and Crossvalidation
# We use nested cross validation to estimate the generalization performance. See the 3rd example [here](http://scikit-learn.org/stable/modules/grid_search.html#grid-search)
# 
# Unfortunately, nested cross validation is not able to return the best model parameters for each fold ([and probably never will be](https://github.com/scikit-learn/scikit-learn/issues/6827)). However, for simplicity, we just assume that the model parameters are stable across the cross validation folds on the training sets.

# In[45]:

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
pipe_ridge = make_pipeline(preprocessing_features, Ridge())
param_grid = {'ridge__alpha' : [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}
pipe_ridge_gs = GridSearchCV(pipe_ridge, param_grid=param_grid, scoring = 'neg_mean_squared_error', cv=3)
result = np.sqrt(-cross_val_score(pipe_ridge_gs, X_train, np.log(y_train), scoring = 'neg_mean_squared_error', cv = 5))
np.mean(result)


# So the result is relatively close to that of [Alexandra Papiu](https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models). But I was not yet able to figure out where the differences come from exactly. One differnece is that I seem to construct more columns.
# 
# Let me comment on the scoring parameter. Why 'neg_mean_squared_error'? This is a scikit learn convention that ensures that grid search and cross validation always [maximize](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) a specific score.
# 
# Additionally, you might want to ask, why we don't use a custom scoring function and than use y_train instead of np.log(y_train). 
# The reason is simply, that the loss functions in the algorithms are typically some form of squared loss. Using the np.log transformation, the optimization during model training directly optimizes the correct loss function. If we use a custom scorer object, the internal optimization is performed using standard quadratic error functional and we simply evaluate the results using that custom scorer. Those results do not necessarily have to be the same and perform worse in general (you can even test it here).
# 
# For completeness, let us just fit an optimized model on the full training data and provide a data set for submission

# In[46]:

pipe_ridge_gs.fit(X_train, np.log(y_train))
predicted = np.exp(pipe_ridge_gs.predict(X_test))
X_test["SalePrice"] = predicted
X_test["SalePrice"].reset_index().to_csv('pipe_ridge_gs.csv', index=False)
pipe_ridge_gs.best_params_


# ## Further comments on additional transformers, next steps and references
# * The FeatureUnion is not yet able to allow parallel processing
# * One might want to construct interaction terms only on e.g. the garage columns. If the year is not available, it typically means that no garage is present. So storing the NaNs of such a column and building interaction terms with e.g. the area might provide viable information. To this end, one would need a transformer that selects columns by a somewhat dynamical name pattern after a first FeatureUnion.
# * I discovered the [sklearn-pandas](https://github.com/paulgb/sklearn-pandas) package just recently. But it seems, that no dataframes but numpy arrays are returned. Thus, the previous idea will be really hard to implement. Nevertheless, it may be easier to just build a transformer
# * Unit tests are missing. Transformers should e.g. only accept DataFrame objects
# * I got several ideas from [Zac Stewart's](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html) blog entry. However, I didn't find source codes for the transformer, so I constructed them on my own .
