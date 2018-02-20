import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from datetime import timedelta

import re


from trans.gtrans import DataFrameConcat

idx = pd.IndexSlice

"""
For a MultiIndex column with two levels: L1, L2
- the members of L1 are referred to as "attributes"
- the members of L2 are referred to as "tickers"
- the pair (l1, l2)  -- or equivalently: idx[l1, l2] --  for l1 in L1 and l2 in L2 is referred to as a "column name"
- So the attribute l1 refers to a groups of columns (l1, l2), for all l2 in L2
"""

class Reg:
    def __init__(self, data, debug=False):
        self.data = data
        self.Debug = debug

    
    def fit_mat(self, mat):
        """
        mat is a matrix:
        - first column is the dependent variable
        - all other columns are independent variables
    
        """
        
        lr = LinearRegression()
        
        dep = mat[:,-1]
        ind = mat[:,0:-1]
        
        lr.fit( ind, dep )
        interc, betas = lr.intercept_, lr.coef_

        return interc, betas
     

    def fit(self,  df):
        """
        df is a DataFrame

        Regress dependent (first column of df) against all other columnsw, on a rolling bsis

        Convert df to a matrix and then call fit_mat
        """

        # Error return values
        interc_empty = np.nan
        betas_empty = [ np.nan ] * ( -1 + df.shape[1])
         
        # Make sure df is non-empty
        if (df.shape[0] == 0):
            print("fit: empty dataframe")
           
            return interc_empty, betas_empty

        # Make sure df does not contain nulls
        colHasNan = df.isnull().any()
        colIdx = colHasNan.index
        
        if isinstance( colIdx, pd.MultiIndex):
            colIdx = colHasNan.index.get_level_values(1)

        colNanNames = colIdx[ colHasNan ].tolist()
            
        if (len(colNanNames) > 0):
            print("Fit: the following columns have naN: ", colNanNames)
            return interc_empty, betas_empty            
        
        mat = np.asarray(df)
        return self.fit_mat(mat)


    def get(self, indAttrs, depAttr):
        """
        Return the sub-DataFrame of self.data that only contains columns
        - indAttrs (independent variables)
        - depAttr  (dependent variable)
        """
        
        attrs = indAttrs +  [ depAttr ]
        data  = self.data
    
        return data.loc[:, attrs ]

    
    def rollingFit(self, df, start, end, windowTimeDelta):
        """

        Rolling regression of a single dependent

        df is a DataFrame
        start, end are datetimes
        windowTimeDelta is a timedelta

        Regress dependent (first column of df) against all other columnsw, on a rolling basis:
        All windows of length windowTimeDelta.  The last window ends on end; each previous window ends windowTimeDelta prior
        """
        
        results = []

        numBetas = df.shape[-1] -1
        colNames = [ "Dt" ] + [ "Beta {:d}".format(i)  for i in range(0,numBetas +1) ]
        
        firstDate = self.data.index[0]
        
        oneDayTimeDelta = timedelta(days=1)
        
        (e,s)  = (end, end - windowTimeDelta + oneDayTimeDelta)

        while (s >= start):
            # NOTE: s,e are DateTimes so s:e are all the rows INCLUSIVE of e (as opposed to if s,e were integers, in which case last row is (e-1)
            thisDf = df.loc[s:e,:]

            if self.Debug:
                print("rollingFit: {s} to {e}".format(s=s, e=e))
                
            intercept, betas = self.fit(thisDf)

            results.append( (e, intercept, *betas) )

            e = e - windowTimeDelta
            s = e - windowTimeDelta + oneDayTimeDelta



        result_df = pd.DataFrame( results, columns= colNames )
        result_df = result_df.set_index("Dt")
        result_df = result_df.sort_index()

        return result_df

    def rollingModel(self, indAttrs, depAttr, start, end, windowTimeDelta):
        """
        Rolling regression

        indAttrs are the independent variables
        depAttr is the dependent variables

        start, end, windowTimeDelta are as in rollingFit
        """
        
        df = self.get(indAttrs, depAttr)
        result = self.rollingFit(df, start, end, windowTimeDelta)

        return result

    def rollingModelAll(self, indAttrs, depAttrs, start, end, windowTimeDelta):
        """
        Rolling regression, repeated separately with each non-dependent variable in self.data as the dependent

        indAttrs, depAttrs, start, end, windowTimeDelta are as in rollingModel
        """
        
        result = []

        # Perform rollingModel for each ticker in depAttrs. Create list of Dataframes, one per ticker
        for depAttr in depAttrs:
            if self.Debug:
                print("rollingModel for {t}".format(t=depAttr) )
            
            thisResult = self.rollingModel(indAttrs, depAttr, start, end, windowTimeDelta)
            result.append( thisResult )


        # Join the per-ticker Dataframes into one big Dataframe
        df_big  = pd.DataFrame
        if (len(result) > 0):
            depTickers = [ a[-1] for a in depAttrs ]
            df_big = pd.concat( result, axis=1, keys=depTickers)
            
            # Make the first level column index be attribute; the second will be ticker
            df_big = df_big.swaplevel(axis=1,i=0,j=1)

            # Always a good idea to sort after concat
            df_big.sortlevel(axis=0, inplace=True)
            df_big.sortlevel(axis=1, inplace=True)
        

        return df_big

    def modelCols(self, indAttrs):
        """"
        indAttrs is an array of independent variable columns (even length 1 must be array)

        Returns 
        - a pair ( (all non-dependent variables of self.data), dependent variable columns of self.data )
        """
        
        df = self.data

        # Get Index with column names
        if isinstance( df.columns, pd.MultiIndex):
            columns = df.columns
            tickers = list( zip( columns.get_level_values(0), columns.get_level_values(1) ) )
        else:
            tickers = df.columns

        # Remove independents
        for t in indAttrs:
            if t in tickers:
                tickers.remove(t)

        return ( indAttrs, tickers)


    def addConst(self, colName, val):
        """
        Add a column to DataFrame df, with name colName, and the constant value val

        colName is a column name
        val is a value
        """
        df = self.data
        df[colName] = val

        
    def depTickersFromSensAttrs(self, sensAttrs):
        """
        Return a list of tickers of the dependent variables of df.
        These are determined by which tickers have sensitivities in the first sensitivity attribute

        sensAttrs is a list of sensitivity attributes
        """

        df = self.data
        depTickers = df.loc[:, idx[ sensAttrs[0] ] ].columns.tolist()
        return depTickers

    def sensAttrs(self, pat):
        """
        Return a list of the attributes of the sensitivities
        The sensitivites are those matching the pattern pat

        pat is a pattern
        """

        df = self.data
        attrs = df.columns.get_level_values(0).unique().tolist()
        sensAttrs = [ attr for attr in attrs if re.search(pat, attr) ]

        return sensAttrs
        
    def retAttrib_setup(self, indCols, depCols, sensAttrs):
        """

        depCols is a list of dependent column names
        indCols is a list of independent column names
        sensAttrs is a list of sensitivity attribute names. Length must be same as len(indCols); there is a positional correspondence
        """

        df = self.data
        df_dep = df.loc[:, depCols]

        list_of_sens_dfs = []
        
        for sensAttr in sensAttrs:
            df_sens = df.loc[:, idx[sensAttr,:] ]
            list_of_sens_dfs.append( df_sens )


        list_of_ind_dfs = []
        
        for indCol in indCols:
            df_ind = df.loc[:, [indCol] ]
            list_of_ind_dfs.append(df_ind)

        return (df_dep, list_of_ind_dfs, list_of_sens_dfs)
    

    def retAttrib_to_np(self, list_of_ind_dfs, df_dep, list_of_sens_dfs):
        """
        Return NumPy arrays corresponding to the inputs

        df_dep: DataFrame of dependent variable
        list_of_ind_dfs:  list of DataFrames, one per independent variable
        list_of_sens_dfs: list of DataFrames, one per sensitivity (to independent variable)
        """
        

        dep_mat = df_dep.as_matrix()
        ind_mat  = np.vstack( [ df.as_matrix()[ np.newaxis, ...] for df in list_of_ind_dfs ] )
        sens_mat = np.vstack( [ df.as_matrix()[ np.newaxis, ...] for df in list_of_sens_dfs ] )

        return(ind_mat, dep_mat, sens_mat)

    def retAttrib_np(self, ind_mat, dep_mat, sens_mat):
        """
        Compute contribution of each independent variable, give the sensitivity to the independent variable and the return of the independent variable.
        Returns
        contribs_mat: contribution indepdent variables
        predict_mat:  sum, across indpedentdent variables, of contribs
        err_mat:      difference between dependent and predict
        """
        
        contribs_mat = sens_mat * ind_mat
        predict_mat  = np.sum(contribs_mat, axis=0)
        err_mat      = dep_mat - predict_mat

        return(contribs_mat, predict_mat, err_mat)

    def retAttrib_to_df(self, contribs_mat, predict_mat, err_mat, dates, indCols, depTickers):
        """
        Return the DataFrames corresponding to the NumPy matrices
        """
        list_of_contribs_dfs = [ pd.DataFrame(contribs_mat[i, ...], index=dates, columns=depTickers) for i in range( contribs_mat.shape[0]) ]
        predict_df = pd.DataFrame(predict_mat, index=dates, columns=depTickers)
        err_df     = pd.DataFrame(err_mat, index=dates, columns=depTickers)

        return(list_of_contribs_dfs, predict_df, err_df)
    
                                
    def retAttrib(self, indCols, depCols, sensAttrs):
        """
        Perform the return attribution:
        - indCols: names of columns of self.data holding independent variables
        - depCols: names of columns with the dependent variable (there are separate regressions per dependent, so mulitple dependents)
        - sensAttrs:names of attributes (not columns) of the sensitvities (betas) of dependent to independents

        Returns:
        dataframe with attributes for:
        - contribution from each independent
        - the total, across all independents,  contribution
        - the residual (difference between dependent and total contribution from independents
        """

        df = self.data
        
        # Get the tickers of the dependent variable
        #  These are tickers that have sensitivity to the attributes in sensAttrs
        depTickers = self.depTickersFromSensAttrs(sensAttrs )

        # Get the tickers of the independent variables
        indTickers = [ col[1] for col in indCols ]

        # Get DataFrames for:
        # df_dep: dependent variable
        # list_of_ind_dfs: list of DataFrames,  one per independent variable
        # list_of_sens_dfs: list of DataFrames, one per (sensitivity) independent variable
        (df_dep, list_of_ind_dfs, list_of_sens_dfs) = self.retAttrib_setup(
                                                                           indCols, 
                                                                           depCols, 
                                                                           sensAttrs)
        
        # Convert the DataFrames to NumPy matrices
        (indMat, depMat, sensMat)  = self.retAttrib_to_np(list_of_ind_dfs, df_dep, list_of_sens_dfs)

        # Perform the attribution on the matrices, returning matrices
        (contribsMat, predMat, errMat) = self.retAttrib_np(indMat, depMat,sensMat)

        # Convert the matrices back to DataFrames
        (list_of_contribs_dfs, predict_df, err_df) = self.retAttrib_to_df(contribsMat, 
                                                                          predMat, 
                                                                          errMat,
                                                                          df.index,
                                                                          sensAttrs,
                                                                          depTickers)

        # Create a single DataFrame with the result of the attribution
        # Create the attribute for each contribution (i.e., from each independent variable)
        contrib_k = [ "Contrib from " + indTickers[i] for i in range( len(list_of_contribs_dfs ))]


        # Create a transformation to concatenate the individual DataFrames, using an attribute for each
        regResTrans = DataFrameConcat( list_of_contribs_dfs + [ predict_df ]
                                       + [ err_df ],
                                       df_keys= contrib_k   + [ "Predicted" ]  + ["Error"])

        regRes_df = regResTrans.fit_transform(pd.DataFrame())

        return regRes_df
    
