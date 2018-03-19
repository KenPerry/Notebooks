import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from datetime import timedelta
import dateutil.relativedelta as rd

nullTimeDelta = rd.relativedelta(days=0)

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
        """
        data: DataFrame containing indendent and dependent variables
        - last column is the dependent; all others are independent
        """
        
        self.data = data.copy()
        self.Debug = debug

    
    def fit_mat(self, mat):
        """
        mat is a matrix:
        - last column is the dependent variable
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

        Regress dependent (last column of df) against all other columns, on a rolling bsis

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


    def get(self, indCols, depCol):
        """
        Return the sub-DataFrame of self.data that only contains columns (in the order listed):
        - indCols (independent variables)
        - depCol  (dependent variable)
        """
        
        cols = indCols +  [ depCol ]
        print("IndCols: {}, depCol {}, cols {}".format(indCols, depCol, cols))
        data  = self.data
    
        return data.loc[:, cols ]

    def sensAttrsDef(self, numBetas):
        """
        Return list of attributes that will store the sensitivities

        numBetas: The number of sensitivities. It is the number of independents (plus one, if there is an interecept)
        """

        # Intercept is an additional sensitivity attribute (attribute 0_
        attrs = [ "Beta {:d}".format(i)  for i in range(0,numBetas) ]
        return attrs
    
    def rollingFit(self, df, start, end, windowTimeDelta, stepTimeDelta=nullTimeDelta):
        """

        Rolling regression of a single dependent.  That is, a fit is performed on multiple dates.

        - the dates for fitting end (inclusive) at (e- i * stepTimeDelta) for i=0, ...
        - fitting for end date e includes data df.loc[e- windowTimeDleta + oneDayTimeDelta: e,:].  Since e is a datetime, date e IS included.

        df is a DataFrame
        start, end are datetimes
        windowTimeDelta is a timedelta.  It is the length of the window for fitting.
        stepTimeDelta   is a timedelta.  It is the interval at which successive fittings are done.

        Regress dependent (last column of df) against all other columns, on a rolling basis:
        All windows of length windowTimeDelta.  The last window ends on end; each previous window ends windowTimeDelta prior
        """

        if (stepTimeDelta == nullTimeDelta):
            stepTimeDelta = windowTimeDelta
            if self.Debug:
                print("rollingFit: step size set to window size")
            
        results = []

        numInds = df.shape[-1] -1

        # Interecept is an additional beta
        numBetas = numInds + 1
        colNames = [ "Dt" ] + self.sensAttrsDef(numBetas)
        
        firstDate = self.data.index[0]
        
        oneDayTimeDelta = timedelta(days=1)

        # End e (inclusive) and start s for first fit
        (e,s)  = (end, end - windowTimeDelta + oneDayTimeDelta)

        while (s >= start):
            # NOTE: s,e are DateTimes so s:e are all the rows INCLUSIVE of e (as opposed to if s,e were integers, in which case last row is (e-1)
            thisDf = df.loc[s:e,:]

            if self.Debug:
                print("rollingFit: {s} to {e}".format(s=s, e=e))
                
            intercept, betas = self.fit(thisDf)

            results.append( (e, intercept, *betas) )

            e = e - stepTimeDelta
            s = e - windowTimeDelta + oneDayTimeDelta



        result_df = pd.DataFrame( results, columns= colNames )
        result_df = result_df.set_index("Dt")
        result_df = result_df.sort_index()

        return result_df

    def rollingModel(self, indCols, depCol, start, end, windowTimeDelta, stepTimeDelta=nullTimeDelta):
        """
        Rolling regression

        indCols are the independent variables
        depCol is the dependent variables

        start, end, windowTimeDelta are as in rollingFit
        """
        
        df = self.get(indCols, depCol)
        result = self.rollingFit(df, start, end, windowTimeDelta, stepTimeDelta)

        return result

    def rollingModelAll(self, indCols, depCols, start, end, windowTimeDelta, stepTimeDelta=nullTimeDelta):
        """
        Rolling regression, repeated separately with each non-dependent variable in self.data as the dependent

        indCols, depCols, start, end, windowTimeDelta are as in rollingModel
        """
        
        result = []

        # Perform rollingModel for each ticker in depCols. Create list of Dataframes, one per ticker
        for depCol in depCols:
            if self.Debug:
                print("rollingModel for {t}".format(t=depCol) )
            
            thisResult = self.rollingModel(indCols, depCol, start, end, windowTimeDelta, stepTimeDelta)
            result.append( thisResult )


        # Join the per-ticker Dataframes into one big Dataframe
        df_big  = pd.DataFrame
        if (len(result) > 0):
            depTickers = [ a[-1] for a in depCols ]
            df_big = pd.concat( result, axis=1, keys=depTickers)
            
            # Make the first level column index be attribute; the second will be ticker
            df_big = df_big.swaplevel(axis=1,i=0,j=1)

            # Always a good idea to sort after concat
            df_big.sortlevel(axis=0, inplace=True)
            df_big.sortlevel(axis=1, inplace=True)
        

        return df_big

    def modelCols(self, indCols):
        """"
        indCols is an array of independent variable columns (even length 1 must be array)

        Returns 
        - a pair ( ( independent variable columns  of self.data), dependent variable columns of self.data )
        """
        
        df = self.data

        # Get Index with column names
        if isinstance( df.columns, pd.MultiIndex):
            columns = df.columns
            depCols = list( zip( columns.get_level_values(0), columns.get_level_values(1) ) )
        else:
            depCols = df.columns

        # Remove independents
        for t in indCols:
            if t in depCols:
                depCols.remove(t)

        return (indCols, depCols)

class RegAttr:
    def __init__(self, data):
        self.data = data.copy()
    
    def addConst(self, colName, val):
        """
        Add a column to DataFrame df, with name colName, and the constant value val

        colName is a column name
        val is a value
        """
        df = self.data
        df.loc[:, colName] = val

        
    def depTickersFromSensAttrs(self, sensAttrs):
        """
        Return a list of tickers (level 1 of column names) of the dependent variables of df.
        These are determined by which tickers have sensitivities in the first sensitivity attribute

        sensAttrs is a list of sensitivity attributes
        """

        beta_df = self.beta_df
        depTickers = beta_df.loc[:, idx[ sensAttrs[0] ] ].columns.tolist()
        return depTickers

    
    def setSens(self, df):
        """ TO DO:
        Turn into inspector too !
        """
        
        self.beta_df = df
        
    def sensAttrs(self, pat):
        """
        Return a list of the attributes of the sensitivities
        The sensitivites are those matching the pattern pat

        pat is a pattern
        """

        beta_df = self.beta_df
        attrs = beta_df.columns.get_level_values(0).unique().tolist()
        sensAttrs = [ attr for attr in attrs if re.search(pat, attr) ]

        return sensAttrs
        
    def retAttrib_setup(self, indCols, depCols, sensAttrs):
        """

        depCols is a list of dependent column names
        indCols is a list of independent column names
        sensAttrs is a list of sensitivity attribute names. Length must be same as len(indCols); there is a positional correspondence
        """

        df = self.data
        beta_df = self.beta_df
        
        df_dep = df.loc[:, depCols]

        list_of_sens_dfs = []
        
        for sensAttr in sensAttrs:
            df_sens = beta_df.loc[:, idx[sensAttr,:] ]
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
    
                                
    def retAttrib(self, indCols, depCols, sensAttrs=[]):
        """
        Perform the return attribution:
        - indCols: names of columns of self.data holding independent variables
        - depCols: names of columns with the dependent variable (there are separate regressions per dependent, so mulitple dependents)
        - sensAttrs: (optional) names of attributes (not columns) of the sensitivities (betas) of dependent to independents
            should rarely need to specify; it defaults to the attribute names of self.beta_df.  Only need to specify is self.beta_df has extraneous non-sensitivity attributes


        Returns:
        dataframe with attributes for:
        - contribution from each independent
        - the total, across all independents,  contribution
        - the residual (difference between dependent and total contribution from independents

        Pre-requisites:
        self.data: Dataframe containing the dependent and indendent variables
        - this must have the columns named in the parameters: depCols, indCols
        self.beta_df: Dataframe containing the sensitivites.
        - this must have the attributes named in the parameter: sensAttrs
        """

        df = self.data
        beta_df = self.beta_df

        # sensAttrs:names of attributes (not columns) of the sensitivities (betas) of dependent to independents
        if (len(sensAttrs) == 0):
            sensAttrs = beta_df.columns.get_level_values(0).unique().tolist()

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
    
