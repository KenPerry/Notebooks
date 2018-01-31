import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from datetime import timedelta

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
        colNames = [ "Dt", "Intercept" ] + [ "Beta {:d}".format(i)  for i in range(1,numBetas +1) ]
        
        firstDate = self.data.index[0]
        
        oneDayTimeDelta = timedelta(days=1)
        
        (e,s)  = (end, end - windowTimeDelta + oneDayTimeDelta)

        while (s >= start):
            # thisMat = np.asarray( df.loc[s:e,:] )
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

    def rollingModel(self, indAttr, depAttrs, start, end, windowTimeDelta):
        """
        Rolling regression

        indAttr are the independent variables
        depAttrs are the dependent variables

        start, end, windowTimeDelta are as in rollingFit
        """
        
        df = self.get(indAttr, depAttrs)
        result = self.rollingFit(df, start, end, windowTimeDelta)

        return result

    def rollingModelAll(self, indAttrs, depAttrs, start, end, windowTimeDelta):
        """
        Rolling regression, repeated separately with each non-dependent variable in self.data as the dependent

        indAttrs, depAttrs, start, end, windowTimeDelta are as in rollingModel
        """
        
        result = []

        # Perform rollingModel for each ticker. Create list of Dataframes, one per ticker
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

    def modelAttrs(self, indAttrs):
        """
        depAttrs is an array of dependent variables (even length 1 must be array)

        Returns 
        - a pair ( (all non-dependent variables of self.data), dependent variables of self.data )
        """
        
        df = self.data

        # Get Index with column names
        if isinstance( df.columns, pd.MultiIndex):
            columns = df.columns
            tickers = list( zip( columns.get_level_values(0), columns.get_level_values(1) ) )
        else:
            tickers = df.columns

        # Remove Dependents
        for t in indAttrs:
            if t in tickers:
                tickers.remove(t)

        return ( indAttrs, tickers)


        
