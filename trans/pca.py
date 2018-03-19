import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import re
from datetime import timedelta

from trans.gtrans import *

idx = pd.IndexSlice



class PrincipalComp:
    """

    """

    # INTERNAL methods: take df as argument
    #-------------------------------------------------------

    def __init__(self, df, debug=False, **params):
        """
        data: DataFrame containing
        """
        
        self.df = df.copy()
        self.Debug = debug


    def scale_df(self, df):
        """
        Apply sklearn StandardScaler to df (to convert each column into a z-score.
        Need to do this to gt PCA of correlation matrix, rather than covariance matrix
        """

        scale_pl = make_pipeline( SklearnPreproccessingTransformer( StandardScaler() ) )

        scaled_df = scale_pl.fit_transform(df)

        return scaled_df
    
    def fit(self,  df):
        """
        df is a DataFrame
        """

        # Scale the data (convert each column to z-score)
        scaled_df = self.scale_df(df)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(scaled_df)
        principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

        return { "pcs": principalComponents,
                 "pca": pca
            }

    def rollingPCA(self, df, start, end, windowTimeDelta, stepTimeDelta=None):
        """
        """

        # Get column names: used for labeling output
        if isinstance( df.columns, pd.MultiIndex):
            varNames = df.columns.get_level_values(1).tolist()
        else:
            varNames = df.columns
                        
        firstDate = self.df.index[0]
        
        oneDayTimeDelta = timedelta(days=1)

        results = []
        
        # End e (inclusive) and start s for first fit
        (e,s)  = (end, end - windowTimeDelta + oneDayTimeDelta)

        while (s >= start):
            # NOTE: s,e are DateTimes so s:e are all the rows INCLUSIVE of e (as opposed to if s,e were integers, in which case last row is (e-1)
            thisDf = df.loc[s:e,:]

            if self.Debug:
                print("rollingFit: {s} to {e}".format(s=s, e=e))
                
            result  = self.fit(thisDf)

            results.append( (e, result) )

            e = e - stepTimeDelta
            s = e - windowTimeDelta + oneDayTimeDelta

        # Convert results to a DatFrame
        rows, dates = [], []

        # Create one row per reslut
        for result in results:
            # Build the column labels too: should be same for each iteration
            row, cols = [], []

            # Extract the date of the pca, and the PCA object (which contains parameters)
            pca_date, pca = result[0], result[1]["pca"]

            # Row starts of with date
            # row, cols  = [ pca_date ], ["Dt"]
            dates.append(pca_date)

            # Extend the row with each component; there are pca.n_components_ of them
            comps = pca.components_
            num_comps = pca.n_components_
            

            # Add each component
            for comp_num in range(num_comps):
                row.extend(comps[comp_num])

                # Label is (component number, varName)
                comp_name = "PC {}".format(comp_num)
                                   
                cols.extend( list( map(lambda t: (comp_name, t), varNames) ) )

            # Extend the row with explained variance; there are pca.n_components_ of them
            row.extend( pca.explained_variance_ )
            cols.extend( [ ("Explained Var", i)      for i in range(num_comps) ] )
                          
            # Extend the row with explained_variance_ratio; there are pca.n_components_ of them
            row.extend( pca.explained_variance_ratio_)
            cols.extend( [ ("Explained Var Ratio", i) for i in range(num_comps) ] )

            rows.append(row)

        if self.Debug:
            print("Columns: ", cols)

        df = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(cols), index=dates)
        df.index.rename("Dt", inplace=True)

        # n.b., good_housekeeping resulting DataFrame doesn't seem to recognize that level 0 spans level 1 columns when displayed, but sort_index w/o level arg seems OK
        # Maybe an artifact of columns created as MultiIndex ?
        # good_housekeeping(df, inplace=True)
        df.sort_index(axis=1, inplace=True)

        return df


    # EXTERNAL methods: takes df from SELF.df
    #-------------------------------------------------------


    def singlePCA(self):
        """
        Single PCA using entire DataFrame (in self.data)
        """

        df = self.df
        start, end = df.index.min(), df.index.max()
        window = end - start + timedelta(days=1)
        step   = window

        results = self.rollingPCA(df, start, end, window, step)

        return results
        # Result is an  (single-element) array of pairs (end, pca_result).  Return just pca_result of sole member of result
        cols, rows = results
        
        return cols, rows[0]




        
 
