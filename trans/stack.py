from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

from trans.pca import PrincipalComp

from datetime import timedelta

class Stack:
    def __init__(self, df, debug=False, **params):
        self.df = df
        self.Debug = debug


    # INTERNAL methods: take df as argument
    #-------------------------------------------------------

    def residual(self, df):
        # Do a (non-rolling) regression on entire DataFrame
        rps = RegPipe(df)
        rps.indCols( [ idx["Pct", "SPY"] ] )
        rps.regressSingle()

        # Set up for attribution
        #  - Backward fill the betas, no rolling.  That is the beta for the single regression is applied to the entire window
        rollAmount = 0
        fillMethod = "bfill"

        rps.attrib_setup(df, rps.beta_df, rollAmount, fillMethod)

        # Perfrom the attribution
        rps.attrib()

        # Access the residuals
        residuals = rps.retAttr_df.loc[:, idx["Error",:]]

        return residuals

    def pca(self, df):
        """
        CATCH: doesn't return a DataFrame: relevant info is stored within the pca object created
        """

        # Do a (non-rolling) PCA on entire DataFrame
        pco = PrincipalComp(df)
        res = pco.singlePCA()

        return res
    
    
    # EXTERNAL methods: takes df from SELF.df
    #-------------------------------------------------------

    def nextChunk(self, start, end):
        """
        Return a chunk any which way you'd like.  Do we even need start and end ?
        """
        df = self.df
        chunk = df.loc[start:end,:]

        if self.Debug:
            print("nextChunk for period {} to {} shape: {}".format(start, end, chunk.shape))
            
        return chunk

    def repeated(self, start , end, windowTimeDelta, stepTimeDelta):
        """
        TO DO: Abstract away self.residual(thisDf) to an over-ridden method. e.g., (residual fed into PCA) pipeline.  Issue is that PCA doesn't return DataFrame
        """
        firstDate = self.df.index[0]
        
        oneDayTimeDelta = timedelta(days=1)

        results = []
        
        # End e (inclusive) and start s for first fit
        (e,s)  = (end, end - windowTimeDelta + oneDayTimeDelta)

        while (s >= start):
            # NOTE: s,e are DateTimes so s:e are all the rows INCLUSIVE of e (as opposed to if s,e were integers, in which case last row is (e-1)
            thisDf = self.nextChunk(s,e)

            if self.Debug:
                print("rollingFit: {s} to {e}".format(s=s, e=e))
                
            result  = self.residual(thisDf)

            results.append( (e, result) )

            e = e - stepTimeDelta
            s = e - windowTimeDelta + oneDayTimeDelta


        return results
