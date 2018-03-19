from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

from trans.pca import PrincipalComp

from datetime import timedelta

from trans.stack_abs import StackBase

class Residual(StackBase):
    def __init__(self, debug=False, **params):
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
    def init(self, **params):
        self.df    = params["df"]
        self.start = params["start"]
        self.end   = params["end"]

        self.windowTimeDelta = params["window"]
        self.stepTimeDelta   = params["step"]

        oneDayTimeDelta = timedelta(days=1)

        # Internal state to define end/start of next chunk to be returned by nextChunk
        (self.e, self.s)  = (self.end, self.end - self.windowTimeDelta + oneDayTimeDelta)


    def is_next(self):
        """
        Return True if next call to self.nextChunk will succeed
        """
        return (self.s >= self.start)
    
    def nextChunk(self):
        """
        Return a chunk any which way you'd like.  Do we even need start and end ?
        """
        df = self.df
        start, end = self.s, self.e

        # Obtain current chunk
        chunk = df.loc[start:end,:]
        label = end

        oneDayTimeDelta = timedelta(days=1)
        
        if self.Debug:
            print("nextChunk for period {} to {} shape: {}".format(start, end, chunk.shape))

        # Update state that defines end/start of next chunk to be returned
        self.e = self.e - self.stepTimeDelta
        self.s = self.e - self.windowTimeDelta + oneDayTimeDelta

        return (label, chunk)
    

    def do(self, df):
        # Perform residaul computation
        return self.residual(df)
