from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

from trans.pca import PrincipalComp

from datetime import timedelta

from trans.stacked.base import StackBase

class Residual(StackBase):
    def __init__(self, indCols=None, rollAmount=0, fillMethod="bfill", attr=None, debug=False, **params):
        self.Debug = debug

        if attr == None:
            attr = "Pct"
            print("DEPRECATED: {cls}:__init__ called w/o \"attr\" arg., defaulting to {defv}.".format(cls=type(self), defv=attr) )

        if indCols == None:
            indCols = [ idx[attr, "SPY"] ]
            print( "DEPRECATED: {cls}:__init__ called w/o \"indCols\" arg., defaulting to {defv}.".format(cls=type(self), defv=indCols) )

        self.attr = attr
        self.indCols = indCols
        self.rollAmount = rollAmount
        self.fillMethod = fillMethod


    # INTERNAL methods: take df as argument
    #-------------------------------------------------------

    def residual(self, df):
        # Do a (non-rolling) regression on entire DataFrame
        attr = self.attr
        
        rps = RegPipe(df, attr=attr)
        indCols = self.indCols
        rps.indCols( indCols )
        rps.regressSingle()

        if self.Debug:
            print("rps beta_df: ", rps.beta_df.tail())
        
        # Set up for attribution
        #  - Backward fill the betas, no rolling.  That is the beta for the single regression is applied to the entire window
        rollAmount = self.rollAmount
        fillMethod = self.fillMethod

        rps.attrib_setup(df, rps.beta_df, rollAmount, fillMethod)

        # Perfrom the attribution
        rps.attrib()

        if self.Debug:
            print("rps retAttr_df: ", rps.retAttr_df.tail())
        
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
