from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

from trans.pca import PrincipalComp

from datetime import timedelta

from trans.stacked.base import StackBase

class PCA_stack(StackBase):
    def __init__(self, debug=False, **params):
        self.Debug = debug


    # INTERNAL methods: take df as argument
    #-------------------------------------------------------

    def pca(self, df):
        """
        CATCH: doesn't return a DataFrame: relevant info is stored within the pca object created
        """

        # Do a (non-rolling) PCA on entire DataFrame
        pco = PrincipalComp(df)
        result = pco.singlePCA()

        return result
    
    
    # EXTERNAL methods: takes df from SELF.df
    #-------------------------------------------------------
    def init(self, **params):
        if "stack" in params:
            self.stack = params["stack"]
        else:
            print("No stack arg. present")


        # Internal state to define position within stakc of next chunk to be returned by nextChunk
        self.idx = 0

    def is_next(self):
        """
        Return True if next call to self.nextChunk will succeed
        """

        return self.idx < len(self.stack)
    
    def nextChunk(self):
        """
        Return a chunk any which way you'd like.  Do we even need start and end ?
        """
        stack, idx = self.stack, self.idx

        # Obtain current chunk
        stackItem = stack[idx]
        label, chunk = stackItem[0], stackItem[1]
        
        if self.Debug:
            print("nextChunk label {} with shape: {}".format(label, chunk.shape))

        # Update state that defines position of next chunk to be returned
        self.idx += 1

        return (label, chunk)
    

    def do(self, df):
        # Perform PCA computation
        return self.pca(df)
