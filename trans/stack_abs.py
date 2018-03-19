from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

from trans.pca import PrincipalComp

from datetime import timedelta

class StackBase:
    def __init__(self, debug=False, **params):
        self.Debug = debug

    def init(self, **params):
        print("init: need to override")

    def nextChunk(self):
        print("nextChunk: need to override")        


    def do(self, df):
        print("do: need to override")

    def done(self):
        print("done: need to override")

    def is_next(self):
        """
        Return True if next call to self.nextChunk will succeed
        """

        print("is_next: need to override")
        
    def repeated(self):
        """
        TO DO: Abstract away self.residual(thisDf) to an over-ridden method. e.g., (residual fed into PCA) pipeline.  Issue is that PCA doesn't return DataFrame
        """

        results = []
        

        while ( self.is_next() ):
            thisLabel, thisDf = self.nextChunk()

            if self.Debug:
                print("repeated: chunk label {} with shape {}".format(thisLabel, thisDf.shape))
                
            result  = self.do(thisDf)

            results.append( (thisLabel, result) )

           
        return results

