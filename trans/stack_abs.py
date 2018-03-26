from trans.data import GetData
gd = GetData()
from trans.gtrans import *
from trans.reg import Reg, RegAttr
from trans.regpipe import RegPipe

from trans.pca import PrincipalComp

from datetime import timedelta

"""
Base class for Stacked computation

A computation stack is a sequence resulting from a single function (e.g., foo(X))  applied succesively to each element in an input sequence of data (e.g., X's: X_0, X_1, ...)

Each element of the input sequence is a DataFrame, and each computaton output is a tuple consisting of a label and a DataFrame.
So the result of a computation stack is [ (label_0, foo(X_0)) , f(label_1, foo(X_1)0, ... ]


The next input in the sequence (a DataFrame, thisDf = X_i) is obtained by the method "nextChunk".
This input is applied to the computation via the method "do", i.e., self.do(thisDf)
The result of this computation is paired with a label, and the sequence of pairs is the output of the stacked computation

The method "repeated" applies this computation sequence by repeatedly executing: 
   thisDf = self.nextChunk;
   result = self.do(thisDf)

   results.append( (thisLabel, result) )

So the result is an array of pairs, each pair being a label and a DataFrame
"""

class StackBase:
    def __init__(self, debug=False, **params):
        self.Debug = debug

    def init(self, **params):
        print("init: need to override")

    def nextChunk(self):
        """
        Abstract method.
        Returns next chunk (DataFrame) of data.
        """
        print("nextChunk: need to override")        


    def do(self, df):
        """
        Abstract method.
        Applies the computation to the next chunk (DataFrame)
        """
        print("do: need to override")

    def done(self):
        """
        Abstract method.
        Can be optionally called at end of sequence of computations.  For cleanup or finalization
        """
        print("done: need to override")

    def is_next(self):
        """
        Return True if next call to self.nextChunk will succeed
        """

        print("is_next: need to override")
        
    def repeated(self):
        """
        Repeatedly: obtain next chunk of data, apply computation to it, save the pair (label, result) to the sequence of results
        """

        results = []
        

        while ( self.is_next() ):
            thisLabel, thisDf = self.nextChunk()

            if self.Debug:
                print("repeated: chunk label {} with shape {}".format(thisLabel, thisDf.shape))
                
            result  = self.do(thisDf)

            results.append( (thisLabel, result) )

           
        return results

