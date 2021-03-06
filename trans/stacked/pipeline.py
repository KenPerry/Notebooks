from trans.stacked.base import StackBase

class Pipeline_stack(StackBase):
    def __init__(self, members, debug=False, **params):
        self.pipe = members

        self.Debug = debug


    # INTERNAL methods: take df as argument
    #-------------------------------------------------------
    
    # EXTERNAL methods: takes df from SELF.df
    #-------------------------------------------------------
    def init(self, members,**params):
        pipe = self.pipe

        obj = pipe[0]
        return obj.init(**params)

    
    def is_next(self):
        """
        Return True if next call to self.nextChunk will succeed

        """
        # Delegate to first member of piple
        pipe = self.pipe

        obj = pipe[0]
        return obj.is_next()
    
    def nextChunk(self):
        """
        Return a chunk any which way you'd like.  Do we even need start and end ?
        """

        # Delegate to first member of piple
        pipe = self.pipe

        obj = pipe[0]
        return obj.nextChunk()

    def do(self, df):
        pipe = self.pipe

        # Apply the first member of the pipeline to the input DataFrame
        obj = pipe[0]
        next_df = obj.do(df)

        # Apply the output of the pipe's previous member as input to the next member
        for obj in pipe[1:]:
            # Apply the next member of the pipeline to the output of the previous member
            next_df = obj.do(next_df)

        return next_df

