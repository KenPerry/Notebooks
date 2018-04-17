from trans.dataprovider.PDR.base import PDRBase

class Tiingo(PDRBase):
    def __init__(self, **params):
        super().__init__("tiingo", **params)

    def modify(self, df):
        df.index = df.index.droplevel(0)
        
        df.index = df.index.rename('Date')
        return df

    
    
