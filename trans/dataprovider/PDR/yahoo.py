from trans.dataprovider.PDR.base import PDRBase

class Yahoo(PDRBase):
    def __init__(self, **params):
        super().__init__("yahoo", **params)


    def modify(self, df):
        df.index = df.index.rename('Date')
        return df
