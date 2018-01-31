import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web

import numpy as np

import pickle
import requests

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline

import re

idx = pd.IndexSlice

class GetData:
    
    def save_sp500_tickers():
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)

        with open("sp500tickers.pickle","wb") as f:
            pickle.dump(tickers,f)

        return tickers



    def get_data_from_yahoo(reload_sp500=False):

        if reload_sp500:
            tickers = save_sp500_tickers()
        else:
            with open("sp500tickers.pickle","rb") as f:
                tickers = pickle.load(f)

        tickers.remove('BRK.B')
        tickers.remove('BF.B')
        if not os.path.exists('stock_dfs'):
            os.makedirs('stock_dfs')

        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2017, 12, 31)

        for ticker in tickers:

            # just in case your connection breaks, we'd like to save our progress!
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                df = web.DataReader(ticker, "yahoo", start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))



    def compile_data():
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

        main_df = pd.DataFrame()

        for count,ticker in enumerate(tickers):
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)

            df.rename(columns={'Adj Close':ticker}, inplace=True)
            df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
        print(main_df.head())
        main_df.to_csv('sp500_joined_closes.csv')



    def compile_data_all(tickers=None):
        if (tickers is None):
            with open("sp500tickers.pickle","rb") as f:
                tickers = pickle.load(f)

        main_df = pd.DataFrame()
        dfs = []

        for count,ticker in enumerate(tickers):
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)

            dfs.append(df)

            if count % 10 == 0:
                print(count)

        df_big = pd.concat( dfs, axis=1, keys=tickers)
        df_big.index.name = "Date"

        return df_big

    def save_data(df, file):
        with open(file,"wb") as fp:
            pickle.dump(df,fp)


    def load_data(file):
        with open(file,"rb") as fp:
            df = pickle.load(fp)

        return df
