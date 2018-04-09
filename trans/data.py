import numpy as np
import pandas as pd

import bs4 as bs
import datetime as dt
import dateutil.parser as dup

import os
import shutil

import pandas_datareader.data as web


import pickle
import requests

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline

import re
import PyPDF2

idx = pd.IndexSlice

class GetData:
    def __init__(self):
        self.dir = "stock_dfs"
        return
    
    def file_from_ticker_(self, ticker):
        return '{dir}/{t}.csv'.format(dir=self.dir, t=ticker)
        
    def get_sp500_tickers(self, reload_sp500=False):
        """
        Get the list of current tickers in the S&P 500, using cached results if available

        Returns:
        list of tickers
        """
        if reload_sp500:
            tickers = self.save_sp500_tickers()
        else:
            with open("sp500tickers.pickle","rb") as f:
                tickers = pickle.load(f)

        return tickers
    
    def save_sp500_tickers(self):
        """
        Get the list of current tickers in the S&P 500

        The tickers are obtained by scraping a Wikipedia page, using Beautiful Soup

        Returns:
        list of tickers
        """
        
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

    def save_sp400_tickers(self):
        """
        Get the list of current tickers in the S&P 400

        The tickers are obtained by scraping a Wikipedia page, using Beautiful Soup

        Returns:
        list of tickers
        """
        
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)

        with open("sp400tickers.pickle","wb") as f:
            pickle.dump(tickers,f)

        return tickers

    def get_r1000_tickers(self, reload=False):
        """
        Get the list of current tickers in the Ruell 1000, using cached results if available

        Returns:
        list of tickers
        """

        file = "r1000tickers.pickle"

        if reload or not os.path.exists(file):
            tickers = self.save_r1000_tickers()
        else:
            with open(file,"rb") as f:
                tickers = pickle.load(f)

        return tickers

    def save_r1000_tickers(self):
        """
        Get the list of current tickers in the Rusell 1000

        The tickers are obtained from the PDF at ftserussell.com/memberhip-russell-com

        Returns:
        list of tickers
        """

        # file_pdf is downloaded from the above URL
        file_pdf = "/home/ubuntu/Downloads/ru1000_membershiplist_20160627.pdf"

        tickers = []
    
        pdfFileObj = open(file_pdf, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # Extract from each page
        for page in range( pdfReader.numPages):
            pageObj = pdfReader.getPage(page)
            line_num = 0

            text = pageObj.extractText()

            # Split the text into lines, and process each line
            for line in text.split("\n"):
                line_num  += 1
                line = line.rstrip(" ")

                # Tickers are no longer than 4 characters
                if ( len(line) == 0 or len(line) > 4):
                    continue

                # Tickers are not all digits
                if ( re.search("^\d+$", line) ):
                     continue

                # Tickers come from a limited character set
                if ( not re.search("^[A-Z\.]+$", line) ):
                    continue

                # Lines alternate: description, tickers
                if ( line_num % 2 == 0 ):
                    tickers.append(line)
                 
        with open("r1000tickers.pickle","wb") as f:
            pickle.dump(tickers,f)

        return tickers

    def get_one(self, ticker, start, end):
        """
        Get data from Web for single ticker for date range from start to end

        Parameters:
        - ticker: string
        - start, end: datetime

        Returns:
        status, DataFrame

        - status: True on success
        - DataFrame: resulting data
        """

        # Note: the index is of type datetime.  When written to a csv file, it will become a date string
        # Issue: if file has existing data as date, and new data comes in as datetime, can wind up with a mixture of date and datetime when combining, if not careful
        
        succeed, tries = False, 0

        start_d, end_d = dt.datetime.strftime(start, "%m/%d/%Y"), dt.datetime.strftime(end, "%m/%d/%Y")
        while( (tries < 2) and not succeed):
            try:
                df = web.DataReader(ticker, "yahoo", start, end)

                # Duplicates are returned sometimes.  Drop them
                df.drop_duplicates(inplace=True)

                succeed = True
            except Exception as e:
                tries += 1
                print("get_one: Yahoo exception for {t}: {et} - {ea}".format(t=ticker, et=type(e), ea=e))
                print("get_one: Yahoo error for {t}, re-try {tn}.".format(t=ticker, tn=tries))


        # Return empty DataFrame on failure
        if not succeed:
            df = pd.DataFrame()

        return succeed, df
    

    def extend(self, ticker, start, end):
        """
        Get data for a single ticker for date range from start to end, 
        If a file containing data for this ticker already exists, extend it to end if needed.

        Note:
        The Date column in a file is a date string (YYYY-MM-YY); the Date column from the Web is a datetime (sub-day resolution).
        Extend concatenates, resulting in the higher resolution Date when extending occurs.

        Parameters:
        - ticker: string
        - start, end: datetime


        Returns:
        status, DataFrame

        - status: True if new data has been obtained
        - DataFrame: the data (updated or original)
        """
        
        file = self.file_from_ticker_(ticker)

        changed = False
        
        if os.path.exists(file):
            # Note: the file's Date column is a string, so the index is of type "object"
            df = pd.read_csv(file)
            df.set_index('Date', inplace=True)

            # Get last date
            lastd = df.index.tolist()[-1]

            # Convert to datetime (use parse, it doesn't force you into a pre-specified format)
            # last_dt = dt.datetime.strptime(lastd, "%Y-%m-%d")
            last_dt = dup.parse(lastd)

            oneDayTimeDelta = dt.timedelta(days=1)

            # Get data for dates following lastd
            start_new = last_dt + oneDayTimeDelta

            if (start_new < end):
                print("Extend {t} from {l} beginning on {sn} to {e}.".format(t=ticker,
                                                                             l =dt.datetime.strftime(last_dt, "%m/%d/%Y"),
                                                                             sn=dt.datetime.strftime(start_new, "%m/%d/%Y"),
                                                                             e =dt.datetime.strftime(end, "%m/%d/%Y"))
                      )

                status, df_newer = self.get_one(ticker, start_new, end)

                # Did get_one really extend df ?
                # NOTE: the file index are Dates (MM/DD/YYYY), get_one index is DateTime

                if status and df_newer.index.min()  > dup.parse(df.index.max()):
                    changed = True

                    # Convert the index from datetime to date (since file format stores as date)
                    df_newer.reset_index(inplace=True)
                    new_dates = df_newer.loc[:, "Date"].map( lambda d_dt: dt.datetime.strftime(d_dt, "%Y-%m-%d") )
                    df_newer.drop([ "Date"], axis=1)
                    df_newer["Date"] = new_dates
                    df_newer.set_index("Date", inplace=True)

                    df = pd.concat([df, df_newer], axis=0)

        else:
            status, df = self.get_one(ticker, start, end)
            if status:
                changed = True
            
        return changed, df
            

    def get_data(self, tickers, start, end):
        """
        Get data for each ticker in tickers, from date range start to end
        - tickers: list of tickers
        - start, end: datetimes

        Returns:
        nothing

        Each ticker's data placed in file 'stock_dfs/{}.csv'.format(ticker)
        """

        # Ensure that output directory exists
        if not os.path.exists('stock_dfs'):
            os.makedirs('stock_dfs')

        changed_tickers = []
        
        # Get data for each ticker
        for ticker in tickers:
            changed, df = self.extend(ticker, start, end)
            
            # just in case your connection breaks, we'd like to save our progress!
            if changed:
                changed_tickers.append(ticker)
                
                file = self.file_from_ticker_(ticker)

                # Create a backup to avoid losing work
                if os.path.exists(file):
                    file_copy = file + '_bkup'
                    shutil.copy2(file, file_copy)
                    
                df.to_csv(file)
            else:
                print('Already have up-to-date {}.'.format(ticker))

        return changed_tickers
            
    def clean_data(self, tickers):
        """
        Clean up messes in each file:
        - Remove duplicate rows form each file.
        - Remove unnamed columns

        Messes will exists ONLY due to bugs and this will clean up

        """

        cleaned = []
        
        for ticker in tickers:
            file = self.file_from_ticker_(ticker)
            changed = False

            
            if os.path.exists(file):
                # Note: the file's Date column is a string, so the index is of type "object"
                df = pd.read_csv(file)

                # One screw-up: having added duplicate rows
                df_new = df.drop_duplicates( subset=[ "Date" ])

                if len( df_new.index ) < len (df.index):
                    print("Removed duplicates from {}.".format(ticker))
                    changed = True

                # Another screw-up: having added "Unnnamed" columns
                unnamed = [ c for c in df_new.columns.tolist() if re.search("Unnamed", c) ]
                if len(unnamed) > 0:
                    df_new.drop(unnamed, axis=1, inplace=True)
                    print("Removed unnamed columns for {}.".format(ticker))
                    changed = True

                if changed:
                    df_new.set_index('Date', inplace=True)
                    df_new.to_csv('stock_dfs/{}.csv'.format(ticker))

                    cleaned.append(ticker)

        return cleaned


        
    def existing(self):
        """
        Lists the tickers that have data files in self.dir

        Returns:
        List of tickers
        """
        
        tickers = []
        
        for f in os.listdir(self.dir):
            m = re.search("(.+)\.csv$", f)
            if m:
                tickers.append( m.group(1) )
                
        return tickers

    def combine_data(self, tickers=None, start=None, end=None):
        """
        Create one Dataframe, with data from all the tickers
        The Dataframe:
        - index is Date (as a string, same as in the per-ticker files)
        - columns: MultiIndex
        -  level 0: attribute
        -  level 1: ticker
        """
        
        if (tickers is None):
            with open("sp500tickers.pickle","rb") as f:
                tickers = pickle.load(f)

        main_df = pd.DataFrame()
        dfs = []

        # Get the data for each ticker as a Dataframe, appending to a list of the Dataframes
        for count,ticker in enumerate(tickers):
            df = pd.read_csv(self.file_from_ticker_(ticker))
            df.set_index('Date', inplace=True)

            dfs.append(df)

        # Combine per-ticker dataframes
        df_big = pd.concat( dfs, axis=1, keys=tickers)
        df_big.index.name = "Date"

        # Make the first level column index be attribute; the second will be ticker
        df_big = df_big.swaplevel(axis=1,i=0,j=1)

        # Always a good idea to sort index after concat

        df_big.sortlevel(axis=1, inplace=True)

        return df_big

    def save_data(self, df, file):
        with open(file,"wb") as fp:
            pickle.dump(df,fp)


    def load_data(self, file):
        with open(file,"rb") as fp:
            df = pickle.load(fp)

        return df

    ## The remainder of the file is the code from which this module was adapted.  Not to be used.

    def get_data_from_yahoo(self, reload_sp500=False):

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



    def compile_data(self):
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



    def compile_data_all(self, tickers=None):
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

    

