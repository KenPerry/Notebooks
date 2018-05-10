import numpy as np
import pandas as pd

import datetime as dt
import dateutil.parser as dup
import datedelta

import os
import shutil

import pickle
import re

idx = pd.IndexSlice

class Date_Manipulator:
    def __init__(self, idx):
        self.idx = idx

        return

    # NOTE: periodic using ddelta=datedelta.datedelta(months=1) can be problematic since, e.g., there is no "31 st" day of every month
    def periodic(self, end_d, ddelta):
        """
        Returns list of dates, ending with end_d, at frequency of datedelta ddelta

        Parameters:
        end_d: datetime end date
        ddelta: datedelta, used to decrement from end_d

        Returns:
        list of dates
        """
        min_d = self.idx.min()

        result = [ end_d ]

        prev_d = end_d - ddelta

        # Generate dates that are successively earlier (by amount equal to datedelta ddelta) than prev_d
        while ( prev_d >= min_d ):
            result.insert(0, prev_d)
            prev_d = prev_d - ddelta

        return result

    def periodic_end_of_month(self, end_d):
        """
        Returns list of dates corresponding to the last day of the month, ending with the month containing end_d

        Parameters:
        end_d: datetime end date

        Returns:
        list of end-of-month dates
        """
        
        min_d = self.idx.min()
        
        # Find last day of month containing end_d
        last_d = end_d.replace(day=1) + datedelta.datedelta(months=1) - datedelta.datedelta(days=1)

        result = [ ]

        # Generate dates that are successively ends of preceding month
        while ( last_d >= min_d ):
            result.insert(0, last_d)
            last_d = last_d.replace(day=1) -datedelta.datedelta(days=1) 

        return result
        
        # Create periodic series, one month apart, ending on last_d
        eom_d = self.periodic( last_d, datedelta.datedelta(months=1) )

        return eom_d


    def in_index(self, list_d, meth="bfill"):
        """
        Returns a list of dates in self.idx that are closest to the dates in the list of dates list_d

        Parameters:
        list_d: list of date
        meth:   method for filling in missing dates in the index (see pd.Index.get_loc)

        Returns:
        list of dates, each of which is in self.idx
        """
        idx = self.idx


        # Make sure final element of list_d is in self.idx
        # Note: older version of pandas does not implement DateTimeIndex.contains, so convert to list and do as list operation
        pos = len(list_d) -1

        idx_l = idx.tolist()
        while (pos > 0 and not list_d[pos] in idx_l):
            pos = pos -1

        # For each date d, find the date in self.idx that is closest
        result = [ idx[ idx.get_loc(d, method=meth) ] for d in list_d[0:pos+1 ] ]

        return result

    def range_in_index(self, list_d):
        """
        Returns list of pairs (s,e). Each (s,e) pair defines a date range s <= d <= e

        Paramters:
        list_d: a list of dates, each of which IS in self.idx

        Returns:
        list of pairs of dates.  All dates are in self.idx
        """
        
        idx = self.idx

        result = []
        
        start = idx.min()
        for d in list_d:
            result.append( (start, d) )
            d_loc = idx.get_loc(d)
            start = idx[ d_loc + 1 ]
            
        return result


    def periodic_in_idx_end_of_month(self, end_d):
        eom_d = self.periodic_end_of_month(end_d)
        eom_in_idx = self.in_index( eom_d, meth="ffill" )

        return eom_in_idx
        



    
