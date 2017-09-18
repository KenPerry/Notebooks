from zipline.api import order, record, symbol
from zipline import run_algorithm

import datetime


def initialize(context):
    context.randomStuff = 1


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data.current(symbol('AAPL'), 'price'))
    context.randomStuff += 1

start = datetime.date(2015, 1, 1)
end = datetime.date(2015, 12, 31)

# run_algorithm(start, end, initialize, 1000000, handle_data)

