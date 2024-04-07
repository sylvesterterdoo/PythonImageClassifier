#import nasdaqdatalink
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import requests


stock_name = 'GOOGL'
BASE_URL = "https://api.coingecko.com/api/v3/coins/"

# curl "https://data.nasdaq.com/api/v3/datasets/WIKI/FB/data.json?api_key=L6jKoeBzyQtbURFhQCUz"
def Main(stock_name):
    result = requests.get(BASE_URL)


#def first_question():
#    nasdaqdatalink.read_key(filename='apikey')
#    # usually highest date is 2018
#    # Retrieve data from nasdaqdatalink using specified date range
#    #data = nasdaqdatalink.get_table(f"WIKI/{stock_name}")
#
#    # Print the retrieved data
#    #data = nasdaqdatalink.get_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close']},
#    #                                ticker=['AAPL'], date={'gte': '2016-01-01', 'lte': '2016-12-31'})
#
#    data = nasdaqdatalink.get_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close']},
#                                    ticker=['GOOGL'], date={'gte': '2016-01-01'})
#
#    latest_date = data['date'][0]
#    one_year_ago_date = latest_date - timedelta(weeks=52)
#
#    filtered_df = data[data['date'] >= one_year_ago_date]
#
#   # plt.figure(figsize=(10, 6))
#   # plt.plot(filtered_df['date'], filtered_df['close'], label='Adjusted Close Price')
#   # plt.xlabel('Date')
#   # plt.ylabel('Adjusted Close Price')
#   # plt.title('Adjusted Close Price for Last 52 Weeks')
#   # plt.xticks(rotation=45)
#   # plt.legend()
#   # plt.grid(True)
#   # plt.show()
#
#    # print the min and max during that timeframe
#    # print the day when it traded the highest and lowest
#    max_costs = filtered_df[['date', 'close']].max()
#    min_costs = filtered_df[['date', 'close']].min()
#
#    print(max_costs)
#    print(min_costs)

if __name__ == '__main__':
    st.text_input("Stock name", key="stock")
    stock_name = st.session_state.stock.upper()
    # verify that the stock is correct here or handle the error when it comes empty after your request
    # or crashes
    print(stock_name)
    Main(stock_name)


    