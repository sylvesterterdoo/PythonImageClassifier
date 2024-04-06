import nasdaqdatalink
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

stock_name = 'GOOGL'

# curl "https://data.nasdaq.com/api/v3/datasets/WIKI/FB/data.json?api_key=L6jKoeBzyQtbURFhQCUz"
def Main(stock_name):
    nasdaqdatalink.read_key(filename='apikey')
    # usually highest date is 2018
    # Retrieve data from nasdaqdatalink using specified date range
    #data = nasdaqdatalink.get_table(f"WIKI/{stock_name}")

    # Print the retrieved data
    #data = nasdaqdatalink.get_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close']},
    #                                ticker=['AAPL'], date={'gte': '2016-01-01', 'lte': '2016-12-31'})

    data = nasdaqdatalink.get_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close']},
                                    ticker=['GOOGL'], date={'gte': '2016-01-01'})

    latest_date = data['date'][0]
    one_year_ago_date = latest_date - timedelta(weeks=52)

    filtered_df = data[data['date'] >= one_year_ago_date]

    print(filtered_df)
    #print(data['date'].head(10))

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['date'], filtered_df['close'], label='Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Adjusted Close Price for Last 52 Weeks')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()

    #print(data)
if __name__ == '__main__':
    Main(stock_name)