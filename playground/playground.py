import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('TESTING: ')

df = pd.read_csv('playground/bitstamp_07-09.csv')
# print(df.head())
# print(df['price_close'].max())
# df['price_close'].plot()
# plt.show()

# creates an inner join, only what's common between both dataframes
# df1.join(df2, how='inner')

for symbol in symbols:
    df_temp = pd.read_csv('data/{}.csv'.format(symbol), index_col='Date',
                          parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
    # rename to prevent clash
    df_temp = df_temp.rename(columns={'Close': symbol})
    df = df1.join(df_temp)
