import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error

data_folder = "cleaned_data/"
csv_files = ["m_S&P500.csv", "m_china.csv", "m_Kosdaq.csv", "m_Kospi.csv", "m_Nikkei.csv", "m_europe.csv", "m_Gold.csv"]     # "m_KOSDAQ150.csv", 

asset_list = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:

    # Read file
    df = pd.read_csv(data_folder+file)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by the 'Date' column
    sorted_df = df.sort_values(by='Date')

    # Select a date range
    start_date = '2000-01-18'
    end_date = '2023-05-26'


    date_range = pd.date_range(start_date, end_date, freq='D')
    date_range_df = pd.DataFrame({'Date': date_range})

    # Perform a left join to fill missing dates with None (or NaN for numerical columns)
    fill_empty_rows = pd.merge(date_range_df, sorted_df, on='Date', how='left')

    # Replace NaN values with None for non-numerical columns if needed
    fill_empty_rows.loc[fill_empty_rows['Close'].isna(), 'Close'] = np.nan

    replace_na_rows = fill_empty_rows.fillna(method='ffill')
    replace_na_rows = fill_empty_rows.fillna(method='bfill')
    replace_na_rows = fill_empty_rows.fillna(method='ffill')
    replace_na_rows = fill_empty_rows.fillna(method='bfill')

    # Change the column name
    column_name_rows = replace_na_rows.rename(columns={'Close': file[2:-4]})

    # print(column_name_rows)


    selected_rows = column_name_rows[(column_name_rows['Date'] >= start_date) & (column_name_rows['Date'] <= end_date)]

    # print("\nSelected rows for the date range:")
    # print(selected_rows)

    asset_list.append(column_name_rows)




result = asset_list[0]

for df in asset_list[1:]:
    result = pd.merge(result, df, on='Date', how='inner')



print(result)


range_365 = (result - result.rolling(window=365).min()) / (result.rolling(window=365).max() - result.rolling(window=365).min())

print(range_365.iloc[-1])



# Assuming df is your DataFrame
result['Date'] = pd.to_datetime(result['Date'])
result = result.set_index('Date')

# Calculate 1-month returns
return_30 = result.pct_change(periods=30)
return_60 = result.pct_change(periods=60)
return_90 = result.pct_change(periods=90)

print((return_30.iloc[-1] + return_60.iloc[-1] + return_90.iloc[-1]) / 3)