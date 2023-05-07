import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_folder = "cleaned_data/"
csv_files = ["m_S&P500.csv", "m_EuroStoxx50.csv", "m_Gold.csv", "m_Kospi200.csv", "m_USD.csv", "m_WTI.csv", "m_K_treasury.csv", "m_K_corp_bond.csv", "m_global_bonds.csv", "KORLOLITONOSTSAM.csv"]     # "m_KOSDAQ150.csv", 

csv_files = ["m_Kospi200.csv", "KORLOLITONOSTSAM.csv"]

asset_list = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:

    print(file)

    # Read file
    df = pd.read_csv(data_folder+file)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by the 'Date' column
    sorted_df = df.sort_values(by='Date')

    # Select a date range
    start_date = '2001-01-01'
    end_date = '2022-12-29'


    date_range = pd.date_range(start_date, end_date, freq='D')
    date_range_df = pd.DataFrame({'Date': date_range})

    # Perform a left join to fill missing dates with None (or NaN for numerical columns)
    fill_empty_rows = pd.merge(date_range_df, sorted_df, on='Date', how='left')

    # Replace NaN values with None for non-numerical columns if needed
    fill_empty_rows.loc[fill_empty_rows['Close'].isna(), 'Close'] = np.nan

    replace_na_rows = fill_empty_rows.fillna(method='ffill')
    replace_na_rows = fill_empty_rows.fillna(method='bfill')
    replace_na_rows = fill_empty_rows.fillna(method='ffill')

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




# Compute the correlation between the columns
correlation = result.corr()

# Display the correlation results
print("Correlation between columns:")
print(correlation)

# Save the correlation matrix as an Excel file
correlation.to_excel('correlation_matrix.xlsx')





# Set the 'Date' column as the index
result.set_index('Date', inplace=True)

normalized_df = result.divide(result.iloc[0])
# print(normalized_df)

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))

for asset in normalized_df.columns:
    plt.plot(normalized_df[asset], label=asset)


# Create the figure and axis objects
fig2, ax1 = plt.subplots()

# Plot the first dataset with the first y-axis
color = 'tab:red'
ax1.set_ylabel('Prediction', color=color)
ax1.plot(result['Kospi200'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
# ax1.axhline(y=0, color=color, linestyle='--')

# Create a second y-axis for the second dataset
ax2 = ax1.twinx()

# Plot the second dataset with the second y-axis
color = 'tab:blue'
ax2.set_ylabel('Real', color=color)
ax2.plot(result['RLOLITONOSTSAM'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
# ax2.axhline(y=0, color=color, linestyle='--')

# Add a title and legend
plt.title('Two related datasets with different volatility')

plt.show()







# Calculate daily returns
daily_returns = result.pct_change()

# Calculate the volatility (standard deviation of daily returns)
volatility = daily_returns.std()

print("\nVolatility:\n", volatility)





max_lag = 360  # Adjust this to your desired maximum lag
max_corr = 0
optimal_lag = 0


for lag in range(1, max_lag + 1):

    def from_1h_to_1w(data) :
        data = data[::7]
        return data

    shifted_revenue = result['RLOLITONOSTSAM'].iloc[:-lag]

    shifted_stock_price = result['Kospi200'].iloc[lag:]

    corr = np.corrcoef(from_1h_to_1w(shifted_stock_price), from_1h_to_1w(shifted_revenue))[0, 1]

    print(corr)
    
    if abs(corr) > abs(max_corr):
        max_corr = corr
        optimal_lag = lag

print(f"Optimal lag between stock price and revenue: {optimal_lag} days")
print(f"Correlation at optimal lag: {max_corr}")