import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_folder = "cleaned_data/"
csv_files = ["m_S&P500.csv", "m_EuroStoxx50.csv", "m_Gold.csv", "m_Kospi200.csv", "m_USD.csv", "m_WTI.csv", "m_K_treasury.csv", "m_K_corp_bond.csv", "m_global_bonds.csv"]     # "m_KOSDAQ150.csv", 

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
    start_date = '2014-03-24'
    end_date = '2022-12-29'
    selected_rows = sorted_df[(sorted_df['Date'] >= start_date) & (sorted_df['Date'] <= end_date)]

    print("\nSelected rows for the date range:")
    print(selected_rows)




    date_range = pd.date_range(start_date, end_date, freq='D')
    date_range_df = pd.DataFrame({'Date': date_range})

    # Perform a left join to fill missing dates with None (or NaN for numerical columns)
    fill_empty_rows = pd.merge(date_range_df, selected_rows, on='Date', how='left')

    # Replace NaN values with None for non-numerical columns if needed
    fill_empty_rows.loc[fill_empty_rows['Close'].isna(), 'Close'] = np.nan

    replace_na_rows = fill_empty_rows.fillna(method='ffill')

    # Change the column name
    column_name_rows = replace_na_rows.rename(columns={'Close': file[2:-4]})

    # print(column_name_rows)

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
print(normalized_df)

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))

for asset in normalized_df.columns:
    plt.plot(normalized_df[asset], label=asset)


plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Plot of Various Financial Indices')
plt.legend(loc='best')

plt.show()