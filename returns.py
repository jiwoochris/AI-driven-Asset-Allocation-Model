import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_folder = "cleaned_data/"
csv_files = ["m_S&P500.csv", "m_Gold.csv", "m_Kospi200.csv", "m_USD.csv", "m_WTI.csv", "m_K_treasury.csv", "m_K_corp_bond.csv", "m_global_bonds.csv"]     # "m_KOSDAQ150.csv", 

# csv_files = ["m_Kospi200.csv", "KORLOLITONOSTSAM.csv"] "m_EuroStoxx50.csv", 

asset_list = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:

    print(file)

    # Read file
    df = pd.read_csv(data_folder+file)

    # Ensure the data is sorted by date
    df = df.sort_values('Date')

    # Calculate daily returns
    daily_returns = df['Close'].pct_change()

    # Calculate the average daily return
    average_daily_return = daily_returns.mean()

    # Annualize the average daily return
    average_annual_return = average_daily_return * 252  # 252 trading days in a year

    print(f"The expected return is: {average_annual_return * 100 : .3f}%")


    # Calculate the standard deviation of daily return
    daily_std = daily_returns.std()

    # Annualize the standard deviation
    annualized_std = daily_std * (252**0.5)  # 252 trading days in a year

    print(f"The risk (annualized volatility) is: {annualized_std * 100 : .3f}%")

    # Assuming a risk-free rate of 2% per year
    risk_free_rate = 0.02
    
    # Calculate the Sharpe ratio
    sharpe_ratio = (average_annual_return - risk_free_rate) / annualized_std

    print(f"The Sharpe ratio is: {sharpe_ratio : .3f}")




# Calculate covariance of returns
cov_matrix = returns.cov()

# Assume we have some portfolio weights
weights = np.array([0.5, 0.5])  # Replace this with your actual portfolio weights

# Calculate portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# Calculate portfolio volatility (standard deviation)
port_volatility = np.sqrt(port_variance)

print(f"The portfolio's risk (volatility) is: {port_volatility}")





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


plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Plot of Various Financial Indices')
plt.legend(loc='best')

plt.show()








# Calculate daily returns
daily_returns = result.pct_change()

# Calculate the volatility (standard deviation of daily returns)
volatility = daily_returns.std()

print("\nVolatility:\n", volatility)

