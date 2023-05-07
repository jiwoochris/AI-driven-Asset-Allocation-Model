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
csv_files = ["m_S&P500.csv", "m_Gold.csv", "m_WTI.csv", "m_USD.csv", "m_10y-treasury.csv", "m_경제심리지수.csv", "m_Kospi200.csv"]     # "m_KOSDAQ150.csv", 

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
    start_date = '2005-02-01'
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

print("data")
print(result)




X = result[['S&P500', 'Gold', 'WTI', 'USD', '10y-treasury']]   # , '10y-treasury', '경제심리지수'
treasury = result[[]]

X_lag_10 = X.shift(10)
X_diff_10 = (X - X_lag_10) / X_lag_10

X_lag_20 = X.shift(20)
X_diff_20 = (X - X_lag_20) / X_lag_20

X_lag_30 = X.shift(30)
X_diff_30 = (X - X_lag_30) / X_lag_30

X_lag_60 = X.shift(60)
X_diff_60 = (X - X_lag_60) / X_lag_60

X_lag_90 = X.shift(90)
X_diff_90 = (X - X_lag_90) / X_lag_90


X_diff = pd.concat([X_diff_10, X_diff_20, X_diff_30, X_diff_60, X_diff_90, treasury], axis=1)
print("X_diff")
print(X_diff)


y_label_asset = 'S&P500'

y_diff_1 = result[y_label_asset].shift(-30) / result[y_label_asset] - 1
y_diff_2 = result[y_label_asset].shift(-25) / result[y_label_asset] - 1
y_diff_3 = result[y_label_asset].shift(-35) / result[y_label_asset] - 1

y_diff = (y_diff_1 + y_diff_2 + y_diff_3) / 3

print("Y_diff")
print(y_diff)



X = X_diff.iloc[90:-35].reset_index(drop=True)
y = y_diff.iloc[90:-35].reset_index(drop=True)

print("X")
print(X)
print("y")
print(y)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)


from sklearn.metrics.pairwise import cosine_similarity
def get_regression_by_similar_pattern(X_train, y_train, pattern, similarity_threshold=0.95):

    # Calculate cosine similarity between the pattern and all training data
    similarities = cosine_similarity(X_train, pattern.to_numpy().reshape(-1, 25))

    # Get the indices of training data with cosine similarity greater than the specified threshold
    similar_indices = np.where(similarities >= similarity_threshold)[0]

    # Check if similar_indices is empty
    if similar_indices.size == 0 :
        # # If empty, use top n similarities
        # similar_indices = np.argsort(similarities, axis=0)[-top_n:].flatten()
        predicted_outputs = [0]

    else:
        # Get the corresponding predicted outputs
        predicted_outputs = y_train[similar_indices]

    # Calculate the average predicted output
    average_predicted_output = np.mean(predicted_outputs)

    return average_predicted_output


# Create an empty list to store the predicted outputs
predicted_outputs = []
# Loop through the validation data (or any other data you'd like to predict)
for pattern in X_test.iterrows():

    # Get the predicted output for the current pattern
    predicted_output = get_regression_by_similar_pattern(X_train, y_train, pattern[1], similarity_threshold=0.95)     # , n=sim_n
    predicted_outputs.append(predicted_output)

print(predicted_outputs)


from sklearn.metrics import mean_absolute_error

# Evaluation metric: accuracy
def evaluate_accuracy(y_true, y_pred):
    y_true_labels = (y_true >= 0).astype(int)
    y_pred_labels = (y_pred >= 0).astype(int)
    return np.mean(y_true_labels == y_pred_labels)

# Assuming y_true and y_pred are your true and predicted values respectively
mae = mean_absolute_error(y_test, predicted_outputs)
print('Mean Absolute Error:', mae)

accuracy = evaluate_accuracy(y_test, np.asarray(predicted_outputs))
print("Accuracy : ", accuracy)










fig, ax = plt.subplots(figsize=(12, 8))

plt.plot(predicted_outputs, y_test)

# Draw a line at x=0 and y=0
plt.axhline(y=0, color='gray', linestyle='--')
plt.axvline(x=0, color='gray', linestyle='--')

plt.xlabel('Prediction')
plt.ylabel('Real')
plt.title('Prediction vs Real')


# Create the figure and axis objects
fig2, ax1 = plt.subplots()

# Plot the first dataset with the first y-axis
color = 'tab:red'
ax1.set_ylabel('Prediction', color=color)
ax1.plot(predicted_outputs, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(y=0, color=color, linestyle='--')

# Create a second y-axis for the second dataset
ax2 = ax1.twinx()

# Plot the second dataset with the second y-axis
color = 'tab:blue'
ax2.set_ylabel('Real', color=color)
ax2.plot(list(y_test), color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=0, color=color, linestyle='--')

# Add a title and legend
plt.title('Two related datasets with different volatility')

plt.show()