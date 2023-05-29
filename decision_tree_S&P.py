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
csv_files = ["m_S&P500.csv", "m_Gold.csv", "m_WTI.csv", "m_USD.csv", "m_10y-treasury.csv", "m_3m-treasury.csv", "m_Kospi200.csv"]     # "m_KOSDAQ150.csv", 

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




X = result[['S&P500', 'Gold']]   # , '10y-treasury', , '경제심리지수', 'Kospi200', 
treasury = result[['10y-treasury', '3m-treasury']]
range_90 = result[['S&P500']]



X_lag_10 = X.shift(10)
X_diff_10 = (X - X_lag_10) / X_lag_10

X_lag_30 = X.shift(30)
X_diff_30 = (X - X_lag_30) / X_lag_30

X_lag_90 = X.shift(90)
X_diff_90 = (X - X_lag_90) / X_lag_90



# Suppose df is your DataFrame and 'price' is the column with the stock price
range_90 = (range_90 - range_90.rolling(window=90).min()) / (range_90.rolling(window=90).max() - range_90.rolling(window=90).min())


treasury_10_3 = result['10y-treasury'] - result['3m-treasury']
treasury_10_3.name = '10-3'


X_diff = pd.concat([X_diff_10, X_diff_30, X_diff_90, range_90, treasury], axis=1)
print("X_diff")
print(X_diff)



y_label_asset = 'S&P500'

y_diff_1 = result[y_label_asset].shift(-30) / result[y_label_asset] - 1
y_diff_2 = result[y_label_asset].shift(-25) / result[y_label_asset] - 1
y_diff_3 = result[y_label_asset].shift(-35) / result[y_label_asset] - 1

y_diff = (y_diff_1 + y_diff_2 + y_diff_3) / 3

print("Y_diff")
print(y_diff)




def classify_value(val):
    if val > 0.015:
        return 1
    elif -0.015 <= val <= 0.015:
        return 0
    elif val < -0.015:
        return -1
    else:
        return None  # to handle NaN values

# Assuming 'df' is your DataFrame and 'S&P500' is the column with the values
y_diff_clf = y_diff.apply(classify_value)



X = X_diff.iloc[90:-35].reset_index(drop=True)
y = y_diff_clf.iloc[90:-35].reset_index(drop=True)



print("X")
print(X)
print("y")
print(y)




# import seaborn as sns
# sns.distplot(y, bins=30, kde=True)
# plt.show()




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)



from sklearn import tree
# 의사결정나무 적합 및 학습데이터 예측
clf = tree.DecisionTreeClassifier(max_depth = 15)
clf = clf.fit(X_train, y_train)

print("X.columns")
print(X.columns)

predict = clf.predict(X_test)

print("predict")
print(predict)



# # visualize tree
# tree.plot_tree(clf)
# plt.show()




from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))





# Create the figure and axis objects
fig2, ax1 = plt.subplots()

# Plot the first dataset with the first y-axis
color = 'tab:red'
ax1.set_ylabel('Prediction', color=color)
ax1.plot(y_pred, color=color)
ax1.tick_params(axis='y', labelcolor=color)


# Create a second y-axis for the second dataset
ax2 = ax1.twinx()

# Plot the second dataset with the second y-axis
color = 'tab:blue'
ax2.set_ylabel('Real', color=color)
ax2.plot(list(y_diff.iloc[90:-35])[-len(y_pred) :], color=color)
ax2.tick_params(axis='y', labelcolor=color)


# Add a title and legend
plt.title('Two related datasets with different volatility')

# plt.show()








now = X_diff.iloc[-35:].reset_index(drop=True)
y_pred_now = clf.predict(now)

print(y_pred_now)

