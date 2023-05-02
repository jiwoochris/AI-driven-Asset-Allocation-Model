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
csv_files = ["m_S&P500.csv", "m_Gold.csv", "m_WTI.csv", "m_10y-treasury.csv"]     # "m_KOSDAQ150.csv", 

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




X = result[['S&P500', 'Gold', 'WTI']]   # , '10y-treasury'
treasury = result[['10y-treasury']]

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


y_diff_1 = result['Gold'].shift(-30) / result['Gold'] - 1
y_diff_2 = result['Gold'].shift(-25) / result['Gold'] - 1
y_diff_3 = result['Gold'].shift(-35) / result['Gold'] - 1

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)


# from sklearn.metrics.pairwise import cosine_similarity
# def get_regression_by_similar_pattern(X_train, y_train, pattern, similarity_threshold=0.95):

#     # Calculate cosine similarity between the pattern and all training data
#     similarities = cosine_similarity(X_train, pattern.to_numpy().reshape(-1, 3))

#     # Get the indices of training data with cosine similarity greater than the specified threshold
#     similar_indices = np.where(similarities >= similarity_threshold)[0]

#     # Check if similar_indices is empty
#     if similar_indices.size == 0 :
#         # # If empty, use top n similarities
#         # similar_indices = np.argsort(similarities, axis=0)[-top_n:].flatten()
#         predicted_outputs = [0]

#     else:
#         # Get the corresponding predicted outputs
#         predicted_outputs = y_train[similar_indices]

#     # Calculate the average predicted output
#     average_predicted_output = np.mean(predicted_outputs)

#     return average_predicted_output


# # Create an empty list to store the predicted outputs
# predicted_outputs = []
# # Loop through the validation data (or any other data you'd like to predict)
# for pattern in X_test.iterrows():

#     # Get the predicted output for the current pattern
#     predicted_output = get_regression_by_similar_pattern(X_train, y_train, pattern[1], similarity_threshold=0.90)     # , n=sim_n
#     predicted_outputs.append(predicted_output)

# print(predicted_outputs)


# from sklearn.metrics import mean_absolute_error
# # Assuming y_true and y_pred are your true and predicted values respectively
# mae = mean_absolute_error(y_test, predicted_outputs)
# print('Mean Absolute Error:', mae)







# Convert the training and testing data to PyTorch tensors
X_train_t = torch.tensor(X_train.values.astype(np.float32))
y_train_t = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
X_test_t = torch.tensor(X_test.values.astype(np.float32))
y_test_t = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train_t.shape[1], 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = Net()


# define the loss function with L1 regularization
criterion = nn.MSELoss()
regularization_strength = 0.01
l1_regularizer = nn.L1Loss(size_average=False)
loss_fn = lambda output, target: criterion(output, target) + regularization_strength * l1_regularizer(model.fc1.weight)


# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)


# Train the model on the training data
tolerance = 30
best_mse = float('inf')
no_improvement_count = 0

from torch.utils.data import DataLoader, TensorDataset


batch_size = 16
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

num_epochs = 500

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0) # multiply by batch size

    # Evaluate the model on the testing data
    y_pred_t = model(X_test_t)
    mse = mean_squared_error(y_pred_t.detach().numpy(), y_test_t.detach().numpy())
    print('Epoch [%d], Train Loss: %.8f, Test Loss: %.8f' % (epoch+1, running_loss/X_train_t.shape[0], mse))

    # Check for early stopping
    if mse < best_mse:
        best_mse = mse
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= tolerance:
            print('Early stopping after epoch %d' % epoch)
            break

# Evaluate the model on the testing data
y_pred_t = model(X_test_t)


print('X_test :\n', X_test)
print('y_test :\n', y_test)
print('y_pred_t :\n', y_pred_t)




# Evaluation metric: accuracy
def evaluate_accuracy(y_true, y_pred):
    y_true_labels = (y_true.detach().numpy() >= 0)
    y_pred_labels = (y_pred.detach().numpy() >= 0)
    return np.mean(y_true_labels == y_pred_labels)


criterion = nn.L1Loss()     # mean absolute error
mae = criterion(y_pred_t, y_test_t)
print('Mean Absolute Error:', mae.item())

accuracy = evaluate_accuracy(y_test_t, y_pred_t)
print("Accuracy : ", accuracy)