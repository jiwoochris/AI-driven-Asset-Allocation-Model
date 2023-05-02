import pandas as pd

# Read the CSV file
folder_path = 'data/'
file_path = '10y-treasury.csv'
df = pd.read_csv(folder_path + file_path)

# Display the DataFrame
print("Original DataFrame:")
print(df)

import re

# Function to convert date format
def convert_date_format(date_string):

    # Define the regular expression pattern to match the date format
    pattern = r"(\d{2})/(\d{2})/(\d{4})"

    # Replace the slashes with dashes and reorder the date components
    formatted_date = re.sub(pattern, r"\3-\1-\2", date_string)

    return formatted_date

# Apply the conversion function to the 'Date' column
df['Date'] = df['Date'].apply(convert_date_format)





# Function to convert date format
def remove_commas(text_with_commas):

    if type(text_with_commas) == str:
        text_without_commas = text_with_commas.replace(",", "")
    else :
        return text_with_commas

    return text_without_commas

# Apply the conversion function to the 'Date' column
df['Price'] = df['Price'].apply(remove_commas)
# Convert column 'A' to integer
df['Price'] = df['Price'].astype(float)



df = df[['Date', 'Price']]

# Change the column name
df = df.rename(columns={'Price': 'Close'})


# Save the modified DataFrame to a new CSV file
df.to_csv(folder_path + "m_" + file_path, index=False)

# Display the modified DataFrame
print("\nModified DataFrame:")
print(df)




# import matplotlib.pyplot as plt

# # Set the 'Date' column as the DataFrame index
# df.set_index('Date', inplace=True)

# # Plot the data
# fig, ax = plt.subplots(figsize=(12, 8))

# plt.plot(df['Close'])


# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('Time Series Plot of Various Financial Indices')
# plt.legend(loc='best')

# plt.show()