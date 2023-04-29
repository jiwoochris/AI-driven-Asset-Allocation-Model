import pandas as pd

# Read the CSV file
folder_path = 'data/'
file_path = 'USD.csv'
df = pd.read_csv(folder_path + file_path)

# Display the DataFrame
print("Original DataFrame:")
print(df)

import re

# Function to convert date format
def convert_date_format(date_string):

    # Define the regular expression pattern to match the date format
    pattern = r"(\d{4})-\s*(\d{2})-\s*(\d{2})"

    # Remove the spaces and keep the dashes between the date components
    formatted_date = re.sub(pattern, r"\1-\2-\3", date_string)

    return formatted_date

# Apply the conversion function to the '날짜' column
df['날짜'] = df['날짜'].apply(convert_date_format)



# Function to convert date format
def remove_commas(text_with_commas):

    text_without_commas = text_with_commas.replace(",", "")

    return text_without_commas

# Apply the conversion function to the 'Date' column
df['종가'] = df['종가'].apply(remove_commas)
# Convert column 'A' to integer
df['종가'] = df['종가'].astype(float)



df = df[['날짜', '종가']]

# Change the column name
df = df.rename(columns={'날짜': 'Date', '종가': 'Close'})


# Save the modified DataFrame to a new CSV file
df.to_csv(folder_path + "m_" + file_path, index=False)

# Display the modified DataFrame
print("\nModified DataFrame:")
print(df)
