import pandas as pd

# Read the CSV file
folder_path = 'data/'
file_path = 'S&P500.csv'
df = pd.read_csv(folder_path + file_path)

# Display the DataFrame
print("Original DataFrame:")
print(df)

import re

# Function to convert date format
def convert_date_format(date_string):
    pattern = r"(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}).*"
    match = re.match(pattern, date_string)

    if match:
        date_part = match.group(1)
        formatted_date = date_part.replace(".", "-").replace(" ", "")
        return formatted_date
    else:
        return date_string


# Apply the conversion function to the 'Date' column
df['Date'] = df['Date'].apply(convert_date_format)



# Save the modified DataFrame to a new CSV file
df.to_csv("cleaned_data/" + "m_" + file_path, index=False)

# Display the modified DataFrame
print("\nModified DataFrame:")
print(df)
