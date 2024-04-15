import pandas as pd

# Paths to the input CSV files
csv_file_path1 = 'steel+plates+faults/output.csv'
csv_file_path2 = 'data/train.csv'

# Path to the output CSV file
output_csv_file_path = 'data/new_train.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2])

# Drop duplicate IDs (assuming the ID column is named 'ID')
# Keep the first occurrence and discard the subsequent ones
combined_df_unique = combined_df.drop_duplicates(subset='id', keep='first')

# Write the resulting DataFrame to a new CSV file
combined_df_unique.to_csv(output_csv_file_path, index=False)

print(f"Combined CSV file saved to {output_csv_file_path}")
