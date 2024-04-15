import pandas as pd

# Global variables to set at the beginning of the file
NNA_FILE_PATH = 'Faults.NNA'
COLUMN_NAMES_FILE_PATH = 'Faults27x7_var'
CSV_FILE_PATH = 'output.csv'
START_ID = 0

def parse_nna_to_csv():
    # Read column names from the additional file
    with open(COLUMN_NAMES_FILE_PATH, 'r') as file:
        column_names = [line.strip() for line in file.readlines()]
    
    # Assuming the NNA file is space-delimited, read it into a DataFrame
    # If the delimiter is different, adjust the 'sep' parameter accordingly
    df = pd.read_csv(NNA_FILE_PATH, sep='\s+', header=None, names=column_names)
    
    # Add an ID column starting from `START_ID`
    df.insert(0, 'id', range(START_ID, START_ID + len(df)))
    
    # Write the DataFrame to a CSV file
    df.to_csv(CSV_FILE_PATH, index=False)

if __name__ == '__main__':
    parse_nna_to_csv()