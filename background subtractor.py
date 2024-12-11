import pandas as pd

# Load your dataset
file_path = 'combined_data_formula_applied2.csv'  # Replace with your file path
df = pd.read_csv(file_path, header=None)  # Load without headers to preserve structure

# Initialize variables
wavelength_columns_start = 1  # Wavelength data starts from the second column
background = None
processed_rows = []
current_dataset_name = None

# Loop through the rows in the DataFrame
for index, row in df.iterrows():
    # Check if the row marks the start of a new dataset (non-numeric in the first column)
    if isinstance(row[0], str) and "ch3_lib" in row[0]:  # Adjust the string pattern if necessary
        current_dataset_name = row[0]  # Save the dataset name
        background = None  # Reset background for the new dataset
        processed_rows.append(row.values)  # Append the dataset name row
    elif pd.isna(row[0]):  # Skip blank rows
        processed_rows.append(row.values)
    else:
        # Check if the row has numeric data
        try:
            measurement_count = int(row[0])  # Convert the first column to int for Measurement Count
        except ValueError:
            continue  # Skip rows where the first column isn't numeric

        # Handle background subtraction
        if measurement_count == 1:
            # If Measurement Count is 1, store the background values (do not add to processed rows)
            background = row[wavelength_columns_start:].astype(float).values
        elif background is not None:
            # Subtract the background from the current row
            new_row = row.copy()
            new_row[wavelength_columns_start:] = row[wavelength_columns_start:].astype(float).values - background
            processed_rows.append(new_row.values)
        else:
            # If no background is available, just append the row unchanged
            processed_rows.append(row.values)

# Convert the processed rows back into a DataFrame
processed_df = pd.DataFrame(processed_rows)

# Save the result to a new file
output_path = 'subtracted_data1.csv'  # Replace with your desired output path
processed_df.to_csv(output_path, index=False, header=False)

print(f"Background subtraction completed. The modified data is saved as '{output_path}'.")
