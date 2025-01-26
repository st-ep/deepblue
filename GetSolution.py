import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV with bootstrap predictions
df = pd.read_csv("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue/testing.csv")
df_test = pd.read_csv("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue/test_main_predictions.csv")

# Copy the well names based on index
df_test['Masked Well Name'] = df['Well Name'].values

# Print the results to verify
print("Original testing.csv head:")
print(df.head())
print("\nRecovered test_main_predictions.csv head:")
print(df_test.head())

df_solution = pd.read_csv("solution.csv")

# Create a mapping dictionary for fuel type to column name
fuel_type_mapping = {
    'Grid': 'Grid',
    'DGB_CNG': 'CNG',
    'Turbine': 'CNG',
    'DGB_Diesel': 'Diesel',
    'Diesel': 'Diesel'
}

# Loop through each row in df_solution
for index, row in df_solution.iterrows():
    well_name = row['Masked Well Name']
    # Use fuel type from df_solution instead of df_test
    fuel_type = row['Fuel Type']
    # Find matching row in df_test
    matching_row = df_test[df_test['Masked Well Name'] == well_name]
    
    if not matching_row.empty:
        if fuel_type in fuel_type_mapping:
            column_to_use = fuel_type_mapping[fuel_type]
            df_solution.at[index, 'Fuel Value'] = matching_row[column_to_use].iloc[0]

# Print results to verify
print("Updated solution.csv head:")
print(df_solution.head())

df_uncert_CD = pd.read_csv("test_uncert_predictions_CD2.csv")
# Copy the well names to df_uncert the same way as df_test
df_uncert_CD['Masked Well Name'] = df['Well Name'].values

# Initialize counter for modified rows
rows_modified = 0

# Loop through each row in df_solution for uncertainty values
for index, row in df_solution.iterrows():
    well_name = row['Masked Well Name']
    # Use fuel type from df_solution instead of df_test
    fuel_type = row['Fuel Type']
    # Find matching row in df_uncert for values
    matching_row_uncert = df_uncert_CD[df_uncert_CD['Masked Well Name'] == well_name]
    
    if not matching_row_uncert.empty:
        # Determine which letter prefix to use based on fuel type
        prefix_mapping = {
            'Grid': 'G',
            'DGB_CNG': 'C',
            'Turbine': 'C',
            'DGB_Diesel': 'D',
            'Diesel': 'D'
        }
        
        if fuel_type in prefix_mapping:
            prefix = prefix_mapping[fuel_type]
            row_modified = False
            # Update R1 through R100
            for i in range(1, 101):
                source_col = f'{prefix}{i}'
                target_col = f'R_{i}'
                # Only update if the source column exists
                if source_col in matching_row_uncert.columns:
                    df_solution.at[index, target_col] = matching_row_uncert[source_col].iloc[0]
                    row_modified = True
            if row_modified:
                rows_modified += 1

print(f"Number of rows modified: {rows_modified}")
print("Updated solution.csv with uncertainty values:")
print(df_solution.head())

df_uncert_G = pd.read_csv("test_uncert_predictions_G3.csv")
# Copy the well names to df_uncert the same way as df_test
df_uncert_G['Masked Well Name'] = df['Well Name'].values

# Initialize counter for modified rows
rows_modified = 0

# Loop through each row in df_solution for uncertainty values
for index, row in df_solution.iterrows():
    well_name = row['Masked Well Name']
    # Use fuel type from df_solution instead of df_test
    fuel_type = row['Fuel Type']
    # Find matching row in df_uncert for values
    matching_row_uncert = df_uncert_G[df_uncert_G['Masked Well Name'] == well_name]
    
    if not matching_row_uncert.empty:
        # Determine which letter prefix to use based on fuel type
        prefix_mapping = {
            'Grid': 'G',
            'DGB_CNG': 'C',
            'Turbine': 'C',
            'DGB_Diesel': 'D',
            'Diesel': 'D'
        }
        
        if fuel_type in prefix_mapping:
            prefix = prefix_mapping[fuel_type]
            row_modified = False
            # Update R1 through R100
            for i in range(1, 101):
                source_col = f'{prefix}{i}'
                target_col = f'R_{i}'
                # Only update if the source column exists
                if source_col in matching_row_uncert.columns:
                    df_solution.at[index, target_col] = matching_row_uncert[source_col].iloc[0]
                    row_modified = True
            if row_modified:
                rows_modified += 1

print(f"Number of rows modified: {rows_modified}")
print("Updated solution.csv with uncertainty values:")
print(df_solution.head())

# Print all values in column R7
print("\nAll values in column R7:")
print(df_solution['R_7'].to_string())

# After both CD and G updates, check for NaN values
nan_wells = df_solution[df_solution['R_7'].isna()]['Masked Well Name']
print("\nWells with NaN values in R7:")
print(nan_wells)

# Also print their fuel types
nan_wells_info = df_test[df_test['Masked Well Name'].isin(nan_wells)][['Masked Well Name', 'Fuel Type']]
print("\nFuel types for wells with NaN values:")
print(nan_wells_info)

# Print unique fuel types in df_solution
print("\nUnique fuel types in df_solution:")
print(df_solution['Fuel Type'].unique())

# Print sample rows with their fuel types
print("\nSample rows from df_solution with fuel types:")
print(df_solution[['Masked Well Name', 'Fuel Type']].head(10))

# Before updates
print("\nColumns in df_solution before updates:")
print(df_solution.columns.tolist())

# After first update with CD
print("\nColumns in df_solution after CD update:")
print(df_solution.columns.tolist())

# After second update with G
print("\nColumns in df_solution after G update:")
print(df_solution.columns.tolist())

# Save the final solution to CSV
df_solution.to_csv("scaled_solution.csv", index=False)
print("\nFinal solution saved to 'final_solution.csv'")

