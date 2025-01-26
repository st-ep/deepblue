import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV with bootstrap predictions
df = pd.read_csv("scaled_solution.csv")

# Print numeric columns and their datatypes
numeric_columns = df.select_dtypes(include=[np.number]).columns
print("\nNumeric columns and their datatypes:")
for col in numeric_columns:
    print(f"{col}: {df[col].dtype}")

# Scale values based on Fuel Type
for index, row in df.iterrows():
    if row['Fuel Type'] == 'Grid':
        df.loc[index, numeric_columns] = row[numeric_columns] * (1000000/3412)
    elif row['Fuel Type'] in ['DGB_Diesel', 'Diesel']:
        df.loc[index, numeric_columns] = row[numeric_columns] * (1000000/(40.67 * 3412))

# Save the scaled results
df.to_csv("solution_1.csv", index=False)
