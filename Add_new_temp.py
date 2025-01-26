import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV with bootstrap predictions
df_temp = pd.read_csv("testing_with_filled_temperatures.csv")
df_test = pd.read_csv("TrainData_recovered_forest_main.csv")

# Print the first few rows of the dataframe
print(df_temp.head())
print(df_test.head())

# Add the new temperature column to the training data
df_test['Ambient Temperature'] = df_temp['Ambient Temperature']

# Save the updated training data
df_test.to_csv('TrainData_recovered_forest_main_with_temp.csv', index=False)
