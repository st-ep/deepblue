import numpy as np                                      # model arrays
import pandas as pd                                     # DataFrames
import matplotlib.pyplot as plt                         # building plots
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")  # set correct working directory
my_data = pd.read_csv("HackathonData2025+Fleet_Type_units_AT.csv")         # load the correct data file
my_data = my_data.iloc[:,1:]  

# Define missing_columns before using it
missing_columns = [
    'Estimated Average Stage Time',
    'Actual Average Stage Time',
    'Ambient Temperature'
]

# Check for missing values
print("\nMissing values in each column:")
print(my_data.isnull().sum())

'''
# Visual representation of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(my_data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
'''

# Get more detailed information about the dataset
print("\nDetailed information about the dataset:")
print(my_data.info())

# First, let's print the actual column names
print("\nActual column names in the dataset:")
print(my_data.columns.tolist())

# Identify categorical columns (note the space after 'Sand Provider')
categorical_columns = ['Frac Fleet', 'Fleet Type', 'Target Formation', 
                      'Field Area', 'Fuel Type', 'Sand Provider ']  # Added space here

# Create one-hot encoded variables
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, drop_first=True)

# Print the shape of the new dataset to see how many features we have after encoding
print("\nShape of dataset after one-hot encoding:", encoded_data.shape)

# Display first few columns of the encoded dataset
print("\nFirst few columns of the encoded dataset:")
print(encoded_data.head())

# Identify numeric columns (excluding the encoded categorical columns)
numeric_columns = ['Estimated Average Stage Time', 'Actual Average Stage Time', 
                  'Ambient Temperature', 'Grid', 'Diesel', 'CNG']

# Initialize the scaler
scaler = StandardScaler()

# Scale numeric columns
encoded_data[numeric_columns] = scaler.fit_transform(encoded_data[numeric_columns])

# Create a subset of data where Estimated Average Stage Time is not missing
complete_data = encoded_data.dropna(subset=['Estimated Average Stage Time'])

# Create a subset of data where Estimated Average Stage Time is missing
missing_data = encoded_data[encoded_data['Estimated Average Stage Time'].isna()]

# Select top important features (importance > 0.01)
important_features = [
    'Fleet Type_Zipper',
    'Diesel',
    'Target Formation_Pecan Tree',
    'Ambient Temperature',
    'CNG',
    'Target Formation_Lone Star',
    'Frac Fleet_Fleet 9',
    'Field Area_The Tower',
    'Target Formation_Longhorn',
    'Field Area_Gregory Gym'
]

# Create and train linear regression model
model = LinearRegression()

# Prepare the data with important features and remove NaN values
X = encoded_data[important_features]
y = encoded_data['Estimated Average Stage Time']
mask = ~y.isna()  # Create mask for non-NaN values
X = X[mask]
y = y[mask]

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
cv_rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))

print("\nCross-validation Results:")
print(f"Mean R² Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
print(f"Mean RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})")

# Train final model on full dataset
model.fit(X, y)

# Predict missing values
missing_mask = encoded_data['Estimated Average Stage Time'].isna()
X_missing = encoded_data[missing_mask][important_features]
predicted_values = model.predict(X_missing)

# Fill in the missing values
encoded_data.loc[missing_mask, 'Estimated Average Stage Time'] = predicted_values

# Print statistics about the predictions
print("\nNumber of missing values filled:", len(predicted_values))
print("Range of predicted values:", f"Min: {predicted_values.min():.2f}, Max: {predicted_values.max():.2f}")
print("Mean of predicted values:", f"{predicted_values.mean():.2f}")

# Print coefficients for interpretability
coefficients = pd.DataFrame({
    'Feature': important_features,
    'Coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients.sort_values(by='Coefficient', key=abs, ascending=False))
