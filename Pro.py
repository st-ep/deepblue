import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# ----------------------------
# 1. Load Data
# ----------------------------
os.chdir("/Users/Step_by-stepa/Documents/PGE_HACK/deepblue")
my_data = pd.read_csv("HackathonData2025+Fleet_Type_units_AT.csv")
my_data = my_data.iloc[:, 1:]  # if first col is an index

# Check missing
print("\nMissing values in each column:")
print(my_data.isnull().sum())

print("\nDataset info:")
print(my_data.info())

print("\nActual column names:")
print(my_data.columns.tolist())

# ----------------------------
# 2. Basic Feature Engineering
# ----------------------------
my_data.columns = my_data.columns.str.strip()  # fix trailing spaces

categorical_columns = ['Frac Fleet', 'Fleet Type', 'Target Formation',
                       'Field Area', 'Fuel Type', 'Sand Provider']

# Create total energy feature
my_data['EnergySum'] = my_data['Grid'] + my_data['Diesel'] + my_data['CNG']

# One-hot encode
encoded_data = pd.get_dummies(my_data, columns=categorical_columns, drop_first=True)

# Scale EnergySum
scaler = StandardScaler()
encoded_data['EnergySum_scaled'] = scaler.fit_transform(encoded_data[['EnergySum']])
encoded_data.drop('EnergySum', axis=1, inplace=True)

# ----------------------------
# 3. Prepare Data (Known vs. Missing Target)
# ----------------------------
mask_not_missing = ~encoded_data['Estimated Average Stage Time'].isna()
mask_missing = ~mask_not_missing

selected_categorical = ['Fleet Type', 'Target Formation', 'Fuel Type']
encoded_columns = [
    col for col in encoded_data.columns
    if any(cat in col for cat in selected_categorical) or col == 'EnergySum_scaled'
]

X_full = encoded_data.loc[mask_not_missing, encoded_columns]
y_full = encoded_data.loc[mask_not_missing, 'Estimated Average Stage Time']
X_missing = encoded_data.loc[mask_missing, encoded_columns]

print("\nTraining data shape:", X_full.shape)
print("Training target shape:", y_full.shape)
print("Data with missing values shape:", X_missing.shape)

# ----------------------------
# 4. Check Feature Importance (Optional)
# ----------------------------
temp_rf = RandomForestRegressor(n_estimators=100, random_state=42)
temp_rf.fit(X_full, y_full)
importances = pd.DataFrame({
    'feature': X_full.columns,
    'importance': temp_rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 features (RandomForest):")
print(importances.head(10))

# Pick top features from your analysis
top_features = ['Fleet Type_Zipper', 'EnergySum_scaled', 'Target Formation_Pecan Tree']

# ----------------------------
# 5. Additional Feature Engineering
# ----------------------------
X_sub = X_full[top_features].copy()
X_sub['Fleet_Energy'] = X_sub['Fleet Type_Zipper'] * X_sub['EnergySum_scaled']
X_sub['EnergySum_squared'] = X_sub['EnergySum_scaled'] ** 2

X_missing_sub = X_missing[top_features].copy()
X_missing_sub['Fleet_Energy'] = X_missing_sub['Fleet Type_Zipper'] * X_missing_sub['EnergySum_scaled']
X_missing_sub['EnergySum_squared'] = X_missing_sub['EnergySum_scaled'] ** 2

final_features = X_sub.columns.tolist()

# ----------------------------
# 6. Train/Validation Split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_sub, y_full, test_size=0.2, random_state=42
)

print("\nFinal feature list:", final_features)
print("Train shape:", X_train.shape, "Val shape:", X_val.shape)


# ----------------------------
# 7. Custom WeightedKNN Class (FIXED)
# ----------------------------
class WeightedKNN(KNeighborsRegressor):
    """
    Custom KNN that applies per-feature weights in the distance metric.
    Exposes KNeighborsRegressor parameters so GridSearchCV sees them.
    """

    def __init__(
        self,
        feature_weights=None,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        **kwargs
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs
        )
        self.feature_weights = feature_weights
        self._X_weighted = None

    def fit(self, X, y):
        if self.feature_weights is not None:
            X_weighted = X.copy()
            # Multiply each column by its weight
            for col in self.feature_weights:
                X_weighted[col] *= self.feature_weights[col]
            self._X_weighted = X_weighted
            return super().fit(self._X_weighted, y)
        else:
            return super().fit(X, y)

    def predict(self, X):
        if self.feature_weights is not None:
            X_weighted = X.copy()
            for col in self.feature_weights:
                X_weighted[col] *= self.feature_weights[col]
            return super().predict(X_weighted)
        else:
            return super().predict(X)


# ----------------------------
# 8. GridSearch for WeightedKNN
# ----------------------------
weight_configs = [
    {
        'Fleet Type_Zipper': 1.0,
        'EnergySum_scaled': 1.0,
        'Fleet_Energy': 1.0,
        'EnergySum_squared': 1.0
    },
    {
        'Fleet Type_Zipper': 1.0,
        'EnergySum_scaled': 0.8,
        'Fleet_Energy': 0.8,
        'EnergySum_squared': 0.8
    },
    {
        'Fleet Type_Zipper': 1.0,
        'EnergySum_scaled': 0.5,
        'Fleet_Energy': 0.5,
        'EnergySum_squared': 0.5
    }
]

param_grid_wknn = {
    'feature_weights': [pd.Series(cfg) for cfg in weight_configs],
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['distance'],
    'metric': ['minkowski', 'manhattan'],
    'p': [1, 2]
}

wknn_model = WeightedKNN()
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

grid_search_wknn = GridSearchCV(
    estimator=wknn_model,
    param_grid=param_grid_wknn,
    scoring='neg_mean_squared_error',
    cv=cv,
    n_jobs=-1
)

grid_search_wknn.fit(X_train, y_train)

best_wknn = grid_search_wknn.best_estimator_
print("\n=== Best WeightedKNN Params ===")
print(grid_search_wknn.best_params_)

# Evaluate WeightedKNN
wknn_val_pred = best_wknn.predict(X_val)
wknn_mae = mean_absolute_error(y_val, wknn_val_pred)
wknn_rmse = np.sqrt(mean_squared_error(y_val, wknn_val_pred))
wknn_r2 = r2_score(y_val, wknn_val_pred)
wknn_mape = np.mean(np.abs((y_val - wknn_val_pred) / y_val)) * 100

print("\n=== WeightedKNN Performance ===")
print(f"MAE:  {wknn_mae:.2f}")
print(f"RMSE: {wknn_rmse:.2f}")
print(f"R²:   {wknn_r2:.4f}")
print(f"MAPE: {wknn_mape:.2f}%")

# ----------------------------
# 9. Predict Missing Values
# ----------------------------
missing_preds = best_wknn.predict(X_missing_sub)
encoded_data.loc[mask_missing, 'Estimated Average Stage Time'] = missing_preds

print("\nFilled missing values in 'Estimated Average Stage Time'.")
print("\nFinal Stats:")
print(encoded_data['Estimated Average Stage Time'].describe())
