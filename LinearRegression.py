# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 2. Load dataset (assuming df is the loaded DataFrame)
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Amazon Deliveriy Time\DataCleaned_FeaturingEngineered.csv")

# 3. Preprocess the data

# Manually define numeric and categorical feature lists
numeric_features = [
    'Distance'
]

categorical_features = features = [
    'Agent_Age', 'Weather_Group', 'Traffic_Code', 
    'Vehicle_Type', 'Area', 'Category_Group', 'Order_Hour', 'Pickup_Hour', 
    'Pickup_TimeOfDay', 'Pickup_Category', 'Is_Weekend', 'Order_DayOfWeek', 
    'Order_Month', 'Weather', 'Traffic', 'Vehicle', 'Category', 'Pickup_Minute',
    'Distance_Category', 'Agent_Age_Group', 'Agent_Rating_Group', 'Time_to_Pickup'
]

X = df[numeric_features + categorical_features]  # Features
y = df['Delivery_Time']  # Target variable (Delivery Time)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Define preprocessing steps (using the manual feature definitions)
numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

# Combine the transformers into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Create a full pipeline with preprocessor and model
model = make_pipeline(preprocessor, LinearRegression())

# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)

# Compute RMSE manually (since sklearn version might not support 'squared=False' in mean_squared_error)
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = mse ** 0.5  # Taking the square root of MSE to get RMSE
print(f'RMSE: {rmse:.2f}')

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.2f}')

# R² (R-squared)
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2:.2f}')
