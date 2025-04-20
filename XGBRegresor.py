# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor  # Import XGBoost Regressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Load dataset
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Amazon Deliveriy Time\DataCleaned_FeaturingEngineered.csv")

# 3. Select relevant features based on your feature list
features = [
    'Agent_Age', 'Distance', 'Weather_Group', 'Traffic_Code', 
    'Vehicle_Type', 'Area', 'Category_Group', 'Order_Hour', 'Pickup_Hour', 
    'Pickup_TimeOfDay', 'Pickup_Category', 'Is_Weekend', 'Order_DayOfWeek', 
    'Order_Month', 'Weather', 'Traffic', 'Vehicle', 'Category', 'Pickup_Minute',
    'Distance_Category', 'Agent_Age_Group', 'Agent_Rating_Group', 'Time_to_Pickup'
]

X = df[features]
y = df['Delivery_Time']

# 4. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Define numeric and categorical columns explicitly
numeric_features = [ 'Distance']
categorical_features = [col for col in X.columns if col not in numeric_features]

# 6. Preprocessing pipelines
numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

# Combine both transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Create a full pipeline with preprocessing and XGBoost Regressor
model = make_pipeline(
    preprocessor,
    XGBRegressor(random_state=42, objective='reg:squarederror')  # Use XGBoost Regressor
)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 XGBoost Regressor Performance Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
