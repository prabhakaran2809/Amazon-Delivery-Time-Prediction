import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Amazon Deliveriy Time\DataCleaned_FeaturingEngineered.csv")

# Features
features = [
    'Agent_Age', 'Distance', 'Weather_Group', 'Traffic_Code', 
    'Vehicle_Type', 'Area', 'Category_Group', 'Order_Hour', 'Pickup_Hour', 
    'Pickup_TimeOfDay', 'Pickup_Category', 'Is_Weekend', 'Order_DayOfWeek', 
    'Order_Month', 'Weather', 'Traffic', 'Vehicle', 'Category', 'Pickup_Minute',
    'Distance_Category', 'Agent_Age_Group', 'Agent_Rating_Group', 'Time_to_Pickup'
]

X = df[features]
y = df['Delivery_Time']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = ['Distance']
categorical_features = [col for col in features if col not in numeric_features]

# Preprocess numeric features
numeric_imputer_scaler = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
X_train[numeric_features] = numeric_imputer_scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = numeric_imputer_scaler.transform(X_test[numeric_features])

# 🚨 Convert categorical columns to string to avoid float errors
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Define CatBoost Regressor
catboost_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    eval_metric='RMSE',
    random_seed=42,
    verbose=200
)

# Fit using Pool
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)

catboost_model.fit(train_pool)

# Predictions
y_pred = catboost_model.predict(test_pool)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 CatBoost Regressor Performance Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Save the trained model to a .cbm file
catboost_model.save_model("catboost_delivery_time_model.cbm")

