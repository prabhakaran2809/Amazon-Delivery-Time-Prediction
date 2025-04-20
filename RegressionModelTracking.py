import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Delivery_Time_Regression_Models")

# Load dataset
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Amazon Deliveriy Time\DataCleaned_FeaturingEngineered.csv")

# Features and target
features = [
    'Agent_Age', 'Distance', 'Weather_Group', 'Traffic_Code',
    'Vehicle_Type', 'Area', 'Category_Group', 'Order_Hour', 'Pickup_Hour',
    'Pickup_TimeOfDay', 'Pickup_Category', 'Is_Weekend', 'Order_DayOfWeek',
    'Order_Month', 'Weather', 'Traffic', 'Vehicle', 'Category', 'Pickup_Minute',
    'Distance_Category', 'Agent_Age_Group', 'Agent_Rating_Group', 'Time_to_Pickup'
]

X = df[features]
y = df['Delivery_Time']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: convert categorical features to string
categorical_features = [col for col in features if col != 'Distance']
numeric_features = ['Distance']

for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Preprocessing pipelines
numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Function to log models to MLflow
def log_model(model, model_name, params, metrics):
    with mlflow.start_run(run_name=model_name):
        print(f"Logging model: {model_name}")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model_name == "CatBoost":
            mlflow.catboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)
        print(f"Model {model_name} logged successfully!")

# 📌 Linear Regression
model_lr = make_pipeline(preprocessor, LinearRegression())
params_lr = {'model': 'LinearRegression'}
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
metrics_lr = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'R²': r2_score(y_test, y_pred_lr)
}
log_model(model_lr, "Linear_Regression", params_lr, metrics_lr)

# 📌 LightGBM
model_lgb = make_pipeline(preprocessor, LGBMRegressor(random_state=42, learning_rate=0.05, n_estimators=100))
params_lgb = {'model': 'LightGBM', 'learning_rate': 0.05, 'n_estimators': 100}
model_lgb.fit(X_train, y_train)
y_pred_lgb = model_lgb.predict(X_test)
metrics_lgb = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lgb)),
    'MAE': mean_absolute_error(y_test, y_pred_lgb),
    'R²': r2_score(y_test, y_pred_lgb)
}
log_model(model_lgb, "LightGBM", params_lgb, metrics_lgb)

# 📌 XGBoost
model_xgb = make_pipeline(preprocessor, XGBRegressor(random_state=42, objective='reg:squarederror', learning_rate=0.05, n_estimators=100))
params_xgb = {'model': 'XGBoost', 'learning_rate': 0.05, 'n_estimators': 100}
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
metrics_xgb = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'R²': r2_score(y_test, y_pred_xgb)
}
log_model(model_xgb, "XGBoost", params_xgb, metrics_xgb)

# 📌 Random Forest
model_rf = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100, random_state=42))
params_rf = {'model': 'RandomForest', 'n_estimators': 100}
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
metrics_rf = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'R²': r2_score(y_test, y_pred_rf)
}
log_model(model_rf, "RandomForest", params_rf, metrics_rf)

# 📌 Gradient Boosting
model_gb = make_pipeline(preprocessor, GradientBoostingRegressor(random_state=42))
params_gb = {'model': 'GradientBoosting'}
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
metrics_gb = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'MAE': mean_absolute_error(y_test, y_pred_gb),
    'R²': r2_score(y_test, y_pred_gb)
}
log_model(model_gb, "GradientBoosting", params_gb, metrics_gb)

# 📌 CatBoost (separate handling without one-hot encoding)

# Preprocess numeric 'Distance' only
numeric_transformer_cb = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
X_train[numeric_features] = numeric_transformer_cb.fit_transform(X_train[numeric_features])
X_test[numeric_features] = numeric_transformer_cb.transform(X_test[numeric_features])

# Convert categorical columns to string (ensure)
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Get categorical column indices for CatBoost
cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features]

# Initialize and fit CatBoost
params_cat = {'model': 'CatBoost', 'iterations': 1000, 'learning_rate': 0.05, 'depth': 8}
model_cat = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    eval_metric='RMSE',
    random_seed=42,
    verbose=200
)

train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feature_indices)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_feature_indices)

model_cat.fit(train_pool)
y_pred_cat = model_cat.predict(test_pool)

metrics_cat = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_cat)),
    'MAE': mean_absolute_error(y_test, y_pred_cat),
    'R²': r2_score(y_test, y_pred_cat)
}

log_model(model_cat, "CatBoost", params_cat, metrics_cat)

print("✅ All models trained and logged to MLflow successfully!")
