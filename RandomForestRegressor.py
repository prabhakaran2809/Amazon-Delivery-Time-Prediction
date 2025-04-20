import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming 'df' is your DataFrame and is already cleaned and preprocessed
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Amazon Deliveriy Time\DataCleaned_FeaturingEngineered.csv")

# Select relevant features for the model
features = [
    'Agent_Age', 'Distance', 'Weather_Group', 'Traffic_Code', 
    'Vehicle_Type', 'Area', 'Category_Group', 'Order_Hour', 'Pickup_Hour', 
    'Pickup_TimeOfDay', 'Pickup_Category', 'Is_Weekend', 'Order_DayOfWeek', 
    'Order_Month', 'Weather', 'Traffic', 'Vehicle', 'Category', 'Pickup_Minute',
    'Distance_Category', 'Agent_Age_Group', 'Agent_Rating_Group', 'Time_to_Pickup'
]

# Filter out the target variable
X = df[features]  # Features
y = df['Delivery_Time']  # Target variable (Delivery Time)

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column transformers for categorical and numeric columns
# Label encoding for categorical variables (excluding 'Distance')
label_columns = [col for col in X.columns if col != 'Distance']  # All except 'Distance'
encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns
for col in label_columns:
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])

# Feature scaling only on the numeric column 'Distance'
scaler = StandardScaler()

# Scale the 'Distance' column
X_train[['Distance']] = scaler.fit_transform(X_train[['Distance']])
X_test[['Distance']] = scaler.transform(X_test[['Distance']])

# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # RMSE
mae = mean_absolute_error(y_test, y_pred)  # MAE
r2 = r2_score(y_test, y_pred)  # R²

# Display the results
print("Random Forest Regressor Performance Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
