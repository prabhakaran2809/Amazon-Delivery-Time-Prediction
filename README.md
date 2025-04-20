
# Amazon Delivery Time Prediction

## 📖 Project Overview
This project aims to predict delivery times for e-commerce orders based on several influencing factors such as product category, delivery distance, traffic conditions, weather, and shipping method. By utilizing a structured dataset containing historical delivery data, this project builds, evaluates, and compares various regression models to predict delivery times with high accuracy.

The project integrates **MLflow** to track experiments, manage model versions, and log performance metrics. The best-performing model is deployed via a user-friendly **Streamlit** app that allows users to input order details and instantly receive predicted delivery times.

## 🧹 Data Cleaning & Preprocessing
- **Data Cleaning:** 
  - Handled missing values.
  - Converted date and time columns to appropriate formats.
  - Removed invalid coordinates and outliers with unreasonable delivery distances.
- **Imputation:** Missing data was imputed using area-wise averages.
- The cleaned data is saved as `Cleaned_Dataset.csv`.

## 🔧 Feature Engineering
- **Feature Extraction:** 
  - Extracted time-related features such as order and pickup times.
  - Categorized delivery distances and included agent info, weather, and traffic conditions.
  - The time difference between order and pickup was calculated.
- Processed data is saved as `DataCleaned_FeaturingEngineered.csv`.

## 📊 Model Training and Tracking with MLflow
- **Models Trained:** 
  - Linear Regression
  - LightGBM
  - XGBoost
  - Random Forest
  - Gradient Boosting
  - CatBoost
- **Model Tracking:** 
  - **MLflow** is used for tracking experiments, logging model performance metrics (RMSE, MAE, R²), and managing hyperparameters and versions.
- **Model Logging:** 
  - Models are preprocessed and logged in **MLflow** for easy comparison and versioning.

## 📦 Delivery Time Prediction App
The **Streamlit app** predicts delivery times using **CatBoost** and **LightGBM** models. Users can input features such as agent age, distance, weather, and traffic, then select the model to estimate the delivery time in hours. It provides an intuitive interface for fast predictions.

### Models Used:
- **CatBoost** 
- **LightGBM**

## 💻 Tech Stack
- **Programming Language:** Python
- **Libraries/Frameworks:**
  - **Machine Learning:** Scikit-learn, CatBoost, LightGBM, XGBoost, Random Forest, Gradient Boosting
  - **Model Tracking:** MLflow
  - **Web Application:** Streamlit
  - **Data Processing:** Pandas, NumPy
  - **Model Serialization:** Pickle, CatBoost model format

## 🚀 Usage Instructions

1. **Clone the Repository:**
   ```
   git clone https://github.com/your-username/Amazon-Delivery-Time-Prediction.git
   cd Amazon-Delivery-Time-Prediction
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.7+ installed. You can install the required libraries using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   Start the Streamlit app with the following command:
   ```
   streamlit run app.py
   ```

4. **Model Selection and Prediction:**
   - On the Streamlit interface, enter the details of the delivery (e.g., agent age, weather, traffic, etc.).
   - Choose the model (CatBoost or LightGBM) and click on **"Predict Delivery Time"** to receive an estimated delivery time.

## 📝 Future Work
- **Additional Model Improvements:** Further optimization of hyperparameters for better prediction accuracy.
- **Model Comparison:** Implement more advanced evaluation metrics and techniques like cross-validation and ensemble methods.
- **Deployment:** Deploy the model as an API using **Flask** or **FastAPI** for scalable production use.


