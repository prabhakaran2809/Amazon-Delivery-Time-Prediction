import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import pickle

# Load the trained model
catboost_model = CatBoostRegressor()
catboost_model.load_model("catboost_delivery_time_model.cbm")

with open("lightgbm_delivery_time_model.pkl", "rb") as f:
    lightgbm_model = pickle.load(f)

# --- Feature Engineering Functions ---

def categorize_agent_age(age):
    if 18 <= age <= 25:
        return 'Young'
    elif 26 <= age <= 40:
        return 'Middle-aged'
    else:
        return 'Senior'

def categorize_distance(distance):
    if distance < 5:
        return 'Short'
    elif 5 <= distance < 20:
        return 'Medium'
    else:
        return 'Long'

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

def categorize_pickup_time_fixed(minutes):
    if minutes <= 5:
        return 'Fast'
    elif minutes <= 10:
        return 'Moderate'
    else:
        return 'Slow'

# --- Sidebar Model Selection ---
model_choice = st.sidebar.selectbox("Select Model", ["CatBoost Regressor", "LightGBM Regressor"])

# --- Streamlit App Title ---
st.title(f"📦 Delivery Time Predictor ({model_choice})")

# Inputs
agent_age = st.number_input("Agent Age", min_value=18, max_value=65, value=30)
agent_rating_group = st.selectbox("Agent Rating Category", ['High', 'Medium', 'Low'])
distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0)
weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Stormy', 'Sandstorms', 'Fog', 'Windy'])
traffic = st.selectbox("Traffic", ['Low ', 'Medium ', 'High ', 'Jam '])
vehicle = st.selectbox("Vehicle", ['bicycle ', 'scooter ', 'motorcycle ', 'van'])
area = st.selectbox("Area", ['Metropolitan', 'Urban', 'Semi-Urban','Other'])
category = st.selectbox("Category", ['Grocery', 'Snacks', 'Books', 'Kitchen', 'Home', 'Pet Supplies', 'Skincare',
                                     'Clothing', 'Shoes', 'Jewelry', 'Apparel', 'Cosmetics', 'Electronics', 'Sports',
                                     'Outdoors', 'Toys'])

# Time to pickup dropdown (select from options: 5 minutes, 10 minutes, 15 minutes)
time_to_pickup_minutes = st.selectbox("Time to Pickup (in minutes)", [5, 10, 15], index=0)

# Calculate order and pickup time
order_dt = datetime.now()
pickup_dt = order_dt + timedelta(minutes=time_to_pickup_minutes)

# Predict button
if st.button("Predict Delivery Time"):

    # Derived fields
    agent_age_group = categorize_agent_age(agent_age)
    distance_category = categorize_distance(distance)

    weather_group = {'Sunny': 'Sunny', 'Cloudy': 'Cloudy', 'Stormy': 'Bad', 'Sandstorms': 'Bad',
                     'Fog': 'Bad', 'Windy': 'Bad'}[weather]

    traffic_code = {'Low ': 0, 'Medium ': 1, 'High ': 2, 'Jam ': 3}[traffic]

    vehicle_type = {'bicycle ': 'Two-wheeler', 'scooter ': 'Two-wheeler',
                    'motorcycle ': 'Two-wheeler', 'van': 'Four-wheeler'}[vehicle]

    category_group = {
        'Grocery': 'Essentials', 'Snacks': 'Essentials',
        'Books': 'Lifestyle', 'Kitchen': 'Lifestyle', 'Home': 'Lifestyle', 'Pet Supplies': 'Lifestyle',
        'Skincare': 'Lifestyle', 'Clothing': 'Fashion', 'Shoes': 'Fashion', 'Jewelry': 'Fashion',
        'Apparel': 'Fashion', 'Cosmetics': 'Fashion', 'Electronics': 'Electronics',
        'Sports': 'Lifestyle', 'Outdoors': 'Lifestyle', 'Toys': 'Lifestyle'
    }[category]

    order_hour = order_dt.hour
    pickup_hour = pickup_dt.hour
    pickup_minute = pickup_dt.minute
    pickup_timeofday = get_time_of_day(pickup_hour)

    print(order_hour)
    print(pickup_hour)
    print(pickup_minute)
    print(pickup_timeofday)

    is_weekend = 'Yes' if order_dt.weekday() in [5, 6] else 'No'
    order_dayofweek = order_dt.strftime("%A")
    order_month = order_dt.strftime("%B")

    print(is_weekend)
    print(order_dayofweek)
    print(order_month)


    time_to_pickup = round((pickup_dt - order_dt).total_seconds() / 60)
    pickup_category = categorize_pickup_time_fixed(time_to_pickup)

    print(time_to_pickup)
    print(pickup_category)

    # Form dataframe for prediction
    input_df = pd.DataFrame([{
        'Agent_Age': agent_age,
        'Distance': distance,
        'Weather_Group': weather_group,
        'Traffic_Code': traffic_code,
        'Vehicle_Type': vehicle_type,
        'Area': area,
        'Category_Group': category_group,
        'Order_Hour': order_hour,
        'Pickup_Hour': pickup_hour,
        'Pickup_TimeOfDay': pickup_timeofday,
        'Pickup_Category': pickup_category,
        'Is_Weekend': is_weekend,
        'Order_DayOfWeek': order_dayofweek,
        'Order_Month': order_month,
        'Weather': weather,
        'Traffic': traffic,
        'Vehicle': vehicle,
        'Category': category,
        'Pickup_Minute': pickup_minute,
        'Distance_Category': distance_category,
        'Agent_Age_Group': agent_age_group,
        'Agent_Rating_Group': agent_rating_group,
        'Time_to_Pickup': time_to_pickup
    }])

    # --- Prediction based on selected model ---
    if model_choice == "CatBoost Regressor":
        prediction = catboost_model.predict(input_df)[0]
    else:
        prediction = lightgbm_model.predict(input_df)[0]

    st.success(f"🚚 Estimated Delivery Time: {prediction:.2f} Hours")
