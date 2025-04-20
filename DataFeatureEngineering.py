import pandas as pd
from geopy.distance import geodesic
from pandas.api.types import CategoricalDtype
from datetime import timedelta

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Amazon Deliveriy Time\Cleaned_Dataset.csv",na_values=['NaN ', 'NULL', 'null', ''])

# Ensure that 'Order_Date' is in datetime format
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

# Extract the Day of Week (Monday, Tuesday, ...)
df['Order_DayOfWeek'] = df['Order_Date'].dt.day_name()

# Extract the Month Name (January, February, ...)
df['Order_Month'] = df['Order_Date'].dt.month_name()

# Check if the order was placed on a weekend (Saturday or Sunday)
df['Is_Weekend'] = df['Order_Date'].dt.weekday.isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday
df['Is_Weekend'] = df['Is_Weekend'].map({1: 'Yes', 0: 'No'})  # Convert 1/0 to Yes/No

# Convert Order_Time to datetime by adding a dummy date, so we can extract the hour
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce')

# Now you can extract the hour
df['Order_Hour'] = df['Order_Time'].dt.hour

# Ensure Pickup_Time is in datetime format
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce')

# Extract the Hour of Pickup
df['Pickup_Hour'] = df['Pickup_Time'].dt.hour

# Extract the Minute of Pickup
df['Pickup_Minute'] = df['Pickup_Time'].dt.minute

# Extract the Time of Day (Morning, Afternoon, Evening, Night)
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'
        
df['Pickup_TimeOfDay'] = df['Pickup_Time'].dt.hour.apply(get_time_of_day)

def calculate_distance(row):
    store_coords = (row['Store_Latitude'], row['Store_Longitude'])
    drop_coords = (row['Drop_Latitude'], row['Drop_Longitude'])
    return geodesic(store_coords, drop_coords).km

df['Distance'] = df.apply(calculate_distance, axis=1)

# Categorize Distance into Short, Medium, and Long
def categorize_distance(distance):
    if distance < 5:
        return 'Short'
    elif 5 <= distance < 20:
        return 'Medium'
    else:
        return 'Long'

df['Distance_Category'] = df['Distance'].apply(categorize_distance)

# Categorize valid ages
def categorize_agent_age(age):
    if 18 <= age <= 25:
        return 'Young'
    elif 26 <= age <= 40:
        return 'Middle-aged'
    else:
        return 'Senior'

df['Agent_Age_Group'] = df['Agent_Age'].apply(categorize_agent_age)

def categorize_agent_rating(rating):
    if 4.5 <= rating <= 6:
        return 'High'
    elif 3 <= rating < 4.5:
        return 'Medium'
    elif rating < 3:
        return 'Low'
    return 'Unknown'

df['Agent_Rating_Group'] = df['Agent_Rating'].apply(categorize_agent_rating)

weather_mapping = {
    'Sunny': 'Sunny',
    'Cloudy': 'Cloudy',
    'Stormy': 'Bad',
    'Sandstorms': 'Bad',
    'Fog': 'Bad',
    'Windy': 'Bad'
}
df['Weather_Group'] = df['Weather'].map(weather_mapping)

traffic_order = CategoricalDtype(categories=['Low ', 'Medium ', 'High ', 'Jam '], ordered=True)
df['Traffic'] = df['Traffic'].astype(traffic_order)
df['Traffic_Code'] = df['Traffic'].cat.codes

df['Area'] = df['Area'].replace({'Metropolitian ': 'Metropolitan'})

category_mapping = {
    'Grocery': 'Essentials',
    'Snacks': 'Essentials',
    'Books': 'Lifestyle',
    'Kitchen': 'Lifestyle',
    'Home': 'Lifestyle',
    'Pet Supplies': 'Lifestyle',
    'Skincare': 'Lifestyle',
    'Clothing': 'Fashion',
    'Shoes': 'Fashion',
    'Jewelry': 'Fashion',
    'Apparel': 'Fashion',
    'Cosmetics': 'Fashion',
    'Electronics': 'Electronics',
    'Sports': 'Lifestyle',
    'Outdoors': 'Lifestyle',
    'Toys': 'Lifestyle'
}
df['Category_Group'] = df['Category'].map(category_mapping)

vehicle_mapping = {
    'bicycle ': 'Two-wheeler',
    'scooter ': 'Two-wheeler',
    'motorcycle ': 'Two-wheeler',
    'van': 'Four-wheeler'
}
df['Vehicle_Type'] = df['Vehicle'].map(vehicle_mapping)

# Convert Order_Date to datetime
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Convert Order_Time and Pickup_Time to datetime.time
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce').dt.time

# Create Order_DateTime
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Time'].astype(str), errors='coerce')

# Now create Pickup_DateTime — if Pickup_Time < Order_Time, add 1 day to Order_Date
df['Pickup_DateTime'] = df.apply(
    lambda row: pd.to_datetime(f"{row['Order_Date']} {row['Pickup_Time']}") + timedelta(days=1)
    if row['Pickup_Time'] < row['Order_Time']
    else pd.to_datetime(f"{row['Order_Date']} {row['Pickup_Time']}"),
    axis=1
)

# Calculate Time to Pickup (in minutes)
df['Time_to_Pickup'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60

# Round to nearest minute
df['Time_to_Pickup'] = df['Time_to_Pickup'].round()

# Drop helper columns
df.drop(columns=['Order_DateTime', 'Pickup_DateTime'], inplace=True)

def categorize_pickup_time_fixed(minutes):
    if minutes <= 5:
        return 'Fast'
    elif minutes <= 10:
        return 'Moderate'
    else:
        return 'Slow'

df['Pickup_Category'] = df['Time_to_Pickup'].apply(categorize_pickup_time_fixed)

df.to_csv("DataCleaned_FeaturingEngineered.csv",index=False)

print(df.shape)

