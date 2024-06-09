import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import requests
from openai import OpenAI

def app():
    st.title("Model 1: Customer Traffic Prediction")
    
    choice = st.selectbox(
        "Choose an option: ",
        ['All year prediction', 'Real time prediction'],
        index=None,
        placeholder="Select your option...")
    
    if choice == 'All year prediction':
        all_year_prediction()
    elif choice == 'Real time prediction':
        st.write("This feature provides predictions about DineInCustomer for **the next five days**.")
        real_time_prediction()
        if 'predicted_df' in st.session_state:
            st.header("Recommendation for the next five days:")
            predicted_df = st.session_state.predicted_df
            recommendations = generate_recommendations(predicted_df)
            st.write(recommendations)
  
def all_year_prediction():
    df = pd.read_excel('Restaurant_2025_testt.xlsx')
    X = df.drop('DineInCustomers', axis=1)
    
    with open('Model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        
    # Make predictions
    y_pred = model.predict(X)
    y_pred = np.round(y_pred).astype(int)
    df['DineInCustomers'] = y_pred
    st.write(df)
    
    st.markdown("""
    <iframe title="Dashboard customer traffic" width="1070" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=ea9e1de3-b320-444e-97cb-1e4290e7926d&autoAuth=true&ctid=5e158e2a-0596-4a6c-8801-3502aef4563f" frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)

def real_time_prediction():
    api_key = "309cc38e45807c88ceaed0ddc7937fb8"  # API key
    location = "Hanoi"
    weather_data = get_weather_data(api_key, location)

    if weather_data:
        df = create_dataframe(weather_data)
        df["Weather"] = df["weather"].apply(classify_weather)
        traffic_cleaned = df.copy()
        traffic_cleaned = traffic_cleaned.drop(columns='weather')
    
    traffic_cleaned['Date'] = pd.to_datetime(traffic_cleaned['Date'], format="%d-%m-%Y")
    traffic_cleaned['Year'] = traffic_cleaned['Date'].dt.year
    traffic_cleaned['Month'] = traffic_cleaned['Date'].dt.month
    traffic_cleaned['Day'] = traffic_cleaned['Date'].dt.day

    traffic_cleaned = traffic_cleaned.drop(columns=["Date","Day of week"])

    # Tạo các cột mới với giá trị TRUE/FALSE
    traffic_cleaned["Weather_Cloudy"] = traffic_cleaned["Weather"] == "Cloudy"
    traffic_cleaned["Weather_Rainy"] = traffic_cleaned["Weather"] == "Rainy"
    traffic_cleaned["Weather_Sunny"] = traffic_cleaned["Weather"] == "Sunny"

    # Chỉ giữ lại các cột mới
    traffic_cleaned = traffic_cleaned.drop(columns='Weather')

    # Chuyển các giá trị TRUE/FALSE thành chữ in hoa
    df = df.map(lambda x: "TRUE" if x else "FALSE")

    traffic_cleaned["Year"]=2025

    # Import new data 2025
    df = pd.read_excel("Data_predicted_2025.xlsx")
    df = df.drop(columns=['Temperature','Weather_Rainy','Weather_Sunny','Weather_Cloudy'])

    new = pd.merge(df, traffic_cleaned, on=["Year", "Month", "Day","Hour"], how="outer")
    new = new.dropna(subset='Temperature')

    new_order = ['Hour', 'DineInCustomers', 'TakeawayOrders', 'OnlineOrders',
        'Promotion', 'Temperature', 'Year', 'Month', 'Day', 'Weekday',
        'Weather_Cloudy', 'Weather_Rainy', 'Weather_Sunny', 'Event_Concert',
        'Event_Local Festival', 'Event_No Event', 'Event_Sports Event',
        'DayOfWeek_Friday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday',
        'DayOfWeek_Sunday', 'DayOfWeek_Thursday', 'DayOfWeek_Tuesday',
        'DayOfWeek_Wednesday']
    new = new[new_order]
    new = new.dropna(subset='Weekday')

    X_test = new.drop('DineInCustomers', axis=1)
    
    with open('Model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    new['DineInCustomers'] = y_pred
    st.write(new)
    
    required_columns = ['Hour', 'DineInCustomers', 'TakeawayOrders', 'OnlineOrders', 'Promotion', 'Temperature', 'Year', 'Month', 'Day', 'Weekday']
    predicted_df = new[required_columns]
    
    st.session_state.predicted_df = predicted_df
        
def get_weather_data(api_key, location):
    # URL cho API dự báo thời tiết  của OpenWeatherMap
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"

    # Gửi yêu cầu HTTP GET tới API
    response = requests.get(url)

    # Kiểm tra xem yêu cầu có thành công không
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}, Response: {response.text}")
        return None

def create_dataframe(weather_data):
    # Danh sách để lưu trữ dữ liệu
    data = []

    for item in weather_data['list']:
        dt = datetime.datetime.fromtimestamp(item['dt'])
        date = dt.date()
        hour = dt.hour
        day_of_week = dt.strftime('%A')
        weather = item['weather'][0]['main']
        temperature = item['main']['temp']

        if 10 <= hour <= 21:  # Lọc giờ từ 10:00 đến 21:00
            data.append([date, hour, day_of_week,weather, temperature])

    # Tạo DataFrame
    df = pd.DataFrame(data, columns=['Date', 'Hour', 'Day of week','weather' , 'Temperature'])
    return df

def classify_weather(weather):
    if weather == "Clouds":
        return "Cloudy"
    elif weather == "Rain":
        return "Rainy"
    else:
        return "Sunny"
    
def generate_recommendations(predicted_df):
    client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])
    data_str = predicted_df.to_string(index=False)
    prompt = f"Based on the following customer traffic predictions for the next five days, provide recommendations for staffing, inventory management, and marketing strategies:\n{data_str}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    
    recommendation = response.choices[0].message.content.strip()
    return recommendation


