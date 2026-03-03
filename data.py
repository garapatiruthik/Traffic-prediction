import pandas as pd
import requests

print("Downloading Los Angeles weather data...")

# Open-Meteo Historical API for Los Angeles (Lat: 34.0522, Lon: -118.2437)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.0522,
    "longitude": -118.2437,
    "start_date": "2012-03-01",
    "end_date": "2012-06-30",
    "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
    "timezone": "America/Los_Angeles" # Crucial: Matches METR-LA local time!
}

# Fetch the data
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    hourly = data["hourly"]
    
    # Create a clean Pandas DataFrame
    df_weather = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_celsius": hourly["temperature_2m"],
        "precipitation_mm": hourly["precipitation"],
        "wind_speed_kmh": hourly["windspeed_10m"]
    })
    
    # Save it to a CSV file directly in your Colab workspace
    csv_filename = "LA_Weather_Hourly_2012.csv"
    df_weather.to_csv(csv_filename, index=False)
    
    print(f"SUCCESS! Saved as: {csv_filename}")
    print(f"Total rows: {len(df_weather)}")
    print("\nFirst 5 rows of your new dataset:")
    print(df_weather.head())

else:
    print("Failed to fetch data. Error code:", response.status_code)