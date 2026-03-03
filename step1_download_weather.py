"""
Step 1: Download Extended Weather Data for METR-LA Period
==========================================================
This script downloads hourly weather data from Open-Meteo API
for Los Angeles to cover the full METR-LA dataset period.

Date Range: March 1, 2012 - June 30, 2012
Location: Los Angeles (Lat: 34.0522, Lon: -118.2437)
Features: temperature, precipitation, wind_speed
"""

import pandas as pd
import requests
import time

print("=" * 60)
print("STEP 1: Downloading Extended Weather Data")
print("=" * 60)

# Open-Meteo Historical API for Los Angeles
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.0522,
    "longitude": -118.2437,
    "start_date": "2012-03-01",
    "end_date": "2012-06-30",
    "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
    "timezone": "America/Los_Angeles"
}

print(f"Requesting weather data from {params['start_date']} to {params['end_date']}")
print(f"Location: Los Angeles ({params['latitude']}, {params['longitude']})")

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
    
    # Save to CSV
    csv_filename = "LA_Weather_Hourly_2012_Full.csv"
    df_weather.to_csv(csv_filename, index=False)
    
    print(f"\n✅ SUCCESS! Saved as: {csv_filename}")
    print(f"   Total rows: {len(df_weather)}")
    print(f"   Date range: {df_weather['datetime'].min()} to {df_weather['datetime'].max()}")
    print(f"\n   First 5 rows:")
    print(df_weather.head().to_string(index=False))
    print(f"\n   Last 5 rows:")
    print(df_weather.tail().to_string(index=False))
    print(f"\n   Weather statistics:")
    print(f"   - Temperature: {df_weather['temperature_celsius'].min():.1f}°C to {df_weather['temperature_celsius'].max():.1f}°C")
    print(f"   - Precipitation: {df_weather['precipitation_mm'].max():.1f} mm (max)")
    print(f"   - Wind Speed: {df_weather['wind_speed_kmh'].min():.1f} to {df_weather['wind_speed_kmh'].max():.1f} km/h")
    
else:
    print(f"❌ Failed to fetch data. Error code: {response.status_code}")
    print(f"   Response: {response.text}")
