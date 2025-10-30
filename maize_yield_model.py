"""
SDG 7: Affordable and Clean Energy
Project: Solar-Powered Agricultural Intelligence — Kenya Maize Yield Predictor

Author: [Your Name]
Institution: PLP Academy
Description:
This script fetches solar and weather data from NASA POWER API,
preprocesses it, and trains a simple regression model to predict maize yield
based on solar irradiance, rainfall, and temperature.
"""

# =======================
# 1. IMPORT LIBRARIES
# =======================
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =======================
# 2. FETCH DATA FROM NASA POWER API
# =======================
print("Fetching solar and climate data from NASA POWER API...")

# Sample coordinates for Nakuru, Kenya
latitude = -0.3031
longitude = 36.0800

# NASA POWER API endpoint
url = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters=T2M,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"
    f"&start=20240101&end=20241231"
    f"&latitude={latitude}&longitude={longitude}&community=AG&format=JSON"
)

response = requests.get(url)
data = response.json()
print(data)  # Debug: print API response

# Defensive check (prevents KeyError)
if "properties" not in data or "parameter" not in data["properties"]:
    print("API Error or unexpected response format.")
    exit(1)

parameters = data["properties"]["parameter"]
df = pd.DataFrame(parameters)
df['Date'] = df.index
# Rearrange columns: Date first, then parameters, ensures all columns present.
df = df[["Date", "T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]]
df["Date"] = pd.to_datetime(df["Date"])

# Convert data types
df = df.astype(float, errors="ignore")

print("[SUCCESS] Data successfully fetched and loaded.")
print(df.head())

# =======================
# 3. DATA CLEANING
# =======================
df.dropna(inplace=True)
df.sort_values("Date", inplace=True)

# Add rolling averages (simulate agricultural seasonal features)
df["rainfall_30d"] = df["PRECTOTCORR"].rolling(window=30).mean()
df["temp_avg_30d"] = df["T2M"].rolling(window=30).mean()
df["solar_30d"] = df["ALLSKY_SFC_SW_DWN"].rolling(window=30).mean()

# Drop NaN rows after rolling
df.dropna(inplace=True)

# =======================
# 4. SIMULATE MAIZE YIELD DATA
# =======================
# (In a real-world project, replace this with actual yield data)
np.random.seed(42)
df["maize_yield"] = (
    1.5 * df["solar_30d"]
    + 0.8 * df["rainfall_30d"]
    - 0.5 * df["temp_avg_30d"]
    + np.random.normal(0, 10, len(df))
)

# =======================
# 5. MODEL TRAINING
# =======================
features = ["solar_30d", "rainfall_30d", "temp_avg_30d"]
X = df[features]
y = df["maize_yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========================")
print("Model Evaluation Results")
print("========================")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# =======================
# 6. VISUALIZATIONS
# =======================
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["rainfall_30d"], label="Rainfall (mm)", color="blue")
plt.plot(df["Date"], df["temp_avg_30d"], label="Temperature (°C)", color="red")
plt.plot(df["Date"], df["solar_30d"], label="Solar Irradiance (MJ/m²)", color="orange")
plt.title("30-Day Climate Trends — Nakuru, Kenya (2024)")
plt.xlabel("Date")
plt.ylabel("Climate Indicators")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("screenshots/climate_trends.png")
plt.show()

# =======================
# 7. SAVE DATA & MODEL OUTPUTS
# =======================
df.to_csv("data/maize_yield_data_nakuru.csv", index=False)
print("\n[SUCCESS] Clean dataset saved to data/maize_yield_data_nakuru.csv")

print("\n[COMPLETE] Project complete — ready for report screenshots and GitHub push.")