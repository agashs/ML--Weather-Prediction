# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement 

Environmental changes such as air pollution, temperature fluctuations, and energy demand have a direct impact on human health, productivity, and sustainability. Accurate prediction of these parameters is essential for effective planning, monitoring, and decision‑making. Traditional statistical methods often fail to capture complex non‑linear relationships in sensor data. Hence, machine learning techniques like Decision Tree Regression are applied to predict PM2.5 pollution levels, daily temperature, and energy consumption (TSR) using environmental sensor inputs

DATASET

Source: Environmental monitoring sensor data collected over time.

Type: Multivariate time‑series dataset with both continuous and categorical attributes.

Size: Contains multiple records of daily environmental readings.

**Features in the Dataset**
1) time – Timestamp of the recorded observation.

2) hum – Humidity level (%) in the environment.

3) tem – Ambient temperature (°C).

4) co2 – Carbon dioxide concentration (ppm).

5) illumination – Light intensity (lux).

6) pressure – Atmospheric pressure (hPa).

7) pm2_5 – Fine particulate matter concentration (µg/m³).

8) pm10 – Coarse particulate matter concentration (µg/m³).

9) wind_direction – Categorical indicator of wind direction (e.g., N, S, E, W).

10) wind_direction_angle – Numerical angle of wind direction (degrees).

11) wind_speed – Wind speed (m/s).

12) wind_speed_level – Categorical level of wind speed (e.g., low, medium, high).

13) tsr – Total solar radiation (W/m²).

14) bat – Battery level of the sensor device (%).
    

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1). Load the dataset, select environmental features, and handle missing values.

2). Split data into training and testing sets for each target (PM2.5, temperature, energy).

3). Train separate Decision Tree Regressor models for each target and generate predictions.

4). Evaluate model performance using RMSE and R² scores, then display accuracy results.
## Program:

Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.

Developed by: AGASH S

RegisterNumber:  212224040014

```
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("weather.csv")


df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)


df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek


df = df.dropna(subset=["tem"])


features = [
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm2_5",
    "pm10",
    "wind_direction",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level",
    "tsr",
    "hour",
    "dayofyear",
    "month",
    "dayofweek"
]

X = df[features]
y = df["tem"]


numeric_features = [
    "hum", "co2", "illumination", "pressure",
    "pm2_5", "pm10", "wind_direction_angle",
    "wind_speed", "wind_speed_level", "tsr",
    "hour", "dayofyear", "month", "dayofweek"
]

categorical_features = ["wind_direction"]


preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])


rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", rf)
])


split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Results")
print("---------------------")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)
```
## Output:

<img width="377" height="129" alt="image" src="https://github.com/user-attachments/assets/c9c5013e-dd3f-49ec-968b-b5e382521a0d" />


## Result:

THUS THE ABOVE PROGRAM HAS BEEN VERIFIED AND EXECUTED SUCCESSFULLY.
