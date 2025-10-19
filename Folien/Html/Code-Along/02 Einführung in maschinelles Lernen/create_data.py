# %%
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
import requests

# Create a univariate linear dataset

# %%
np.random.seed(2025)
univariate_data_size = 50
data_x = np.random.uniform(low=0.5, high=9.0, size=univariate_data_size)
data_y = 2 * data_x + np.random.normal(loc=0, scale=1, size=univariate_data_size)


# Create the student dataset

# %%
np.random.seed(2025)
num_students = 100
midterm_grades = np.clip(np.random.normal(loc=70, scale=15, size=num_students), 0, 100)
study_hours = np.clip(np.random.normal(loc=10, scale=10, size=num_students), 0, None)
missed_classes = np.random.poisson(lam=2, size=num_students)
random_noise = np.random.normal(loc=0, scale=5, size=num_students)

# %%
final_grades = np.clip(
    (0.9 * midterm_grades + 1.8 * study_hours - 5.0 * missed_classes) + random_noise,
    0,
    100,
)

# %%
print(
    "Created dataset:\n"
    f"  mean midterm grade: {np.mean(midterm_grades)}\n"
    f"  mean final grade: {np.mean(final_grades)}"
)

# Create a DataFrame and save to CSV and SQLite

# %%
df = pd.DataFrame(
    {
        "student_id": range(1, num_students + 1),
        "midterm_grade": midterm_grades,
        "study_hours": study_hours,
        "missed_classes": missed_classes,
        "final_grade": final_grades,
    }
)

# %%
df.to_csv("student_grades.csv", index=False)

# %%
conn = sqlite3.connect("student_grades.db")
df.to_sql("grades", conn, if_exists="replace", index=False)
conn.close()

# Create the ice cream dataset

# Read the weather data for Munich in 2024 from Open-Meteo

# %%
response = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": 48.1374,
        "longitude": 11.5755,
        "hourly": "temperature_2m,cloud_cover",
        "start_date": "2024-05-01",
        "end_date": "2024-10-31",
    },
)

# %%
weather_data = response.json()

# Compute the average temperature and cloud cover for each day, limiting the time
# between 10am and 6pm

# %%
temperatures = weather_data["hourly"]["temperature_2m"]
cloud_covers = weather_data["hourly"]["cloud_cover"]
measurement_datetime = [
    datetime.fromisoformat(dt) for dt in weather_data["hourly"]["time"]
]

# %%
ice_cream_data = pd.DataFrame(
    {
        "datetimes": measurement_datetime,
        "temperature": temperatures,
        "cloud_cover": cloud_covers,
    }
)

# %%
# Drop all lines outside of 10am to 6pm
ice_cream_data = ice_cream_data[ice_cream_data["datetimes"].dt.hour.between(10, 18)]

# %%
# Aggregate to daily data
ice_cream_data = (
    ice_cream_data.groupby(ice_cream_data["datetimes"].dt.date)
    .agg({"temperature": "mean", "cloud_cover": "mean"})
    .reset_index()
)
ice_cream_data.rename(columns={"datetimes": "date"}, inplace=True)

# %%
ice_cream_data

# %%
ice_cream_data.describe()

# %%
# Simulate ice cream sales based on temperature and cloud cover
np.random.seed(2025)
num_days = len(ice_cream_data)
random_noise = np.random.normal(loc=0, scale=100, size=num_days)

# %%
ice_cream_data["sales"] = np.clip(
    100 * ice_cream_data["temperature"]
    - 10 * ice_cream_data["cloud_cover"]
    + random_noise,
    0,
    None,
).astype(int)

# %%
ice_cream_data

# %%
ice_cream_data.describe()

# Write to CSV and SQLite

# %%
ice_cream_data.to_csv("ice_cream_sales.csv", index=False)

# %%
conn = sqlite3.connect("ice_cream_sales.db")
ice_cream_data.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()

# %%
