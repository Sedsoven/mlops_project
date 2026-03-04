import hopsworks
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load data from Hopsworks
# -------------------------------

API_KEY = "PUT_YOUR_HOPSWORKS_API_KEY_HERE"

project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

truck_data = fs.get_feature_group("truck_final", version=1).select_all().read()

hopsworks.logout()

# -------------------------------
# Columns
# -------------------------------

num_cols = [
'route_avg_temp','route_avg_wind','route_avg_precip','route_avg_humidity',
'route_avg_visibility','route_avg_pressure','origin_avg_temp','origin_avg_wind',
'origin_avg_precip','origin_avg_humidity','origin_avg_visibility',
'origin_avg_pressure','dest_avg_temp','dest_avg_wind','dest_avg_precip',
'dest_avg_humidity','dest_avg_visibility','dest_avg_pressure','avg_nov',
'accident','truck_age','load_capacity_pounds','mileage_mpg','age',
'experience','ratings','average_speed_mph'
]

cat_cols = [
'route_description','origin_description','dest_description',
'fuel_type','gender','driving_style'
]

target = "delay"

# -------------------------------
# Train Test Split
# -------------------------------

train_data = truck_data[truck_data['departure_date'] < pd.Timestamp('2019-02-06', tz='UTC')]
test_data  = truck_data[truck_data['departure_date'] >= pd.Timestamp('2019-02-06', tz='UTC')]

X_train = train_data[num_cols + cat_cols]
y_train = train_data[target]

X_test = test_data[num_cols + cat_cols]
y_test = test_data[target]

# -------------------------------
# Missing values
# -------------------------------

mode_val = X_train['load_capacity_pounds'].mode()[0]

X_train['load_capacity_pounds'].fillna(mode_val, inplace=True)
X_test['load_capacity_pounds'].fillna(mode_val, inplace=True)

# -------------------------------
# Encoding
# -------------------------------

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[cat_cols])

encoded_train = pd.DataFrame(
    encoder.transform(X_train[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)

encoded_test = pd.DataFrame(
    encoder.transform(X_test[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)

X_train = pd.concat([X_train, encoded_train], axis=1)
X_test = pd.concat([X_test, encoded_test], axis=1)

X_train.drop(columns=cat_cols, inplace=True)
X_test.drop(columns=cat_cols, inplace=True)

# -------------------------------
# Scaling
# -------------------------------

scaler = StandardScaler()

scaler.fit(X_train[num_cols])

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
metrics = {"accuracy": accuracy}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Training completed and model + metrics saved.")