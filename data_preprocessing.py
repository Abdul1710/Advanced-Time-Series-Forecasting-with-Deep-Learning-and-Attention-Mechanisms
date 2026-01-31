import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "jena_climate_2009_2016.csv"
LOOKBACK = 24          # 24 time steps (e.g., last 24 hours)
TARGET_COL = "T (degC)"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Parse datetime and sort
df['Date Time'] = pd.to_datetime(df['Date Time'])
df = df.sort_values('Date Time')
df.set_index('Date Time', inplace=True)

# -----------------------------
# MISSING VALUE HANDLING
# -----------------------------
# Forward fill is justified: sensor continuity assumption
df = df.ffill()

# -----------------------------
# FEATURE SELECTION
# -----------------------------
# Drop highly redundant or identifier columns
features = [
    'T (degC)', 'p (mbar)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
    'VPdef (mbar)', 'sh (g/kg)', 'rho (g/m**3)',
    'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'
]

df = df[features]

# -----------------------------
# TRAIN / VAL / TEST SPLIT (TIME-BASED)
# -----------------------------
n = len(df)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# -----------------------------
# SCALING (FIT ONLY ON TRAIN)
# -----------------------------
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

# -----------------------------
# WINDOWING FUNCTION
# -----------------------------
def create_sequences(data, lookback, target_index):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback, target_index])
    return np.array(X), np.array(y)

target_index = features.index(TARGET_COL)

X_train, y_train = create_sequences(train_scaled, LOOKBACK, target_index)
X_val, y_val = create_sequences(val_scaled, LOOKBACK, target_index)
X_test, y_test = create_sequences(test_scaled, LOOKBACK, target_index)

# -----------------------------
# SHAPE CHECK (CRITICAL)
# -----------------------------
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
