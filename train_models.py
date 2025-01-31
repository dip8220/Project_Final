import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = r'D:\Final Year Project\Expanded_Dataset_1K.xlsx'
data = pd.read_excel(file_path)

# Check and handle missing values
data.dropna(inplace=True)  # Remove rows with missing values

# Features and target
X = data[['Cs', 'FA', 'MA', 'Cl', 'Br', 'I']]
y = data['Calculated_bandgap']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Standardize target variable (y)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Save scalers to files
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# Train and save models
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'knn': KNeighborsRegressor(n_neighbors=5),
    'svr': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    # Save the trained models
    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Train and save the ANN model
ann = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Save ANN model (HDF5 format)
ann.save('ann_model.h5')

# Print success message
print("Models and scalers have been trained and saved successfully!")
