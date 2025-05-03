import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from google.colab import files

# Step 1: Upload your file
uploaded = files.upload()

# Step 2: Load the Excel file
file_path = list(uploaded.keys())[0]  # Automatically takes the uploaded file name
sheet_data = pd.read_excel(file_path, sheet_name='Sheet1')

# Step 3: Filter out saturated data
sheet_data['Wavelength'] = pd.to_numeric(sheet_data['Wavelength'], errors='coerce')
sheet_data = sheet_data.dropna(subset=['Wavelength'])
filtered_data = sheet_data[(sheet_data['Wavelength'] < 200) | (sheet_data['Wavelength'] > 284)]

# Step 4: Prepare the data
control = filtered_data.iloc[:, 1:4].values
glucose_6_7 = filtered_data.iloc[:, 4:7].values
glucose_13_9 = filtered_data.iloc[:, 7:10].values
glucose_27_9 = filtered_data.iloc[:, 10:13].values
new_group_4 = filtered_data.iloc[:, 13:16].values
new_group_4_7 = filtered_data.iloc[:, 16:19].values
new_group_5_4 = filtered_data.iloc[:, 19:22].values
new_group_6_83 = filtered_data.iloc[:, 22:25].values

X = np.hstack([control, glucose_6_7, glucose_13_9, glucose_27_9, 
               new_group_4, new_group_4_7, new_group_5_4, new_group_6_83]).T
y = np.array([4, 4, 4, 4.05, 4.05, 4.05, 4.11, 4.11, 4.11, 4.23, 4.23, 4.23,
              4, 4, 4, 4.7, 4.7, 4.7, 5.4, 5.4, 5.4, 6.83, 6.83, 6.83])

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Define MWPLS
def moving_window_pls(X, y, window_size, n_components):
    n_features = X.shape[1]
    mse_scores = []
    start_indices = []
    
    for start in range(0, n_features - window_size + 1):
        end = start + window_size
        X_window = X[:, start:end]
        X_train, X_test, y_train, y_test = train_test_split(X_window, y, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train_scaled, y_train)
        y_pred = pls.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        start_indices.append(start)
    
    return mse_scores, start_indices

# Step 7: Run MWPLS
window_size = 50
n_components = 3
mse_scores, start_indices = moving_window_pls(X_scaled, y, window_size, n_components)

# Step 8: Visualize MWPLS results
wavelengths = filtered_data['Wavelength'].values
window_midpoints = [wavelengths[start + window_size // 2] for start in start_indices]

plt.plot(window_midpoints, mse_scores, label='MSE per Window')
plt.xlabel('Wavelength')
plt.ylabel('Mean Squared Error')
plt.title('MWPLS Performance')
plt.legend()
plt.show()

# Step 9: Optimal window identification
optimal_window_index = np.argmin(mse_scores)
optimal_start = start_indices[optimal_window_index]
optimal_end = optimal_start + window_size
print(f"Optimal Window: {optimal_start}-{optimal_end} with MSE = {mse_scores[optimal_window_index]}")

# Step 10: Re-train PLS on optimal window
X_optimal = X_scaled[:, optimal_start:optimal_end]
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_optimal, y, test_size=0.25, random_state=42)
pls_optimal = PLSRegression(n_components=n_components)
pls_optimal.fit(X_train_opt, y_train_opt)
y_pred_opt_test = pls_optimal.predict(X_test_opt)
y_pred_opt_train = pls_optimal.predict(X_train_opt)

# Step 11: Visualize Actual vs Predicted for Training and Test Data
plt.scatter(y_train_opt, y_pred_opt_train, label='Training Data', color='blue')
plt.scatter(y_test_opt, y_pred_opt_test, label='Test Data', color='red')
plt.plot([min(y), max(y)], [min(y), max(y)], '--k', label='Ideal')
plt.xlabel('Actual HbA1c Value')
plt.ylabel('Predicted HbA1c Value')
plt.legend()
plt.title('Actual vs Predicted HbA1c Values')
plt.show()

# Step 12: Predict HbA1c values for each sample
individual_predictions = pls_optimal.predict(X_optimal).flatten()

# Display Predictions
for i, (actual, predicted) in enumerate(zip(y, individual_predictions), start=1):
    print(f"Sample {i}: Actual HbA1c = {actual}, Predicted HbA1c = {predicted:.2f}")
