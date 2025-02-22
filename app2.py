import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from flask import Flask, request, jsonify

# Load Dataset
file_path = "House Price India.csv\House Price India.csv"
df = pd.read_csv(file_path)

# Data Cleaning
# Drop irrelevant columns
df.drop(columns=['id', 'Date', 'Postal Code'], inplace=True, errors='ignore')

# Handle missing values
df.fillna(df.median(), inplace=True)

# Feature Engineering
df['House Age'] = 2025 - df['Built Year']
df['Total Area'] = df['living area'] + df['Area of the basement']
df['Was Renovated'] = df['Renovation Year'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['Built Year', 'Renovation Year', 'living area', 'Area of the basement'], inplace=True)

# EDA - Visualizations
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title("House Price Distribution")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting features and target variable
X = df.drop(columns=['Price'])
y = df['Price']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features)
])

# Model Selection & Training
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# Hyperparameter Tuning
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Feature Importance
feature_importances = model.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
plt.title('Feature Importance')
plt.show()

# Save Model
pickle.dump(model, open("house_price_model.pkl", "wb"))

# Flask API for Deployment
app = Flask(__name__)
model = pickle.load(open("house_price_model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
