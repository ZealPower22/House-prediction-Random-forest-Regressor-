import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("House Price India.csv\House Price India.csv")

# Data Cleaning
# Drop irrelevant columns
df.drop(columns=['id', 'Date', 'Postal Code'], inplace=True)

# Handle missing values (if any)
df.fillna(df.median(), inplace=True)

# Feature Engineering
# Create new features
df['House Age'] = 2025 - df['Built Year']
df['Total Area'] = df['living area'] + df['Area of the basement']
df['Was Renovated'] = df['Renovation Year'].apply(lambda x: 1 if x > 0 else 0)

# Drop now redundant columns
df.drop(columns=['Built Year', 'Renovation Year', 'living area', 'Area of the basement'], inplace=True)

# Splitting features and target variable
X = df.drop(columns=['Price'])
y = df['Price']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = []  # No categorical columns in the current dataset

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features)
])

# Model Selection & Training
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
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

# Feature Importance
feature_importances = model.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
plt.title('Feature Importance')
plt.show()
