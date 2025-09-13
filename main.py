import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
n_samples = 500
data = {
    'Age': np.random.randint(18, 80, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'PriorNoShows': np.random.randint(0, 5, n_samples),
    'TravelTime': np.random.randint(5, 120, n_samples), # in minutes
    'AppointmentTime': np.random.choice(['Morning', 'Afternoon'], n_samples),
    'ScheduledDayWeek': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], n_samples),
    'NoShow': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) # 20% no-shows
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preprocessing ---
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'AppointmentTime', 'ScheduledDayWeek'], drop_first=True)
# --- 3. Feature Selection and Model Training ---
X = df.drop('NoShow', axis=1)
y = df['NoShow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a RandomForestClassifier (you can experiment with other models)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
# --- 5. Visualization (Feature Importance) ---
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
output_filename = 'feature_importance.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")