import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")
