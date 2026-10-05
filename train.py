import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the real dataset
df = pd.read_csv("placementdata_cleaned.csv")
print("📚 Data loaded successfully!")

# 2. Select features to train on
X = df[['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'AptitudeTestScore']]

# 3. Convert Target labels (Placed/NotPlaced) to numbers (1/0)
y = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})

# 4. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
print("🎯 Model training complete!")

# 7. Check accuracy
predictions = model.predict(X_test_scaled)
score = accuracy_score(y_test, predictions)
print(f"📊 Model Accuracy: {score * 100:.2f}%")

# 8. Save model and scaler files to your laptop
with open("placement_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("placement_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("💾 'placement_model.pkl' and 'placement_scaler.pkl' saved!")