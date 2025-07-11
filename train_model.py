import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from extract_features import extract_features

# Paths
real_path = 'dataset/real'
fake_path = 'dataset/fake'

# Storage for features and labels
features = []
labels = []

# Loop through real currency images
for filename in os.listdir(real_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(real_path, filename)
        feat = extract_features(path)
        features.append(feat)
        labels.append(0)  # 0 = Real

# Loop through fake currency images
for filename in os.listdir(fake_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(fake_path, filename)
        feat = extract_features(path)
        features.append(feat)
        labels.append(1)  # 1 = Fake

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, labels)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
