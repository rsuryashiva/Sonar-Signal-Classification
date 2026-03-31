import pickle
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import load_data, split_data, scale_data
from src.train import train_model
from src.predict import predict

# 1. Load data
df = load_data("data/sonar_data.csv")

# 2. Split
X_train, X_test, y_train, y_test = split_data(df)

# 3. Scale
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# 4. Train
model = train_model(X_train_scaled, y_train)

# 5. Evaluate
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

# 6. Save model
with open("models/svc_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("\nModel saved successfully!")

# 7. Sample prediction
sample = X_test.iloc[0].tolist()
result = predict(model, scaler, sample)

print("\nSample Prediction:", result)