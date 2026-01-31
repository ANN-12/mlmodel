# TypeID Keystroke Dynamics - train_model_raw.py (STABLE VERSION)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

print(" TypeID - Training on RAW keystroke data (656 samples)")

# --------------------------------------------------
# 1. LOAD DATA (FIXED PATH)
# --------------------------------------------------
DATA_PATH = "keystroke_data.csv"  # adjust if needed
df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. FEATURE SELECTION
# --------------------------------------------------
features = [
    "ks_count", "ks_rate",
    "dwell_mean", "dwell_std",
    "flight_mean", "flight_std",
    "digraph_mean", "digraph_std",
    "backspace_rate", "wps", "wpm"
]

X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["user_id"]

# --------------------------------------------------
# 3. LABEL ENCODING + SCALING
# --------------------------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nUsers per class:")
print(pd.Series(y).value_counts())

# --------------------------------------------------
# 4. MODEL DEFINITION
# --------------------------------------------------
model = xgb.XGBClassifier(
    objective="multi:softprob",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# 5. CROSS-VALIDATION (STABLE ACCURACY)
# --------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv)

print("\n📊 CROSS-VALIDATION RESULTS")
print("Fold accuracies:", np.round(cv_scores, 3))
print("Mean CV accuracy:", f"{cv_scores.mean()*100:.2f}%")

# --------------------------------------------------
# 6. FINAL TRAIN-TEST (FOR REPORTS)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n📈 FINAL TEST RESULTS")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n🏆 Top Features:")
print(importance_df.head())

# --------------------------------------------------
# 8. SAVE MODEL ARTIFACTS
# --------------------------------------------------
os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "artifacts/xgb_model_raw.pkl")
joblib.dump(scaler, "artifacts/scaler_raw.pkl")
joblib.dump(le, "artifacts/encoder_raw.pkl")
joblib.dump(features, "artifacts/feature_cols_raw.pkl")

print("\n✅ MODEL SAVED SUCCESSFULLY")
print("⚠️ Update Flask API to load: xgb_model_raw.pkl")
