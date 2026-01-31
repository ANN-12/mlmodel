import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# --------------------------------------------------
# 1️⃣ CREATE FOLDER FOR CONFUSION MATRICES
# --------------------------------------------------
OUTPUT_DIR = "confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2️⃣ LOAD DATA
# --------------------------------------------------
data = pd.read_csv("keystroke_data.csv", engine="python", on_bad_lines="skip")

X = data.drop("user_id", axis=1)
y = data["user_id"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------------------------------
# 3️⃣ TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# --------------------------------------------------
# 4️⃣ MODELS & PARAM GRIDS
# --------------------------------------------------
models = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        {"model__C": [0.1, 1, 10]}
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        {"model__n_neighbors": [3, 5, 7]}
    ),

    "Naive Bayes": (
        GaussianNB(),
        {}
    ),

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(),
        {"max_depth": [None, 10, 20]}
    ),

    "Random Forest": (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            objective="multi:softmax",
            eval_metric="mlogloss",
            num_class=len(le.classes_)
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    )
}

# --------------------------------------------------
# 5️⃣ TRAIN, EVALUATE & SAVE CONFUSION MATRICES
# --------------------------------------------------
for name, (model, params) in models.items():
    print(f"\n===== {name} =====")

    grid = GridSearchCV(model, params, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Best Params:", grid.best_params_)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Convert labels back to original
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    # Confusion matrix
    cm = confusion_matrix(
        y_test_labels,
        y_pred_labels,
        labels=le.classes_
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    file_path = f"{OUTPUT_DIR}/{name.replace(' ', '_')}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"Confusion matrix saved: {file_path}")

print("\n ALL CONFUSION MATRICES SAVED SUCCESSFULLY")
