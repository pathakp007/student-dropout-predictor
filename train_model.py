import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data.csv", sep=";")

df.columns = (
    df.columns.str.strip()
    .str.replace('"', '')
    .str.replace("'", "")
    .str.replace(" ", "_")
    .str.replace(r"[()]", "", regex=True)
    .str.lower()
)

# ==============================
# TARGET
# ==============================
df["target"] = df["target"].map({
    "Dropout": 1,
    "Graduate": 0,
    "Enrolled": 0
})

y = df["target"].fillna(0).astype(int)

# ==============================
# FEATURE ENGINEERING
# ==============================
df["sgpa"] = (
    df["curricular_units_1st_sem_grade"] +
    df["curricular_units_2nd_sem_grade"]
) / 20

df["academic_risk"] = (
    df["curricular_units_1st_sem_without_evaluations"] +
    df["curricular_units_2nd_sem_without_evaluations"]
)

df["success_rate_1st"] = (
    df["curricular_units_1st_sem_approved"] /
    (df["curricular_units_1st_sem_enrolled"] + 1)
)

df["success_rate_2nd"] = (
    df["curricular_units_2nd_sem_approved"] /
    (df["curricular_units_2nd_sem_enrolled"] + 1)
)

df["performance_gap"] = abs(
    df["curricular_units_1st_sem_grade"] -
    df["curricular_units_2nd_sem_grade"]
)

df["total_approved"] = (
    df["curricular_units_1st_sem_approved"] +
    df["curricular_units_2nd_sem_approved"]
)

# ==============================
# FEATURES
# ==============================
features = [
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "sgpa",
    "academic_risk",
    "success_rate_1st",
    "success_rate_2nd",
    "performance_gap",
    "total_approved",
    "tuition_fees_up_to_date",
    "debtor",
    "scholarship_holder",
    "age_at_enrollment",
    "gender",
    "application_mode",
    "course"
]

X = df[features].fillna(0)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SMOTE
# ==============================
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# MODELS
# ==============================
models = {
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "LDA": LinearDiscriminantAnalysis(),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False
    )
}

# ==============================
# TRAIN + EVALUATE
# ==============================
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    print(f"\n{name} Results:")
    print("Accuracy:", round(accuracy, 3))
    print("Sensitivity:", round(sensitivity, 3))
    print("Specificity:", round(specificity, 3))

# ==============================
# SAVE ONLY BEST MODEL (XGBoost)
# ==============================
final_model = models["XGBoost"]

joblib.dump(final_model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(features, "features.joblib")

print("\n✅ Model saved successfully (joblib)")