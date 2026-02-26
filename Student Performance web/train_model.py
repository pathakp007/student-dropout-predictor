import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data.csv", sep=";")

df.columns = (
    df.columns
    .str.strip()
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

# Extra strong features (helps accuracy)
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
# SCALE
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# CLASS WEIGHT
# ==============================
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ==============================
# MODEL (OPTIMIZED)
# ==============================
model = XGBClassifier(
    n_estimators=2600,
    learning_rate=0.012,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.05,
    min_child_weight=2,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ==============================
# EVALUATE
# ==============================
pred = model.predict(X_test)
print("\n🔥 FINAL Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(y_test, pred))

# ==============================
# SAVE
# ==============================
pickle.dump(model, open("dropout_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(features, open("feature_list.pkl", "wb"))

print("\n✅ Upgraded Model Saved")