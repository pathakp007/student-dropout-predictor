from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import joblib
import os

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv

# ==============================
# INIT
# ==============================
load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# ==============================
# LOGIN SYSTEM
# ==============================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

bcrypt = Bcrypt(app)
users_db = {}

class User(UserMixin):
    def __init__(self, id, name, email, password):
        self.id = id
        self.name = name
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

# ==============================
# LOAD MODEL FILES
# ==============================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
features = joblib.load(os.path.join(BASE_DIR, "features.joblib"))

# ==============================
# LOAD DATA
# ==============================
students_df = pd.read_csv("data.csv", sep=";")

students_df.columns = students_df.columns.str.strip().str.lower()
students_df.columns = students_df.columns.str.replace(" ", "_")
students_df.columns = students_df.columns.str.replace("(", "")
students_df.columns = students_df.columns.str.replace(")", "")
students_df.columns = students_df.columns.str.replace("-", "_")

# ==============================
# AUTH ROUTES
# ==============================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password_raw = request.form.get("password")

        if not name or not email or not password_raw:
            return render_template("signup.html", error="All fields required ❌")

        for u in users_db.values():
            if u.email == email:
                return render_template("signup.html", error="Email exists ❌")

        password = bcrypt.generate_password_hash(password_raw).decode("utf-8")
        user_id = str(len(users_db) + 1)
        users_db[user_id] = User(user_id, name, email, password)

        return render_template("login.html", success="Account created ✅")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        for user in users_db.values():
            if user.email == email and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect("/")

        return render_template("login.html", error="Invalid credentials ❌")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# ==============================
# HOME
# ==============================
@app.route("/")
@login_required
def home():
    return render_template("index.html", user=current_user)

# ==============================
# 🎯 PREDICT
# ==============================
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    roll_number = request.form.get("roll_number", "")

    if not roll_number.isdigit():
        return render_template("index.html", error="Invalid Roll ❌")

    index = int(roll_number) % len(students_df)
    student = students_df.iloc[index]

    sem1 = float(student.get("curricular_units_1st_sem_grade", 0))
    sem2 = float(student.get("curricular_units_2nd_sem_grade", 0))

    approved1 = float(student.get("curricular_units_1st_sem_approved", 0))
    enrolled1 = float(student.get("curricular_units_1st_sem_enrolled", 1))

    approved2 = float(student.get("curricular_units_2nd_sem_approved", 0))
    enrolled2 = float(student.get("curricular_units_2nd_sem_enrolled", 1))

    # features
    sgpa = (sem1 + sem2) / 2
    success_rate_1st = approved1 / enrolled1 if enrolled1 else 0
    success_rate_2nd = approved2 / enrolled2 if enrolled2 else 0
    performance_gap = sem2 - sem1
    total_approved = approved1 + approved2
    academic_risk = 1 if sgpa < 6 else 0

    cgpa = round(sgpa / 2, 2)

    input_dict = {}

    for col in features:
        if col == "sgpa":
            input_dict[col] = sgpa
        elif col == "success_rate_1st":
            input_dict[col] = success_rate_1st
        elif col == "success_rate_2nd":
            input_dict[col] = success_rate_2nd
        elif col == "performance_gap":
            input_dict[col] = performance_gap
        elif col == "total_approved":
            input_dict[col] = total_approved
        elif col == "academic_risk":
            input_dict[col] = academic_risk
        elif col in student:
            input_dict[col] = student[col]
        else:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)

    probability = float(model.predict_proba(input_scaled)[0][1] * 100)

    if probability >= 75:
        risk = "High"
        color = "danger"
    elif probability >= 40:
        risk = "Medium"
        color = "warning"
    else:
        risk = "Low"
        color = "success"

    return render_template(
        "result.html",
        cgpa=cgpa,
        sem1=sem1,
        sem2=sem2,
        dropout=risk,
        probability=round(probability, 2),
        roll_number=roll_number,
        attendance=70,
        backlogs=1,
        color=color
    )

# ==============================
# 📊 DASHBOARD (FIXED)
# ==============================
@app.route("/dashboard")
@login_required
def dashboard():
    df = students_df.copy()

    # same feature engineering as training
    df["sgpa"] = (
        df["curricular_units_1st_sem_grade"] +
        df["curricular_units_2nd_sem_grade"]
    ) / 2

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

    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1] * 100

    high = int((probs >= 75).sum())
    medium = int(((probs >= 40) & (probs < 75)).sum())
    low = int((probs < 40).sum())
    total = int(len(probs))

    return render_template(
        "dashboard.html",
        total=total,
        high=high,
        medium=medium,
        low=low
    )

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
