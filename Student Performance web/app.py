from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("dropout_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("feature_list.pkl", "rb"))

prediction_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ===== USER INPUT =====
        sem1 = float(request.form.get("Sem1_GPA", 0))
        sem2 = float(request.form.get("Sem2_GPA", 0))
        backlogs = float(request.form.get("Backlogs", 0))
        tuition = int(request.form.get("Tuition_fees_up_to_date", 0))
        debtor = int(request.form.get("Debtor", 0))
        scholarship = int(request.form.get("Scholarship_holder", 0))
        age = float(request.form.get("Age_at_enrollment", 20))
        gender = int(request.form.get("Gender", 0))
        application_mode = int(request.form.get("Application_mode", 0))
        course = int(request.form.get("Course", 0))

        # ===== FEATURE ENGINEERING =====
        first_grade = sem1 * 10
        second_grade = sem2 * 10
        sgpa = (first_grade + second_grade) / 20

        success_rate_1st = sem1 / (sem1 + 1)
        success_rate_2nd = sem2 / (sem2 + 1)

        academic_risk = backlogs   # BACKLOG IMPACT

        input_dict = {
            "curricular_units_2nd_sem_approved": sem2,
            "curricular_units_2nd_sem_grade": second_grade,
            "curricular_units_1st_sem_approved": sem1,
            "curricular_units_1st_sem_grade": first_grade,
            "sgpa": sgpa,
            "academic_risk": academic_risk,
            "success_rate_1st": success_rate_1st,
            "success_rate_2nd": success_rate_2nd,
            "tuition_fees_up_to_date": tuition,
            "debtor": debtor,
            "scholarship_holder": scholarship,
            "age_at_enrollment": age,
            "gender": gender,
            "application_mode": application_mode,
            "course": course
        }

        input_df = pd.DataFrame([input_dict])

        # AUTO-SYNC FEATURES
        for f in features:
            if f not in input_df.columns:
                input_df[f] = 0

        input_df = input_df[features]

        # SCALE + PREDICT
        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if probability >= 70:
            risk = "High"
            color = "danger"
        elif probability >= 40:
            risk = "Medium"
            color = "warning"
        else:
            risk = "Low"
            color = "success"

        prediction_history.append(risk)

        return render_template(
            "result.html",
            dropout=risk,
            probability=round(probability, 2),
            color=color,
            backlogs=int(backlogs),
            sem1=round(sem1, 2),
            sem2=round(sem2, 2),
            improvement=round(sem2 - sem1, 2)
        )

    except Exception as e:
        return f"Prediction Error: {e}"

@app.route("/dashboard")
def dashboard():
    total = len(prediction_history)
    high = prediction_history.count("High")
    medium = prediction_history.count("Medium")
    low = prediction_history.count("Low")

    return render_template(
        "dashboard.html",
        total=total,
        high=high,
        medium=medium,
        low=low
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    