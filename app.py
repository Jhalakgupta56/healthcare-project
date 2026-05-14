from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
app.secret_key = "secret123"

# DATABASE
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ================= MODELS =================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    glucose = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    age = db.Column(db.Float)
    cholesterol = db.Column(db.Float)
    result = db.Column(db.String(50))
    heart_risk = db.Column(db.String(50))
    neuro_risk = db.Column(db.String(50))

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    specialization = db.Column(db.String(100))
    hospital = db.Column(db.String(100))
    contact = db.Column(db.String(50))

# CREATE DB
with app.app_context():
    db.create_all()

    if Doctor.query.count() == 0:
        db.session.add_all([
            Doctor(name="Dr. Sharma", specialization="Diabetologist", hospital="Apollo", contact="9876543210"),
            Doctor(name="Dr. Patel", specialization="Cardiologist", hospital="Sterling", contact="9876500000"),
            Doctor(name="Dr. Mehta", specialization="Neurologist", hospital="Zydus", contact="9876511111"),
        ])
        db.session.commit()

# Dummy Image Model
image_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='softmax')
])

# ================= ROUTES =================

@app.route('/')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html', user=session['user'])

# ---------- AUTH ----------

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        user = User(
            username=request.form['username'],
            password=request.form['password']
        )
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(
            username=request.form['username'],
            password=request.form['password']
        ).first()

        if user:
            session['user'] = user.username
            return redirect('/')
        return "Invalid Login"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# ---------- PREDICTION ----------

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if 'user' not in session:
        return redirect('/login')

    glucose = float(request.form['glucose'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    age = float(request.form['age'])
    cholesterol = float(request.form['cholesterol'])

    # Diabetes Prediction (simple logic)
    prob = (glucose/200 + bmi/50) / 2
    result = "Diabetic" if prob > 0.5 else "Non-Diabetic"

    # -------- HEART RISK (UPDATED) --------
    if cholesterol > 240 or age > 60:
        heart_risk = "High Risk"
    elif cholesterol > 200 or age > 45:
        heart_risk = "Moderate Risk"
    else:
        heart_risk = "Low Risk"

    # -------- NEURO RISK (UPDATED) --------
    if glucose > 180:
        neuro_risk = "High Risk"
    elif glucose > 140 or age > 50:
        neuro_risk = "Moderate Risk"
    else:
        neuro_risk = "Low Risk"

    # Save History
    record = History(
        username=session['user'],
        glucose=glucose,
        insulin=insulin,
        bmi=bmi,
        age=age,
        cholesterol=cholesterol,
        result=result,
        heart_risk=heart_risk,
        neuro_risk=neuro_risk
    )
    db.session.add(record)
    db.session.commit()

    # Doctor Recommendation
    doctors = []
    if result == "Diabetic":
        doctors += Doctor.query.filter_by(specialization="Diabetologist").all()
    if heart_risk != "Low Risk":
        doctors += Doctor.query.filter_by(specialization="Cardiologist").all()
    if neuro_risk != "Low Risk":
        doctors += Doctor.query.filter_by(specialization="Neurologist").all()

    return render_template("index.html",
        user=session['user'],
        result=result,
        heart_risk=heart_risk,
        neuro_risk=neuro_risk,
        doctors=doctors
    )

# ---------- IMAGE ----------

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'user' not in session:
        return redirect('/login')

    file = request.files.get('image')
    if not file:
        return redirect('/')

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = image_model.predict(img)
    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    return render_template("index.html",
        user=session['user'],
        image_result=classes[np.argmax(pred)]
    )

# ---------- HISTORY ----------

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    records = History.query.filter_by(username=session['user']).all()
    return render_template('history.html', records=records)

# ---------- ANALYTICS ----------

@app.route('/analytics')
def analytics():
    if 'user' not in session:
        return redirect('/login')

    records = History.query.filter_by(username=session['user']).all()

    labels = list(range(1, len(records)+1))
    glucose = [r.glucose for r in records]
    bmi = [r.bmi for r in records]

    diabetic = sum(1 for r in records if r.result == "Diabetic")
    non_diabetic = len(records) - diabetic

    heart_counts = {
        "Low": sum(1 for r in records if r.heart_risk == "Low Risk"),
        "Moderate": sum(1 for r in records if r.heart_risk == "Moderate Risk"),
        "High": sum(1 for r in records if r.heart_risk == "High Risk"),
    }

    neuro_counts = {
        "Low": sum(1 for r in records if r.neuro_risk == "Low Risk"),
        "Moderate": sum(1 for r in records if r.neuro_risk == "Moderate Risk"),
        "High": sum(1 for r in records if r.neuro_risk == "High Risk"),
    }

    return render_template("analytics.html",
        labels=labels,
        glucose=glucose,
        bmi=bmi,
        diabetic=diabetic,
        non_diabetic=non_diabetic,
        heart_counts=heart_counts,
        neuro_counts=neuro_counts
    )

# ---------- DOCTORS ----------

@app.route('/doctors')
def doctors():
    if 'user' not in session:
        return redirect('/login')

    doctors = Doctor.query.all()
    return render_template("doctors.html", doctors=doctors)

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)