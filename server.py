"""
server.py — Flask backend for the NutriKit web interface.

Serves the HTML frontend AND handles:
  - Food CNN predictions via /predict
  - Exercise recommendations via /recommend_exercise

Usage:
    pip install flask flask-cors tensorflow pillow numpy pandas scikit-learn
    python server.py

Then open: http://127.0.0.1:5000
"""

import os
import base64
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

app = Flask(__name__, static_folder=".")
CORS(app)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "food_cnn_scratch.keras"
HTML_FILE  = "health_nutrition_toolkit.html"
EXERCISE_CSV = "gym_members_exercise_tracking.csv"
IMG_SIZE   = 128
TOP_K      = 5

FOOD_CLASSES = [
    'apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare',
    'beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito',
    'bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake',
    'ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla',
    'chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder',
    'club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes',
    'deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots',
    'falafel','filet_mignon','fish_and_chips','foie_gras','french_fries',
    'french_onion_soup','french_toast','fried_calamari','fried_rice',
    'frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich',
    'grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup',
    'hot_dog','huevos_rancheros','hummus','ice_cream','lasagna','lobster_bisque',
    'lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels',
    'nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes',
    'panna_cotta','peking_duck','pho','pizza','pork_chop','poutine','prime_rib',
    'pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa',
    'sashimi','scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese',
    'spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi',
    'tacos','takoyaki','tiramisu','tuna_tartare','waffles'
]

# ─────────────────────────────────────────────────────────────────────
# Exercise library (same as project.py)
# ─────────────────────────────────────────────────────────────────────
EXERCISE_LIBRARY = {
    "Cardio": {
        1: [
            {"name": "Brisk Walking",        "duration": 30, "calories": 150, "equipment": "Home"},
            {"name": "Light Cycling",        "duration": 30, "calories": 180, "equipment": "Home"},
            {"name": "Low Impact Aerobics",  "duration": 25, "calories": 160, "equipment": "Home"},
            {"name": "Swimming (slow)",      "duration": 20, "calories": 140, "equipment": "Home"},
            {"name": "Treadmill Walking",    "duration": 30, "calories": 170, "equipment": "Gym"},
            {"name": "Elliptical (light)",   "duration": 25, "calories": 160, "equipment": "Gym"},
            {"name": "Stationary Bike",      "duration": 30, "calories": 175, "equipment": "Gym"},
        ],
        2: [
            {"name": "Jogging",              "duration": 30, "calories": 300, "equipment": "Home"},
            {"name": "Cycling (moderate)",   "duration": 30, "calories": 260, "equipment": "Home"},
            {"name": "Jump Rope",            "duration": 20, "calories": 250, "equipment": "Home"},
            {"name": "Stair Climbing",       "duration": 25, "calories": 270, "equipment": "Gym"},
            {"name": "Rowing Machine",       "duration": 25, "calories": 280, "equipment": "Gym"},
        ],
        3: [
            {"name": "Running (6mph+)",      "duration": 30, "calories": 450, "equipment": "Home"},
            {"name": "Cycling (vigorous)",   "duration": 30, "calories": 400, "equipment": "Home"},
            {"name": "Swimming (fast)",      "duration": 30, "calories": 400, "equipment": "Home"},
            {"name": "Sprint Intervals",     "duration": 20, "calories": 380, "equipment": "Home"},
            {"name": "Jump Rope (fast)",     "duration": 20, "calories": 350, "equipment": "Home"},
        ],
    },
    "HIIT": {
        1: [
            {"name": "Modified Burpees",     "duration": 20, "calories": 200, "equipment": "Home"},
            {"name": "Jumping Jacks",        "duration": 15, "calories": 150, "equipment": "Home"},
            {"name": "Mountain Climbers",    "duration": 15, "calories": 160, "equipment": "Home"},
        ],
        2: [
            {"name": "Burpees",              "duration": 20, "calories": 280, "equipment": "Home"},
            {"name": "High Knees",           "duration": 15, "calories": 220, "equipment": "Home"},
            {"name": "Box Jumps",            "duration": 20, "calories": 260, "equipment": "Gym"},
        ],
        3: [
            {"name": "Tabata Sprints",       "duration": 20, "calories": 400, "equipment": "Home"},
            {"name": "Plyometric Circuit",   "duration": 25, "calories": 420, "equipment": "Gym"},
            {"name": "Battle Ropes",         "duration": 15, "calories": 350, "equipment": "Gym"},
        ],
    },
    "Strength": {
        1: [
            {"name": "Bodyweight Squats",    "duration": 20, "calories": 120, "equipment": "Home"},
            {"name": "Push-ups (knee)",      "duration": 15, "calories": 100, "equipment": "Home"},
            {"name": "Resistance Bands",     "duration": 25, "calories": 140, "equipment": "Home"},
            {"name": "Light Dumbbells",      "duration": 30, "calories": 160, "equipment": "Gym"},
        ],
        2: [
            {"name": "Goblet Squats",        "duration": 25, "calories": 220, "equipment": "Gym"},
            {"name": "Push-ups",             "duration": 20, "calories": 180, "equipment": "Home"},
            {"name": "Dumbbell Rows",        "duration": 25, "calories": 230, "equipment": "Gym"},
            {"name": "Lunges",               "duration": 20, "calories": 200, "equipment": "Home"},
        ],
        3: [
            {"name": "Barbell Squats",       "duration": 30, "calories": 350, "equipment": "Gym"},
            {"name": "Deadlifts",            "duration": 30, "calories": 380, "equipment": "Gym"},
            {"name": "Bench Press",          "duration": 30, "calories": 320, "equipment": "Gym"},
            {"name": "Pull-ups",             "duration": 20, "calories": 280, "equipment": "Gym"},
        ],
    },
    "Yoga": {
        1: [
            {"name": "Hatha Yoga (basic)",   "duration": 30, "calories": 120, "equipment": "Home"},
            {"name": "Gentle Stretching",    "duration": 20, "calories": 80,  "equipment": "Home"},
            {"name": "Seated Poses",         "duration": 25, "calories": 100, "equipment": "Home"},
        ],
        2: [
            {"name": "Vinyasa Flow",         "duration": 45, "calories": 220, "equipment": "Home"},
            {"name": "Sun Salutations",      "duration": 30, "calories": 180, "equipment": "Home"},
            {"name": "Power Yoga",           "duration": 40, "calories": 250, "equipment": "Home"},
        ],
        3: [
            {"name": "Ashtanga Yoga",        "duration": 60, "calories": 350, "equipment": "Home"},
            {"name": "Hot Yoga",             "duration": 60, "calories": 400, "equipment": "Gym"},
            {"name": "Advanced Inversions",  "duration": 45, "calories": 280, "equipment": "Home"},
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────
# Globals (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────
model = None                # CNN
exercise_models = None      # tuple: (rf, clf_scaler, le, cal_model, reg_scaler)


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────
def load_cnn_model():
    """Load the trained Food-101 CNN once at startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] CNN model file '{MODEL_PATH}' not found.")
        print("       Food image classification will be unavailable.")
        return
    print(f"Loading {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"CNN loaded! ({model.count_params():,} parameters)")


def load_exercise_models():
    """Train Random Forest + Linear Regression on the gym dataset."""
    global exercise_models
    if not os.path.exists(EXERCISE_CSV):
        print(f"[WARN] Exercise dataset '{EXERCISE_CSV}' not found.")
        print("       Exercise recommendations will be unavailable.")
        return

    print(f"Loading {EXERCISE_CSV} and training exercise models ...")
    ex_df = pd.read_csv(EXERCISE_CSV)
    ex_df.columns = [
        'Age', 'Gender', 'Weight', 'Height', 'Max_BPM', 'Avg_BPM',
        'Resting_BPM', 'Session_Duration', 'Calories_Burned', 'Workout_Type',
        'Fat_Percentage', 'Water_Intake', 'Workout_Frequency',
        'Experience_Level', 'BMI'
    ]
    ex_df['Gender_encoded'] = (ex_df['Gender'] == 'Male').astype(int)
    workout_le = LabelEncoder()
    ex_df['Workout_encoded'] = workout_le.fit_transform(ex_df['Workout_Type'])

    # Random Forest
    clf_features = [
        'Age', 'Gender_encoded', 'Weight', 'BMI',
        'Experience_Level', 'Workout_Frequency', 'Session_Duration'
    ]
    X_clf = ex_df[clf_features]
    y_clf = ex_df['Workout_encoded']
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    clf_scaler = StandardScaler()
    X_clf_train_s = clf_scaler.fit_transform(X_clf_train)
    X_clf_test_s = clf_scaler.transform(X_clf_test)

    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_clf_train_s, y_clf_train)
    rf_acc = accuracy_score(y_clf_test, rf_model.predict(X_clf_test_s))

    # Linear Regression
    reg_features = [
        'Age', 'Gender_encoded', 'Weight', 'BMI',
        'Session_Duration', 'Workout_Frequency', 'Experience_Level'
    ]
    X_reg = ex_df[reg_features]
    y_reg = ex_df['Calories_Burned']
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_scaler = StandardScaler()
    X_reg_train_s = reg_scaler.fit_transform(X_reg_train)
    X_reg_test_s = reg_scaler.transform(X_reg_test)

    cal_model = LinearRegression()
    cal_model.fit(X_reg_train_s, y_reg_train)
    cal_pred = cal_model.predict(X_reg_test_s)
    r2_cal = r2_score(y_reg_test, cal_pred)

    print(f"Exercise models trained! "
          f"RF accuracy: {round(rf_acc * 100, 2)}%, "
          f"LR R²: {round(r2_cal, 3)}")

    exercise_models = (rf_model, clf_scaler, workout_le, cal_model, reg_scaler)


def goal_aware_override(predicted_workout, goal, proba, workout_classes,
                        confidence_threshold=0.35):
    """Combine ML prediction with goal-based rules."""
    GOAL_PREFERENCES = {
        "loss":     ["Cardio", "HIIT"],
        "gain":     ["Strength"],
        "maintain": ["Cardio", "HIIT", "Strength", "Yoga"],
    }
    ideal_types = GOAL_PREFERENCES.get(goal, list(workout_classes))
    proba_map = {wtype: float(p) for wtype, p in zip(workout_classes, proba)}
    top_confidence = proba_map.get(predicted_workout, 0.0)

    if predicted_workout in ideal_types:
        return predicted_workout, "ML prediction aligns with your goal"

    candidates = [(t, proba_map.get(t, 0.0)) for t in ideal_types]
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0][0]

    if top_confidence < confidence_threshold:
        return best, (
            f"ML confidence was low ({top_confidence*100:.1f}%); "
            f"using goal-based recommendation"
        )
    return best, (
        f"ML predicted {predicted_workout} but it doesn't fit your goal; "
        f"using goal-based recommendation"
    )


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    """Serve the HTML frontend."""
    return send_from_directory(".", HTML_FILE)


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve any other static file in the project folder."""
    return send_from_directory(".", filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Food image classification.
    Accept JSON: { "image": "<base64 string>" }
    Return JSON: { "predictions": [{"class": "...", "confidence": 0.xx}, ...] }
    """
    if model is None:
        return jsonify({"error": "CNN model not loaded"}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        top_indices = preds.argsort()[-TOP_K:][::-1]

        results = [
            {"class": FOOD_CLASSES[i], "confidence": round(float(preds[i]), 4)}
            for i in top_indices
        ]
        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend_exercise", methods=["POST"])
def recommend_exercise():
    """
    Exercise recommendation.
    Accept JSON: { age, gender, weight_kg, height_cm, experience, frequency, goal }
    Return JSON: { ml_workout, final_workout, confidences, est_calories,
                   override_reason, exercises, bmi }
    """
    if exercise_models is None:
        return jsonify({"error": "Exercise models not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Validate inputs
        age       = int(data.get("age", 0))
        gender    = str(data.get("gender", "")).strip()
        weight    = float(data.get("weight_kg", 0))
        height_cm = float(data.get("height_cm", 0))
        exp       = int(data.get("experience", 0))
        freq      = int(data.get("frequency", 0))
        goal      = str(data.get("goal", "")).lower().strip()

        if not (10 <= age <= 100):
            return jsonify({"error": "Age must be between 10 and 100"}), 400
        if not (25 <= weight <= 250):
            return jsonify({"error": "Weight must be between 25 and 250 kg"}), 400
        if not (100 <= height_cm <= 230):
            return jsonify({"error": "Height must be between 100 and 230 cm"}), 400
        if exp not in (1, 2, 3):
            return jsonify({"error": "Experience must be 1, 2, or 3"}), 400
        if not (1 <= freq <= 7):
            return jsonify({"error": "Sessions/week must be between 1 and 7"}), 400
        if goal not in ("loss", "gain", "maintain"):
            return jsonify({"error": "Goal must be loss, gain, or maintain"}), 400

        rf_model, clf_scaler, workout_le, cal_model, reg_scaler = exercise_models

        height_m = height_cm / 100
        bmi = round(weight / (height_m ** 2), 2)
        gender_enc = 1 if gender.lower() == 'male' else 0
        session_dur_hours = 1.0  # 1-hour session is typical

        # Random Forest prediction
        clf_columns = [
            'Age', 'Gender_encoded', 'Weight', 'BMI',
            'Experience_Level', 'Workout_Frequency', 'Session_Duration'
        ]
        sample_clf_df = pd.DataFrame(
            [[age, gender_enc, weight, bmi, exp, freq, session_dur_hours]],
            columns=clf_columns
        )
        sample_clf = clf_scaler.transform(sample_clf_df)
        predicted_workout_enc = rf_model.predict(sample_clf)[0]
        ml_predicted_workout  = workout_le.inverse_transform([predicted_workout_enc])[0]
        proba                 = rf_model.predict_proba(sample_clf)[0]

        # Goal-aware override
        final_workout, override_reason = goal_aware_override(
            ml_predicted_workout, goal, proba, workout_le.classes_
        )

        # Linear Regression: calories burned
        reg_columns = [
            'Age', 'Gender_encoded', 'Weight', 'BMI',
            'Session_Duration', 'Workout_Frequency', 'Experience_Level'
        ]
        sample_reg_df = pd.DataFrame(
            [[age, gender_enc, weight, bmi, session_dur_hours, freq, exp]],
            columns=reg_columns
        )
        sample_reg = reg_scaler.transform(sample_reg_df)
        est_calories = round(float(cal_model.predict(sample_reg)[0]), 1)

        # Build confidences as a sorted list
        confidences = [
            {"workout": str(wtype), "probability": round(float(p), 4)}
            for wtype, p in zip(workout_le.classes_, proba)
        ]
        confidences.sort(key=lambda x: -x["probability"])

        # Lookup exercises for the final workout
        exercises = []
        if final_workout in EXERCISE_LIBRARY and exp in EXERCISE_LIBRARY[final_workout]:
            exercises = EXERCISE_LIBRARY[final_workout][exp]

        return jsonify({
            "bmi":             bmi,
            "ml_workout":      ml_predicted_workout,
            "final_workout":   final_workout,
            "confidences":     confidences,
            "est_calories":    est_calories,
            "override_reason": override_reason,
            "exercises":       exercises,
            "experience":      exp,
            "goal":            goal,
            "frequency":       freq,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_cnn_model()
    load_exercise_models()
    print("\n" + "=" * 55)
    print("  NutriKit server running at http://127.0.0.1:5000")
    print("  Open that URL in your browser.")
    print("=" * 55 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
