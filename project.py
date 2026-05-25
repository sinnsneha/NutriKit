"""
Health & Nutrition Toolkit
Menu-driven script combining 7 features:
  1. BMI Calculator
  2. Meal Recommender (KNN)
  3. Meal Health Classifier (Logistic Regression vs Decision Tree)
  4. Weight Predictor (Linear Regression)
  5. Mess Advisory (Daily plate planner)
  6. Food Image Classifier (CNN) with Nutrition Advice
  7. Exercise Recommendation (Random Forest + Linear Regression)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────
# 1. BMI CALCULATOR
# ─────────────────────────────────────────────────────────────────────
def calculate_bmi():
    age = int(input("Enter age: "))
    height_cm = float(input("Enter height (cm): "))
    weight = float(input("Enter weight (kg): "))

    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 24.9:
        category = "Normal"
    elif bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obese"

    print("\nSummary:")
    print(f"Age      : {age}")
    print(f"Height   : {height_cm} cm ({height_m:.2f} m)")
    print(f"Weight   : {weight} kg")
    print(f"BMI      : {round(bmi, 2)}")
    print(f"Category : {category}")


# ─────────────────────────────────────────────────────────────────────
# 2. MEAL RECOMMENDER (KNN)
# ─────────────────────────────────────────────────────────────────────
def load_meal_data(csv_path="healthy_meal_plans.csv"):
    """Load and denormalize the meal dataset."""
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()
    df['meal_name'] = df['meal_name'].str.lower()

    # Denormalize to real-world values
    df['calories'] = df['calories'] * 800
    df['protein']  = df['protein']  * 60
    df['fat']      = df['fat']      * 50
    df['carbs']    = df['carbs']    * 100

    df['calculated_calories'] = (
        df['protein'] * 4 + df['carbs'] * 4 + df['fat'] * 9
    )
    return df


def recommend_meals(df):
    goal = input("Enter your goal (loss/gain/maintain): ").lower()
    preference = input("Diet preference (vegetarian/vegan/keto/none): ").lower()
    gluten = input("Gluten free required? (yes/no): ").lower()

    filtered_df = df.copy()

    if preference != "none" and preference in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[preference] == 1]

    if gluten == "yes" and "gluten_free" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["gluten_free"] == 1]

    if filtered_df.empty:
        print("No meals found matching your filters.")
        return

    filtered_df = filtered_df.reset_index(drop=True)

    # Target: [calories, protein, fat, carbs]
    targets = {
        "loss":     [400, 35, 12, 45],
        "gain":     [750, 50, 25, 100],
        "maintain": [550, 40, 18, 70],
    }
    if goal not in targets:
        print("Invalid goal.")
        return
    target = targets[goal]

    features = ['calculated_calories', 'protein', 'fat', 'carbs']
    X = filtered_df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform([target])

    k = min(5, len(filtered_df))
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(target_scaled)

    print("\n--- Recommended Meals ---")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        meal = filtered_df.iloc[idx]
        print(f"{i+1}. {meal['meal_name'].title()}")
        print(f"   Calories: {round(meal['calculated_calories'], 1)} kcal")
        print(f"   Protein : {round(meal['protein'], 1)} g")
        print(f"   Fat     : {round(meal['fat'], 1)} g")
        print(f"   Carbs   : {round(meal['carbs'], 1)} g")
        print(f"   Match   : {round(dist, 2)}\n")


# ─────────────────────────────────────────────────────────────────────
# 3. MEAL HEALTH CLASSIFIER (LR vs Decision Tree)
# ─────────────────────────────────────────────────────────────────────
def label_health(row):
    score = 0
    if row['calories'] <= 450: score += 1
    if row['fat']      <= 25:  score += 1
    if row['protein']  >= 25:  score += 1
    if row['carbs']    <= 55:  score += 1
    return 1 if score >= 3 else 0


def train_health_classifier(df):
    df = df.copy()
    df['is_healthy'] = df.apply(label_health, axis=1)

    print("Class distribution:")
    print(df['is_healthy'].value_counts())

    X = df[['calories', 'protein', 'fat', 'carbs']]
    y = df['is_healthy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    print("\n--- Logistic Regression ---")
    print("Accuracy:", round(lr_accuracy, 4))
    print(classification_report(y_test, lr_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))

    # Cross-validation
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    cv_scores = cross_val_score(lr_pipeline, X, y, cv=5)
    print("\n--- Cross Validation (5-Fold) ---")
    print("Scores       :", [round(s, 4) for s in cv_scores])
    print("Mean Accuracy:", round(cv_scores.mean(), 4))
    print("Std Dev      :", round(cv_scores.std(), 4))

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dt_model.fit(X_train_scaled, y_train)
    dt_pred = dt_model.predict(X_test_scaled)
    dt_accuracy = accuracy_score(y_test, dt_pred)

    print("\n--- Decision Tree ---")
    print("Accuracy:", round(dt_accuracy, 4))
    print(classification_report(y_test, dt_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, dt_pred))

    print("\n--- Model Comparison ---")
    print(f"Logistic Regression: {round(lr_accuracy, 4)}")
    print(f"Decision Tree      : {round(dt_accuracy, 4)}")

    if lr_accuracy >= dt_accuracy:
        print(">> Logistic Regression selected as better model")
        best_model = lr_model
    else:
        print(">> Decision Tree selected as better model")
        best_model = dt_model

    return best_model, scaler


def check_meal_health(model, scaler):
    print("\nEnter meal nutritional values:")
    calories = float(input("Calories (kcal): "))
    protein  = float(input("Protein (g): "))
    fat      = float(input("Fat (g): "))
    carbs    = float(input("Carbs (g): "))

    sample = pd.DataFrame(
        [[calories, protein, fat, carbs]],
        columns=['calories', 'protein', 'fat', 'carbs']
    )
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    print("\n--- Meal Health Result ---")
    if prediction == 1:
        print("✅ This meal is HEALTHY")
    else:
        print("❌ This meal is NOT HEALTHY")


# ─────────────────────────────────────────────────────────────────────
# 4. WEIGHT PREDICTOR (Linear Regression)
# ─────────────────────────────────────────────────────────────────────
def predict_weight():
    current_weight  = float(input("Enter current weight (kg): "))
    daily_calories  = float(input("Enter daily calorie intake: "))
    activity_hours  = float(input("Enter daily active hours: "))
    goal            = input("Goal (loss/gain/maintain): ").lower()
    days            = int(input("Enter number of days: "))

    # Synthetic training data
    np.random.seed(42)
    n = 300
    syn_calories  = np.random.randint(1200, 3500, n)
    syn_activity  = np.random.uniform(0.5, 4.0, n)
    syn_days      = np.random.randint(7, 180, n)
    syn_start_wt  = np.random.uniform(45, 120, n)

    burned        = syn_activity * 200
    net           = syn_calories - burned
    weight_change = (net * syn_days) / 7700
    syn_final_wt  = syn_start_wt + weight_change + np.random.normal(0, 0.5, n)

    X = np.column_stack([syn_calories, syn_activity, syn_days, syn_start_wt])
    y = syn_final_wt
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2   = r2_score(y_test, y_pred_test)
    print(f"\nModel Evaluation > R2: {round(r2, 4)} | RMSE: {round(rmse, 4)} kg")

    # Adjust calories per goal
    if goal == "loss":
        adjusted = daily_calories - 300
    elif goal == "gain":
        adjusted = daily_calories + 300
    else:
        adjusted = daily_calories

    weights = []
    calorie_balance = []
    for day in range(1, days + 1):
        x_day = np.array([[adjusted, activity_hours, day, current_weight]])
        weights.append(model.predict(x_day)[0])
        calorie_balance.append(adjusted - activity_hours * 200)

    final_weight = weights[-1]

    # Plot weight over time
    plt.figure()
    plt.plot(range(1, days + 1), weights, color='blue', label='Predicted Weight')
    plt.axhline(y=current_weight, color='red', linestyle='--', label='Starting Weight')
    plt.xlabel("Days")
    plt.ylabel("Weight (kg)")
    plt.title("Weight Prediction Over Time (Linear Regression)")
    plt.legend()
    plt.show()

    # Plot calorie balance
    plt.figure()
    plt.plot(range(1, days + 1), calorie_balance, color='green')
    plt.xlabel("Days")
    plt.ylabel("Net Calories")
    plt.title("Daily Calorie Balance")
    plt.show()

    print(f"\nFinal Predicted Weight: {round(final_weight, 2)} kg")
    print(f"Expected Change: {round(final_weight - current_weight, 2)} kg")

    if goal == "loss" and final_weight >= current_weight:
        print("\nYour weight is not decreasing as expected.")
        print("Suggestions:")
        print("- Reduce daily calorie intake by 300–500 kcal")
        print("- Increase activity time by 30–60 minutes")
    elif goal == "gain" and final_weight <= current_weight:
        print("\nYour weight is not increasing as expected.")
        print("Suggestions:")
        print("- Increase calorie intake by 300–500 kcal")
        print("- Reduce excessive physical activity")
    else:
        print("\nYour current plan aligns with your goal. Keep going!")


# ─────────────────────────────────────────────────────────────────────
# 5. MESS ADVISORY
# ─────────────────────────────────────────────────────────────────────
MESS_MENU = {
    "monday":    ["plain rice", "arhar dal", "kadhai paneer", "chapati", "salad"],
    "tuesday":   ["jeera rice", "rajmah", "cabbage aloo", "chapati", "sprout salad"],
    "wednesday": ["plain rice", "kadhi pakoda", "aloo", "chapati", "salad"],
    "thursday":  ["peas rice", "chole", "beans aloo", "bhature", "onion"],
    "friday":    ["jeera rice", "dal makhani", "mix veg", "chapati", "salad"],
    "saturday":  ["plain rice", "black chana", "sita fal", "poori", "salad"],
}

NUTRITION_MAP = {
    "plain rice":    {"cal": 200, "protein": 4,  "carbs": 45, "fat": 1},
    "jeera rice":    {"cal": 220, "protein": 4,  "carbs": 46, "fat": 3},
    "peas rice":     {"cal": 230, "protein": 6,  "carbs": 45, "fat": 3},
    "arhar dal":     {"cal": 150, "protein": 8,  "carbs": 20, "fat": 3},
    "rajmah":        {"cal": 200, "protein": 10, "carbs": 30, "fat": 4},
    "kadhi pakoda":  {"cal": 250, "protein": 7,  "carbs": 25, "fat": 10},
    "chole":         {"cal": 280, "protein": 12, "carbs": 35, "fat": 8},
    "dal makhani":   {"cal": 300, "protein": 10, "carbs": 30, "fat": 12},
    "black chana":   {"cal": 220, "protein": 11, "carbs": 30, "fat": 5},
    "kadhai paneer": {"cal": 320, "protein": 14, "carbs": 10, "fat": 20},
    "mix veg":       {"cal": 150, "protein": 4,  "carbs": 20, "fat": 5},
    "cabbage aloo":  {"cal": 180, "protein": 3,  "carbs": 25, "fat": 6},
    "beans aloo":    {"cal": 170, "protein": 3,  "carbs": 24, "fat": 5},
    "aloo":          {"cal": 160, "protein": 3,  "carbs": 25, "fat": 5},
    "sita fal":      {"cal": 130, "protein": 2,  "carbs": 28, "fat": 1},
    "chapati":       {"cal": 100, "protein": 3,  "carbs": 20, "fat": 1},
    "bhature":       {"cal": 300, "protein": 6,  "carbs": 40, "fat": 15},
    "poori":         {"cal": 250, "protein": 5,  "carbs": 35, "fat": 12},
    "salad":         {"cal": 50,  "protein": 2,  "carbs": 10, "fat": 0},
    "sprout salad":  {"cal": 80,  "protein": 6,  "carbs": 12, "fat": 1},
    "onion":         {"cal": 30,  "protein": 1,  "carbs": 7,  "fat": 0},
}


def mess_advisory():
    day  = input("Enter day: ").lower()
    goal = input("Enter goal (loss/gain/maintain): ").lower()

    if day not in MESS_MENU:
        print("Invalid day")
        return

    if goal == "loss":
        factor, portion = 0.7, "small portion"
    elif goal == "gain":
        factor, portion = 1.3, "large portion"
    else:
        factor, portion = 1.0, "medium portion"

    total_cal = total_protein = total_carbs = total_fat = 0

    print("\nRecommended Plate:\n")
    for item in MESS_MENU[day]:
        data = NUTRITION_MAP.get(item)
        if not data:
            print(f"({item} → no nutrition data, skipped)")
            continue

        cal     = data["cal"]     * factor
        protein = data["protein"] * factor
        carbs   = data["carbs"]   * factor
        fat     = data["fat"]     * factor

        total_cal     += cal
        total_protein += protein
        total_carbs   += carbs
        total_fat     += fat

        print(f"{item} → {portion}")
        print(f"  Calories: {int(cal)} kcal | Protein: {round(protein, 1)} g | "
              f"Carbs: {round(carbs, 1)} g | Fat: {round(fat, 1)} g")

    print("\nTOTAL NUTRITION:")
    print(f"Calories: {int(total_cal)} kcal")
    print(f"Protein : {round(total_protein, 1)} g")
    print(f"Carbs   : {round(total_carbs, 1)} g")
    print(f"Fat     : {round(total_fat, 1)} g")

    values = [total_protein, total_carbs, total_fat]
    if sum(values) == 0:
        print("No data to display chart")
        return

    plt.figure()
    plt.pie(values, labels=['Protein', 'Carbs', 'Fat'], autopct='%1.1f%%')
    plt.title("Macronutrient Distribution")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# 6. FOOD IMAGE CLASSIFIER (CNN) — WITH NUTRITION ADVICE
# ─────────────────────────────────────────────────────────────────────
FOOD_MODEL_PATH = "food_cnn_scratch.keras"
FOOD_IMG_SIZE = 128

FOOD_CLASS_NAMES = [
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

# Nutrition data per typical serving (calories, protein, carbs, fat in grams)
FOOD_NUTRITION = {
    'apple_pie':              {'cal': 411, 'protein': 4,  'carbs': 58, 'fat': 19},
    'baby_back_ribs':         {'cal': 540, 'protein': 38, 'carbs': 5,  'fat': 41},
    'baklava':                {'cal': 334, 'protein': 5,  'carbs': 29, 'fat': 23},
    'beef_carpaccio':         {'cal': 220, 'protein': 22, 'carbs': 2,  'fat': 14},
    'beef_tartare':           {'cal': 250, 'protein': 23, 'carbs': 3,  'fat': 16},
    'beet_salad':             {'cal': 150, 'protein': 4,  'carbs': 18, 'fat': 8},
    'beignets':               {'cal': 380, 'protein': 5,  'carbs': 45, 'fat': 20},
    'bibimbap':               {'cal': 490, 'protein': 22, 'carbs': 65, 'fat': 14},
    'bread_pudding':          {'cal': 310, 'protein': 7,  'carbs': 45, 'fat': 12},
    'breakfast_burrito':      {'cal': 470, 'protein': 20, 'carbs': 40, 'fat': 25},
    'bruschetta':             {'cal': 195, 'protein': 5,  'carbs': 24, 'fat': 9},
    'caesar_salad':           {'cal': 320, 'protein': 9,  'carbs': 12, 'fat': 27},
    'cannoli':                {'cal': 330, 'protein': 7,  'carbs': 30, 'fat': 21},
    'caprese_salad':          {'cal': 280, 'protein': 14, 'carbs': 7,  'fat': 22},
    'carrot_cake':            {'cal': 410, 'protein': 5,  'carbs': 55, 'fat': 21},
    'ceviche':                {'cal': 180, 'protein': 24, 'carbs': 8,  'fat': 5},
    'cheesecake':             {'cal': 401, 'protein': 7,  'carbs': 33, 'fat': 28},
    'cheese_plate':           {'cal': 450, 'protein': 22, 'carbs': 6,  'fat': 38},
    'chicken_curry':          {'cal': 380, 'protein': 28, 'carbs': 20, 'fat': 22},
    'chicken_quesadilla':     {'cal': 530, 'protein': 28, 'carbs': 38, 'fat': 28},
    'chicken_wings':          {'cal': 430, 'protein': 30, 'carbs': 5,  'fat': 32},
    'chocolate_cake':         {'cal': 425, 'protein': 5,  'carbs': 60, 'fat': 20},
    'chocolate_mousse':       {'cal': 355, 'protein': 5,  'carbs': 35, 'fat': 22},
    'churros':                {'cal': 280, 'protein': 4,  'carbs': 35, 'fat': 14},
    'clam_chowder':           {'cal': 200, 'protein': 9,  'carbs': 18, 'fat': 11},
    'club_sandwich':          {'cal': 590, 'protein': 32, 'carbs': 45, 'fat': 30},
    'crab_cakes':             {'cal': 290, 'protein': 18, 'carbs': 14, 'fat': 18},
    'creme_brulee':           {'cal': 365, 'protein': 5,  'carbs': 28, 'fat': 26},
    'croque_madame':          {'cal': 545, 'protein': 28, 'carbs': 35, 'fat': 32},
    'cup_cakes':              {'cal': 305, 'protein': 3,  'carbs': 45, 'fat': 13},
    'deviled_eggs':           {'cal': 145, 'protein': 6,  'carbs': 1,  'fat': 13},
    'donuts':                 {'cal': 290, 'protein': 4,  'carbs': 33, 'fat': 16},
    'dumplings':              {'cal': 250, 'protein': 11, 'carbs': 30, 'fat': 9},
    'edamame':                {'cal': 190, 'protein': 17, 'carbs': 15, 'fat': 8},
    'eggs_benedict':          {'cal': 470, 'protein': 23, 'carbs': 25, 'fat': 30},
    'escargots':              {'cal': 175, 'protein': 16, 'carbs': 3,  'fat': 11},
    'falafel':                {'cal': 333, 'protein': 13, 'carbs': 32, 'fat': 18},
    'filet_mignon':           {'cal': 350, 'protein': 38, 'carbs': 0,  'fat': 22},
    'fish_and_chips':         {'cal': 595, 'protein': 28, 'carbs': 50, 'fat': 32},
    'foie_gras':              {'cal': 460, 'protein': 12, 'carbs': 5,  'fat': 44},
    'french_fries':           {'cal': 365, 'protein': 4,  'carbs': 48, 'fat': 17},
    'french_onion_soup':      {'cal': 290, 'protein': 12, 'carbs': 25, 'fat': 16},
    'french_toast':           {'cal': 290, 'protein': 9,  'carbs': 36, 'fat': 13},
    'fried_calamari':         {'cal': 310, 'protein': 18, 'carbs': 22, 'fat': 16},
    'fried_rice':             {'cal': 380, 'protein': 10, 'carbs': 55, 'fat': 13},
    'frozen_yogurt':          {'cal': 220, 'protein': 6,  'carbs': 38, 'fat': 5},
    'garlic_bread':           {'cal': 200, 'protein': 4,  'carbs': 24, 'fat': 10},
    'gnocchi':                {'cal': 250, 'protein': 7,  'carbs': 45, 'fat': 5},
    'greek_salad':            {'cal': 230, 'protein': 7,  'carbs': 12, 'fat': 18},
    'grilled_cheese_sandwich':{'cal': 400, 'protein': 17, 'carbs': 30, 'fat': 24},
    'grilled_salmon':         {'cal': 367, 'protein': 39, 'carbs': 0,  'fat': 22},
    'guacamole':              {'cal': 230, 'protein': 3,  'carbs': 12, 'fat': 21},
    'gyoza':                  {'cal': 240, 'protein': 10, 'carbs': 28, 'fat': 10},
    'hamburger':              {'cal': 540, 'protein': 25, 'carbs': 40, 'fat': 30},
    'hot_and_sour_soup':      {'cal': 160, 'protein': 9,  'carbs': 14, 'fat': 7},
    'hot_dog':                {'cal': 290, 'protein': 10, 'carbs': 23, 'fat': 17},
    'huevos_rancheros':       {'cal': 410, 'protein': 18, 'carbs': 35, 'fat': 22},
    'hummus':                 {'cal': 170, 'protein': 6,  'carbs': 14, 'fat': 10},
    'ice_cream':              {'cal': 270, 'protein': 5,  'carbs': 31, 'fat': 14},
    'lasagna':                {'cal': 450, 'protein': 24, 'carbs': 38, 'fat': 22},
    'lobster_bisque':         {'cal': 230, 'protein': 11, 'carbs': 14, 'fat': 15},
    'lobster_roll_sandwich':  {'cal': 460, 'protein': 22, 'carbs': 38, 'fat': 24},
    'macaroni_and_cheese':    {'cal': 410, 'protein': 16, 'carbs': 42, 'fat': 20},
    'macarons':               {'cal': 100, 'protein': 1,  'carbs': 14, 'fat': 4},
    'miso_soup':              {'cal': 80,  'protein': 6,  'carbs': 8,  'fat': 3},
    'mussels':                {'cal': 220, 'protein': 24, 'carbs': 9,  'fat': 9},
    'nachos':                 {'cal': 565, 'protein': 18, 'carbs': 50, 'fat': 32},
    'omelette':               {'cal': 290, 'protein': 18, 'carbs': 4,  'fat': 22},
    'onion_rings':            {'cal': 410, 'protein': 5,  'carbs': 45, 'fat': 23},
    'oysters':                {'cal': 80,  'protein': 9,  'carbs': 4,  'fat': 3},
    'pad_thai':               {'cal': 470, 'protein': 18, 'carbs': 60, 'fat': 18},
    'paella':                 {'cal': 470, 'protein': 24, 'carbs': 55, 'fat': 16},
    'pancakes':               {'cal': 350, 'protein': 8,  'carbs': 50, 'fat': 13},
    'panna_cotta':            {'cal': 295, 'protein': 4,  'carbs': 22, 'fat': 22},
    'peking_duck':            {'cal': 415, 'protein': 22, 'carbs': 8,  'fat': 33},
    'pho':                    {'cal': 350, 'protein': 22, 'carbs': 45, 'fat': 9},
    'pizza':                  {'cal': 285, 'protein': 12, 'carbs': 36, 'fat': 10},
    'pork_chop':              {'cal': 320, 'protein': 32, 'carbs': 0,  'fat': 21},
    'poutine':                {'cal': 740, 'protein': 19, 'carbs': 70, 'fat': 41},
    'prime_rib':              {'cal': 580, 'protein': 38, 'carbs': 0,  'fat': 47},
    'pulled_pork_sandwich':   {'cal': 540, 'protein': 28, 'carbs': 45, 'fat': 26},
    'ramen':                  {'cal': 450, 'protein': 18, 'carbs': 60, 'fat': 16},
    'ravioli':                {'cal': 340, 'protein': 14, 'carbs': 45, 'fat': 12},
    'red_velvet_cake':        {'cal': 420, 'protein': 4,  'carbs': 55, 'fat': 22},
    'risotto':                {'cal': 380, 'protein': 9,  'carbs': 50, 'fat': 15},
    'samosa':                 {'cal': 260, 'protein': 5,  'carbs': 30, 'fat': 14},
    'sashimi':                {'cal': 130, 'protein': 22, 'carbs': 0,  'fat': 4},
    'scallops':               {'cal': 140, 'protein': 20, 'carbs': 5,  'fat': 4},
    'seaweed_salad':          {'cal': 105, 'protein': 2,  'carbs': 9,  'fat': 7},
    'shrimp_and_grits':       {'cal': 420, 'protein': 22, 'carbs': 35, 'fat': 22},
    'spaghetti_bolognese':    {'cal': 450, 'protein': 22, 'carbs': 55, 'fat': 16},
    'spaghetti_carbonara':    {'cal': 580, 'protein': 22, 'carbs': 60, 'fat': 28},
    'spring_rolls':           {'cal': 220, 'protein': 6,  'carbs': 28, 'fat': 10},
    'steak':                  {'cal': 460, 'protein': 42, 'carbs': 0,  'fat': 32},
    'strawberry_shortcake':   {'cal': 350, 'protein': 4,  'carbs': 48, 'fat': 16},
    'sushi':                  {'cal': 220, 'protein': 9,  'carbs': 38, 'fat': 4},
    'tacos':                  {'cal': 220, 'protein': 12, 'carbs': 20, 'fat': 11},
    'takoyaki':               {'cal': 290, 'protein': 11, 'carbs': 32, 'fat': 13},
    'tiramisu':               {'cal': 360, 'protein': 6,  'carbs': 35, 'fat': 22},
    'tuna_tartare':           {'cal': 220, 'protein': 24, 'carbs': 4,  'fat': 12},
    'waffles':                {'cal': 310, 'protein': 8,  'carbs': 38, 'fat': 14},
}


def get_food_advice(nutrition):
    """
    Decide EAT / MODERATE / AVOID based on nutritional values.
    Penalty-based scoring: 2+ red flags (high cal, high fat, high carbs)
    push verdict to AVOID, even if protein passes.
    """
    cal     = nutrition['cal']
    protein = nutrition['protein']
    carbs   = nutrition['carbs']
    fat     = nutrition['fat']

    score = 0       # green checks (passed)
    penalty = 0     # red flags (failed badly)
    reasons = []

    # Calorie check
    if cal <= 350:
        score += 1
        reasons.append(f"  ✓ Calories OK ({cal} kcal)")
    elif cal <= 500:
        reasons.append(f"  ⚠ Calories moderate ({cal} kcal)")
    else:
        penalty += 1
        reasons.append(f"  ✗ High calories ({cal} kcal)")

    # Fat check
    if fat <= 15:
        score += 1
        reasons.append(f"  ✓ Fat OK ({fat} g)")
    elif fat <= 25:
        reasons.append(f"  ⚠ Fat moderate ({fat} g)")
    else:
        penalty += 1
        reasons.append(f"  ✗ High fat ({fat} g)")

    # Protein check
    if protein >= 20:
        score += 1
        reasons.append(f"  ✓ Good protein ({protein} g)")
    elif protein >= 10:
        reasons.append(f"  ⚠ Moderate protein ({protein} g)")
    else:
        reasons.append(f"  ✗ Low protein ({protein} g)")

    # Carbs check
    if carbs <= 40:
        score += 1
        reasons.append(f"  ✓ Carbs OK ({carbs} g)")
    elif carbs <= 55:
        reasons.append(f"  ⚠ Carbs moderate ({carbs} g)")
    else:
        penalty += 1
        reasons.append(f"  ✗ High carbs ({carbs} g)")

    # Verdict — penalties take priority over score
    if penalty >= 2:
        verdict = "AVOID"
        emoji   = "❌"
        message = "AVOID or eat rarely — not the healthiest pick."
    elif score >= 3 and penalty == 0:
        verdict = "EAT"
        emoji   = "✅"
        message = "This food is HEALTHY — good choice!"
    elif penalty == 1 and score <= 2:
        verdict = "AVOID"
        emoji   = "❌"
        message = "AVOID or eat rarely — not the healthiest pick."
    else:
        verdict = "MODERATE"
        emoji   = "⚠️"
        message = "Eat in MODERATION — okay occasionally."

    return verdict, emoji, message, reasons, score


def load_food_model():
    """Load the trained Food-101 CNN once, cache it."""
    if not os.path.exists(FOOD_MODEL_PATH):
        print(f"Model file '{FOOD_MODEL_PATH}' not found in current folder.")
        return None
    print("Loading food classifier model...")
    model = tf.keras.models.load_model(FOOD_MODEL_PATH)
    print(f"Model loaded! ({model.count_params():,} parameters)")
    return model


def predict_food_image(model, image_path, top_k=5):
    """Predict food category, show nutrition values, and give eat/avoid advice."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Preprocess (must match training pipeline)
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((FOOD_IMG_SIZE, FOOD_IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    top_class_key = FOOD_CLASS_NAMES[top_indices[0]]
    top_name      = top_class_key.replace('_', ' ').title()
    top_conf      = preds[top_indices[0]] * 100

    # Get nutrition data
    nutrition = FOOD_NUTRITION.get(top_class_key)

    # Show image with prediction + verdict in title
    if nutrition:
        verdict, emoji, message, reasons, score = get_food_advice(nutrition)
        title = f"Prediction: {top_name} ({top_conf:.1f}%)\n{emoji} Verdict: {verdict}"
    else:
        title = f"Prediction: {top_name} ({top_conf:.1f}%)"

    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=12)
    plt.tight_layout()
    # Note: plt.show() moved to end so text prints before image window blocks

    # Print prediction header (single line with verdict)
    if nutrition:
        print(f"\nPrediction: {top_name} ({top_conf:.1f}%)   {emoji} Verdict: {verdict}")
    else:
        print(f"\nPrediction: {top_name} ({top_conf:.1f}%)")
        print(f"No nutrition data available for '{top_name}'.")
        return

    # Nutrition values
    print(f"NUTRITIONAL VALUES — {top_name} (per serving)")
    print(f"  Calories : {nutrition['cal']} kcal")
    print(f"  Protein  : {nutrition['protein']} g")
    print(f"  Carbs    : {nutrition['carbs']} g")
    print(f"  Fat      : {nutrition['fat']} g")

    # Health advice
    print("HEALTH ADVICE")
    print(f"  {emoji} VERDICT: {verdict} ({score}/4 health checks passed)")
    print(f"  {message}")
    print("  Breakdown:")
    for r in reasons:
        print(r)

    # Goal-based suggestions
    print("GOAL-BASED SUGGESTIONS")
    if nutrition['cal'] > 500:
        print("  • For weight loss: cut portion size in half, or skip this meal.")
    elif nutrition['cal'] < 300:
        print("  • For weight gain: pair with a side (rice/bread) for more calories.")
    else:
        print("  • Calories suit a maintenance plan — fits a balanced day.")

    if nutrition['protein'] >= 20:
        print("  • High protein — great post-workout option.")
    elif nutrition['protein'] < 10:
        print("  • Low protein — add eggs, dal, paneer, or chicken to balance.")

    if nutrition['fat'] > 25:
        print("  • High fat — keep your other meals today lean.")
    print()

    # Now show the image (blocking — user closes window to continue)
    plt.show()


def food_classifier_menu(model):
    """Prompt for image path and run prediction with nutrition advice."""
    if model is None:
        print("Cannot run — model not loaded.")
        return
    image_path = input("Enter path to food image: ").strip().strip('"').strip("'")
    predict_food_image(model, image_path)


# ─────────────────────────────────────────────────────────────────────
# 7. EXERCISE RECOMMENDATION (Random Forest + Linear Regression)
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


def train_exercise_models(csv_path="gym_members_exercise_tracking.csv"):
    """
    Train Random Forest (workout type classifier) and Linear Regression
    (calories burned predictor) on the gym members dataset.
    Returns: (rf_model, clf_scaler, workout_le, cal_model, reg_scaler)
    """
    if not os.path.exists(csv_path):
        print(f"Dataset file '{csv_path}' not found in current folder.")
        return None

    print("Loading exercise dataset and training models...")
    ex_df = pd.read_csv(csv_path)

    # Standardise column names
    ex_df.columns = [
        'Age', 'Gender', 'Weight', 'Height', 'Max_BPM', 'Avg_BPM',
        'Resting_BPM', 'Session_Duration', 'Calories_Burned', 'Workout_Type',
        'Fat_Percentage', 'Water_Intake', 'Workout_Frequency',
        'Experience_Level', 'BMI'
    ]

    # Encode categorical variables
    ex_df['Gender_encoded'] = (ex_df['Gender'] == 'Male').astype(int)
    workout_le = LabelEncoder()
    ex_df['Workout_encoded'] = workout_le.fit_transform(ex_df['Workout_Type'])

    print(f"Dataset loaded! Total records: {len(ex_df)}")
    print(f"Workout types  : {list(workout_le.classes_)}")
    print(f"Experience lvls: 1=Beginner, 2=Intermediate, 3=Advanced")

    # ── Random Forest: Workout type classification ──────────────
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
    rf_pred = rf_model.predict(X_clf_test_s)
    rf_acc = accuracy_score(y_clf_test, rf_pred)

    print(f"\n--- Random Forest: Workout Type Classification ---")
    print(f"Accuracy : {round(rf_acc * 100, 2)}%")
    print("Classification Report:")
    print(classification_report(
        y_clf_test, rf_pred, target_names=workout_le.classes_, zero_division=0
    ))

    cv_rf = cross_val_score(rf_model, clf_scaler.fit_transform(X_clf), y_clf, cv=5)
    print(f"Cross Validation (5-Fold) Mean Accuracy: {round(cv_rf.mean() * 100, 2)}%")

    # ── Linear Regression: Calories burned per session ──────────
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
    rmse_cal = np.sqrt(mean_squared_error(y_reg_test, cal_pred))

    print(f"\n--- Linear Regression: Calories Burned Prediction ---")
    print(f"R² Score : {round(r2_cal, 4)}")
    print(f"RMSE     : {round(rmse_cal, 2)} kcal")

    return rf_model, clf_scaler, workout_le, cal_model, reg_scaler


def goal_aware_override(predicted_workout, goal, proba, workout_classes,
                        confidence_threshold=0.35):
    """
    Decide the FINAL workout recommendation by combining the ML prediction
    with goal-based rules. Returns (final_workout, override_reason).

    Logic:
      - If goal is loss → ideal = Cardio or HIIT
      - If goal is gain → ideal = Strength
      - If goal is maintain → any workout is fine

      - If model's prediction matches an ideal type → keep it (agreement)
      - If model's confidence < threshold OR prediction conflicts with goal,
        override with the best goal-fit type that has the highest model probability
    """
    GOAL_PREFERENCES = {
        "loss":     ["Cardio", "HIIT"],
        "gain":     ["Strength"],
        "maintain": ["Cardio", "HIIT", "Strength", "Yoga"],
    }

    ideal_types = GOAL_PREFERENCES.get(goal, list(workout_classes))

    # Map class name → its probability
    proba_map = {wtype: float(p) for wtype, p in zip(workout_classes, proba)}
    top_confidence = proba_map.get(predicted_workout, 0.0)

    # Case 1: model agrees with goal
    if predicted_workout in ideal_types:
        return predicted_workout, "ML prediction aligns with your goal"

    # Case 2: model confidence is low — trust the goal more
    if top_confidence < confidence_threshold:
        # Pick the goal-ideal workout with highest model probability
        candidates = [(t, proba_map.get(t, 0.0)) for t in ideal_types]
        candidates.sort(key=lambda x: -x[1])
        best = candidates[0][0]
        return best, (
            f"ML confidence was low ({top_confidence*100:.1f}%); "
            f"using goal-based recommendation"
        )

    # Case 3: model confident but conflicts with goal — override
    candidates = [(t, proba_map.get(t, 0.0)) for t in ideal_types]
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0][0]
    return best, (
        f"ML predicted {predicted_workout} but it doesn't fit your goal; "
        f"using goal-based recommendation"
    )


def recommend_exercise(rf_model, clf_scaler, workout_le, cal_model, reg_scaler):
    """Interactive exercise recommendation using the trained models."""
    print("\n===== Exercise Recommendation System =====")

    # Validated inputs
    age = int(input("Enter your age: "))
    if not (10 <= age <= 100):
        print(f"Error: age {age} looks unrealistic (expected 10-100). Aborting.")
        return

    gender = input("Enter gender (Male/Female): ").strip()

    weight = float(input("Enter weight (kg): "))
    if not (25 <= weight <= 250):
        print(f"Error: weight {weight} kg looks unrealistic (expected 25-250). Aborting.")
        return

    height_cm = float(input("Enter height (cm): "))
    if not (100 <= height_cm <= 230):
        print(f"Error: height {height_cm} cm looks unrealistic (expected 100-230). "
              f"Did you enter meters by mistake? Aborting.")
        return
    height_m = height_cm / 100

    exp = int(input("Experience Level (1=Beginner, 2=Intermediate, 3=Advanced): "))
    if exp not in (1, 2, 3):
        print(f"Error: experience level must be 1, 2, or 3. Aborting.")
        return

    freq = int(input("Sessions per week: "))
    if not (1 <= freq <= 7):
        print(f"Error: sessions per week must be 1-7. Aborting.")
        return

    goal = input("Goal (loss/gain/maintain): ").lower()
    if goal not in ("loss", "gain", "maintain"):
        print(f"Error: goal must be loss, gain, or maintain. Aborting.")
        return

    bmi = round(weight / (height_m ** 2), 2)
    print(f"\nYour BMI: {bmi}")

    # Predict workout type via Random Forest
    gender_enc  = 1 if gender.lower() == 'male' else 0
    session_dur = 45  # default session length for prediction

    # Build inputs as DataFrames so feature names match training (silences warnings)
    clf_columns = [
        'Age', 'Gender_encoded', 'Weight', 'BMI',
        'Experience_Level', 'Workout_Frequency', 'Session_Duration'
    ]
    sample_clf_df = pd.DataFrame(
        [[age, gender_enc, weight, bmi, exp, freq, session_dur]],
        columns=clf_columns
    )
    sample_clf = clf_scaler.transform(sample_clf_df)
    predicted_workout_enc = rf_model.predict(sample_clf)[0]
    ml_predicted_workout  = workout_le.inverse_transform([predicted_workout_enc])[0]
    proba                 = rf_model.predict_proba(sample_clf)[0]

    # Apply goal-aware override on top of the ML prediction
    final_workout, override_reason = goal_aware_override(
        ml_predicted_workout, goal, proba, workout_le.classes_
    )

    # Predict calories burned via Linear Regression
    reg_columns = [
        'Age', 'Gender_encoded', 'Weight', 'BMI',
        'Session_Duration', 'Workout_Frequency', 'Experience_Level'
    ]
    sample_reg_df = pd.DataFrame(
        [[age, gender_enc, weight, bmi, session_dur, freq, exp]],
        columns=reg_columns
    )
    sample_reg   = reg_scaler.transform(sample_reg_df)
    est_calories = round(cal_model.predict(sample_reg)[0], 1)

    # ── Print ML output (transparent — user sees both) ──────────
    print("\n--- Random Forest Prediction ---")
    print(f"  ML Predicted Workout : {ml_predicted_workout}")
    print(f"\n  Confidence Scores:")
    for wtype, p in zip(workout_le.classes_, proba):
        marker = " <- ML pick" if wtype == ml_predicted_workout else ""
        print(f"    {wtype:<12} {round(p*100, 1)}%{marker}")
    print(f"\n  Estimated Calories Burned/Session : {est_calories} kcal")

    # ── Final recommendation (after goal override) ──────────────
    print("\n--- Final Recommendation ---")
    print(f"  Recommended Workout : {final_workout}")
    if final_workout != ml_predicted_workout:
        print(f"  Note: {override_reason}")
    else:
        print(f"  Reason: {override_reason}")

    # Show specific exercises from library based on FINAL workout
    if final_workout in EXERCISE_LIBRARY and exp in EXERCISE_LIBRARY[final_workout]:
        exercises = EXERCISE_LIBRARY[final_workout][exp]
        levels = ['', 'Beginner', 'Intermediate', 'Advanced']
        print(f"\n--- Your Specific {final_workout} Exercises ---")
        print(f"  Level: {levels[exp]}")
        total_burn = 0
        for i, ex in enumerate(exercises):
            print(f"\n  {i+1}. {ex['name']}")
            print(f"     Duration : {ex['duration']} mins")
            print(f"     Burns    : ~{ex['calories']} kcal")
            print(f"     Location : {ex['equipment']}")
            total_burn += ex['calories']
        print(f"\n  Total Estimated Burn (all exercises) : ~{total_burn} kcal")

    # Goal-based advice
    print(f"\n--- Goal Based Advice ({goal}) ---")
    if goal == "loss":
        print(f"{final_workout} is ideal for fat burning")
        print("   Tip: Aim for 6 days/week and keep sessions above 45 mins")
    elif goal == "gain":
        print(f"{final_workout} is ideal for muscle building")
        print("   Tip: Focus on progressive overload and rest 48h between sessions")
    else:
        print(f"{final_workout} is ideal for overall fitness")
        print(f"   Tip: Maintain consistency with {freq} sessions/week")

    # Final summary
    levels = ['', 'Beginner', 'Intermediate', 'Advanced']
    print(f"\n--- Final Summary ---")
    print(f"  ML Suggested      : {ml_predicted_workout}")
    print(f"  Final Pick        : {final_workout}")
    print(f"  Calories/Session  : {est_calories} kcal")
    print(f"  Your BMI          : {bmi}")
    print(f"  Experience Level  : {levels[exp]}")
    print(f"  Sessions/Week     : {freq}")


# ─────────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────────
def main():
    df = None
    classifier = None
    classifier_scaler = None
    food_model = None
    exercise_models = None   # tuple: (rf, clf_scaler, le, cal_model, reg_scaler)

    while True:
        print("\n" + "=" * 50)
        print("HEALTH & NUTRITION TOOLKIT")
        print("=" * 50)
        print("1. BMI Calculator")
        print("2. Meal Recommender (KNN)")
        print("3. Train + Use Meal Health Classifier")
        print("4. Weight Predictor")
        print("5. Mess Advisory")
        print("6. Food Image Classifier (CNN)")
        print("7. Exercise Recommendation")
        print("0. Exit")

        choice = input("\nChoose option: ").strip()

        if choice == "1":
            calculate_bmi()
        elif choice == "2":
            if df is None:
                df = load_meal_data("healthy_meal_plans.csv")
            recommend_meals(df)
        elif choice == "3":
            if df is None:
                df = load_meal_data("healthy_meal_plans.csv")
            if classifier is None:
                classifier, classifier_scaler = train_health_classifier(df)
            check_meal_health(classifier, classifier_scaler)
        elif choice == "4":
            predict_weight()
        elif choice == "5":
            mess_advisory()
        elif choice == "6":
            if food_model is None:
                food_model = load_food_model()
            food_classifier_menu(food_model)
        elif choice == "7":
            if exercise_models is None:
                exercise_models = train_exercise_models("gym_members_exercise_tracking.csv")
            if exercise_models is not None:
                recommend_exercise(*exercise_models)
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
