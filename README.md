# NutriKit — Your Food, Decoded
A full-stack health companion that uses computer vision to identify food from photos, recommends personalized workouts via classical ML, and tracks nutrition goals in one unified Flask app.
## What it does
**Snap & Score** — Upload a food image; a Keras CNN trained on Food-101 returns the top-5 predicted dishes with confidence scores.
**Workout Recommender** — Enter your age, gender, weight, height, fitness experience, weekly frequency, and goal. A Random Forest classifier maps you to a workout category (Strength / Cardio / HIIT / Mobility), while a Linear Regression model predicts expected calorie burn. Both models train at server startup on the Gym Members Exercise Tracking dataset.
**BMI, Meals, Weight Tracker** — Quick-access tools for daily health logging, all in one interface.
## Architecture
Single-file Flask backend (`server.py`) serves both the SPA frontend and JSON endpoints. Models load once at startup; exercise models retrain in seconds from the CSV.
## Tech Stack
- **Backend**: Flask, Flask-CORS
- **Deep Learning**: TensorFlow / Keras (CNN trained from scratch, 100 classes)
- **Classical ML**: scikit-learn (Random Forest, Linear Regression, StandardScaler, LabelEncoder)
- **Data**: pandas, NumPy, Pillow
- **Frontend**: Vanilla HTML / CSS / JavaScript — no framework, no build step
## Setup
```bash
git clone https://github.com/sinnsneha/nutrikit.git
cd nutrikit
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
python server.py
```
Open `http://127.0.0.1:5000` in your browser.
> **Note**: The trained CNN (`food_cnn_scratch.keras`, ~163 MB) is not included in this repo due to GitHub's file size limit. To run the classifier locally, train your own or contact me for a download link.
## What I Learned
- End-to-end deployment of a CNN behind a real REST API
- Trade-offs of training from scratch (~37% top-1 confidence on out-of-distribution inputs) vs. transfer learning
- Handling base64 image transport between browser and Flask
- Coordinating multiple ML models with different lifecycles in one server (pre-trained CNN vs. retrain-on-startup tabular models)
## Known Limitations
- **CNN trained from scratch** on Food-101 underperforms a transfer-learning approach. Out-of-distribution inputs (e.g., raw fruit) get confidently misclassified into the nearest Food-101 dish.
- **No confidence thresholding** yet — the UI displays the top prediction even when the model is essentially guessing.
- **Exercise classifier accuracy is modest** (~22%) — the underlying dataset isn't large or labeled finely enough for high-fidelity prediction.
## Roadmap
- [ ] Retrain the food classifier using transfer learning from EfficientNetB0 on Food-101 + Fruits-360 (target: 80%+ top-1 accuracy, 200+ classes including raw produce)
- [ ] Add confidence threshold to flag uncertain predictions instead of hiding them
- [ ] Per-class confusion matrix and out-of-distribution robustness testing
- [ ] Deploy a live demo on Hugging Face Spaces
## License
MIT

\## License



MIT

