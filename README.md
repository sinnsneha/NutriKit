\# NutriKit — Your Food, Decoded



A Flask web app that combines computer vision and machine learning for health and nutrition tracking.



\## Features



\- \*\*Food Classifier\*\*: Upload a food photo, get a prediction from a CNN trained on Food-101

\- \*\*BMI Calculator\*\*: Quick health stats

\- \*\*Exercise Recommender\*\*: ML-driven workout suggestions based on your stats and goals (trained on the Gym Members Exercise Tracking dataset)

\- \*\*Meal Planner \& Weight Tracker\*\*: Track meals and progress over time



\## Tech Stack



\- Backend: Flask, Flask-CORS

\- ML: TensorFlow/Keras (CNN), scikit-learn (Random Forest, Linear Regression)

\- Frontend: HTML/CSS/JS



\## Setup



```bash

python -m venv .venv

.venv\\Scripts\\activate          # Windows

pip install -r requirements.txt

python server.py

```



Then open `http://127.0.0.1:5000` in your browser.



\## Project Status



Active development. Currently improving the food classifier with transfer learning and a fruit/vegetable dataset.



\## License



MIT

