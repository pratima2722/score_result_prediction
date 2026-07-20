import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dynamic path resolution to ensure Render finds your pickle file
MODEL_PATH = os.path.join(BASE_DIR, "SVM_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Fallback in case the file was pushed without the .pkl extension
    MODEL_PATH = os.path.join(BASE_DIR, "SVM_model_pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Glassmorphism UI Template with smooth CSS Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        :root {
            --primary: #6c5ce7;
            --secondary: #a29bfe;
            --bg-grad: linear-gradient(135deg, #74b9ff, #a29bfe);
            --card-bg: rgba(255, 255, 255, 0.85);
            --text: #2d3436;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: var(--bg-grad);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            color: var(--text);
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 700px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-out;
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: var(--primary);
            font-size: 2.2rem;
            position: relative;
        }

        h1::after {
            content: '';
            display: block;
            width: 50px;
            height: 4px;
            background: var(--primary);
            margin: 8px auto 0 auto;
            border-radius: 2px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            color: #4a4a4a;
        }

        input, select {
            padding: 12px;
            border: 2px solid #dfe6e9;
            border-radius: 10px;
            outline: none;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.9);
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 8px rgba(108, 92, 231, 0.3);
            transform: translateY(-2px);
        }

        button {
            width: 100%;
            padding: 15px;
            background: var(--primary);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
        }

        button:hover {
            background: #5b4cc4;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(108, 92, 231, 0.6);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
            animation: slideUp 0.5s ease-out;
            background: #dff9fb;
            border-left: 5px solid #130cb7;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Student Prediction Portal</h1>
    <form method="POST" action="/">
        <div class="grid">
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" min="10" max="100" required>
            </div>

            <div class="form-group">
                <label for="study_hours">Study Hours / Week</label>
                <input type="number" step="0.1" id="study_hours" name="study_hours" min="0" required>
            </div>

            <div class="form-group">
                <label for="attendance">Attendance Rate (%)</label>
                <input type="number" step="0.1" id="attendance" name="attendance" min="0" max="100" required>
            </div>

            <div class="form-group">
                <label for="parent_education">Parent Education Level</label>
                <select id="parent_education" name="parent_education" required>
                    <option value="0">None / Primary</option>
                    <option value="1">Secondary</option>
                    <option value="2">Higher / Degree</option>
                </select>
            </div>

            <div class="form-group">
                <label for="internet">Internet Access</label>
                <select id="internet" name="internet" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="extracurricular">Extracurricular Activities</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="previous_score">Previous Score</label>
                <input type="number" step="0.1" id="previous_score" name="previous_score" required>
            </div>

            <div class="form-group">
                <label for="final_score">Expected Final Score</label>
                <input type="number" step="0.1" id="final_score" name="final_score" required>
            </div>
        </div>

        <button type="submit">Run Model Prediction</button>
    </form>

    {% if prediction %}
    <div class="result-box">
        Prediction Status: <strong>{{ prediction }}</strong>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Extract inputs matching the model's exact expected numerical schema
            features = [
                float(request.form["gender"]),
                float(request.form["age"]),
                float(request.form["study_hours"]),
                float(request.form["attendance"]),
                float(request.form["parent_education"]),
                float(request.form["internet"]),
                float(request.form["extracurricular"]),
                float(request.form["previous_score"]),
                float(request.form["final_score"])
            ]
            
            # Structure into the required 2D format for scikit-learn
            final_features = np.array([features])
            prediction = model.predict(final_features)[0]
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
