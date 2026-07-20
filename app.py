import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dynamic path resolution to ensure Render finds it
# Change "SVM_model.pkl" if your file has a slightly different extension (like .pkl)
MODEL_PATH = os.path.join(BASE_DIR, "SVM_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Fallback just in case the file has no dot extension in your repository
    MODEL_PATH = os.path.join(BASE_DIR, "SVM_model_pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
