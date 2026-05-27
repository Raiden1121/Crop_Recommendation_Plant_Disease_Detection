from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PLANT_MODELS_DIR = MODELS_DIR / "plant"
PLANT_DISEASE_MODEL_PATH = PLANT_MODELS_DIR / "advanced_cnn.keras"
UPLOAD_DIR = BACKEND_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def load_crop_model():
    return joblib.load(MODELS_DIR / "crop_model.pkl")


def load_disease_model():
    return tf.keras.models.load_model(PLANT_DISEASE_MODEL_PATH)


def load_class_names():
    class_names_path = PLANT_MODELS_DIR / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as file:
            return json.load(file)

    label_transform_path = PLANT_MODELS_DIR / "label_transform.pkl"
    if label_transform_path.exists():
        with open(label_transform_path, "rb") as file:
            encoder = pickle.load(file)
            return list(encoder.classes_)

    raise FileNotFoundError("Cannot find class_names.json or label_transform.pkl")


crop_model = load_crop_model()
disease_model = load_disease_model()
class_names = load_class_names()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: Path):
    input_shape = disease_model.input_shape
    height = input_shape[1] or 224
    width = input_shape[2] or 224

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/crop")
def crop_page():
    return render_template("crop.html")


@app.route("/disease")
def disease_page():
    return render_template("disease.html")


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        field_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        values = [float(request.form[field]) for field in field_names]
        features = np.array([values], dtype=np.float32)
        prediction = crop_model.predict(features)[0]

        return render_template(
            "crop.html",
            prediction=prediction,
            form_data=request.form,
        )
    except Exception as exc:
        return render_template(
            "crop.html",
            error=f"Prediction failed: {exc}",
            form_data=request.form,
        )


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    file = request.files.get("plant_image")

    if not file or file.filename == "":
        return render_template("disease.html", error="Please select an image file.")

    if not allowed_file(file.filename):
        return render_template(
            "disease.html",
            error="Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP.",
        )

    try:
        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        image_batch = preprocess_image(save_path)
        probabilities = disease_model.predict(image_batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        prediction = class_names[predicted_index]
        confidence = round(float(probabilities[predicted_index]) * 100, 2)

        return render_template(
            "disease.html",
            prediction=prediction,
            confidence=confidence,
            image_path=f"uploads/{filename}",
        )
    except Exception as exc:
        return render_template("disease.html", error=f"Prediction failed: {exc}")


if __name__ == "__main__":
    app.run(debug=True)
