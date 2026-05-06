# Crop Recommendation Plant Disease Detection

This project combines two agriculture machine learning tasks:

- Crop recommendation from soil and weather data.
- Plant disease detection from PlantVillage leaf images.

## Project Structure

```text
data/
  crop/
    Crop_recommendation.csv
  plant_disease/
    PlantVillage/
src/
  train_crop_model.py
  train_plant_model.py
  test_predict.py
models/
  crop_model.pkl
  plant_disease_model.h5
  class_names.json
backend/
  app.py
requirements.txt
README.md
```

## Files And Folders

- `data/`: Stores datasets used for training and testing.
- `data/crop/Crop_recommendation.csv`: Tabular crop recommendation dataset.
- `data/plant_disease/PlantVillage/`: Image dataset for plant disease classification.
- `src/`: Stores Python scripts for training and prediction.
- `src/train_crop_model.py`: Script for training the crop recommendation model.
- `src/train_plant_model.py`: Script for training the plant disease image model.
- `src/test_predict.py`: Script for testing model predictions.
- `models/`: Stores trained model files and label metadata.
- `models/crop_model.pkl`: Expected output file for the trained crop recommendation model.
- `models/plant_disease_model.h5`: Expected output file for the trained plant disease model.
- `models/class_names.json`: Stores plant disease class names used by the image model.
- `backend/app.py`: Flask backend entry point for serving predictions.
- `requirements.txt`: Python package dependencies for this project.
- `README.md`: Project overview and file structure documentation.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

dataset:
https://www.kaggle.com/datasets/emmarex/plantdisease
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data