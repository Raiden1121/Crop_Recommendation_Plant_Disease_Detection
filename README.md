# Crop Recommendation Plant Disease Detection

本專案結合兩個農業相關的機器學習任務：

- 根據土壤與環境數據推薦適合種植的作物。
- 根據植物葉片圖片判斷植物是否感染疾病。

## 資料集下載

請自行從 Kaggle 下載後放到指定位置。

- 植物疾病圖片資料集 PlantVillage：
  `https://www.kaggle.com/datasets/emmarex/plantdisease`
- 作物推薦資料集 Crop Recommendation Dataset：
  `https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data`

下載後請放到以下位置：

```text
data/
  crop/
    Crop_recommendation.csv
  plant_disease/
    PlantVillage/
```

## 專案結構

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
  plant_disease_model.keras
  label_transform.pkl
  class_names.json
backend/
  app.py
requirements.txt
README.md
```

## 檔案與資料夾說明

- `data/`：存放本機資料集。此資料夾中的實際資料不會推上 GitHub。
- `data/crop/Crop_recommendation.csv`：作物推薦模型使用的表格資料集。
- `data/plant_disease/PlantVillage/`：植物疾病辨識模型使用的葉片圖片資料集。
- `src/`：存放模型訓練與測試用的 Python 程式。
- `src/train_crop_model.py`：訓練作物推薦模型的程式。
- `src/train_plant_model.py`：訓練植物疾病圖片分類模型的程式。
- `src/test_predict.py`：測試模型預測結果的程式。
- `models/`：存放訓練完成的模型與類別名稱資料。
- `models/crop_model.pkl`：作物推薦模型訓練後預計輸出的模型檔。
- `models/plant_disease_model.keras`：植物疾病辨識模型訓練後預計輸出的模型檔。
- `models/label_transform.pkl`：植物疾病模型訓練時建立的標籤轉換器。
- `models/class_names.json`：植物疾病模型對應的類別名稱。
- `backend/app.py`：Flask 後端 API 入口，之後可用來提供模型預測服務。
- `requirements.txt`：專案需要安裝的 Python 套件清單。
- `README.md`：專案介紹、檔案說明與使用方式。

## 環境建立

建立虛擬環境：

```bash
python -m venv .venv
```

啟動虛擬環境：

```bash
source .venv/bin/activate
```

安裝套件：

```bash
pip install -r requirements.txt
```
