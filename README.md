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
  plant/
    advanced_cnn.keras
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
- `models/plant/advanced_cnn.keras`：植物疾病辨識模型訓練後輸出的模型檔，檔名對應 `src/plant_models/advanced_cnn.py`。
- `models/plant/label_transform.pkl`：植物疾病模型訓練時建立的標籤轉換器。
- `models/plant/class_names.json`：植物疾病模型對應的類別名稱。
- `backend/app.py`：Flask 後端 API 入口，之後可用來提供模型預測服務。
- `requirements.txt`：專案需要安裝的 Python 套件清單。
- `README.md`：專案介紹、檔案說明與使用方式。

## 網頁介面功能

本專案的 Flask 網頁介面包含首頁、作物推薦頁面與植物病害辨識頁面。

- 首頁提供兩個主要功能入口：`Crop Recommendation` 與 `Disease Detection`。
- 網頁右上角提供中英文切換按鈕，可在英文與繁體中文介面之間切換。
- 語言選擇會儲存在瀏覽器 `localStorage`，重新整理或切換頁面後仍會保留。

### Crop Recommendation 頁面

作物推薦頁面可輸入七個土壤與環境數值，送出後會顯示推薦作物。

輸入欄位與單位如下：

| 欄位 | 說明 | 單位 |
| --- | --- | --- |
| `N` | 土壤氮含量比例 | ratio |
| `P` | 土壤磷含量比例 | ratio |
| `K` | 土壤鉀含量比例 | ratio |
| `temperature` | 溫度 | degree Celsius / °C |
| `humidity` | 相對濕度 | % |
| `ph` | 土壤 pH 值 | value |
| `rainfall` | 降雨量 | mm |

頁面新增功能：

- `Required Features` 區塊改成清楚的格狀資訊卡，避免標籤擠在一起。
- 表單欄位標籤與說明文字補上對應單位。
- `Generate Random Data / 隨機產生資料` 按鈕可自動填入合理範圍的測試資料。
- 推薦結果卡片改為更醒目的樣式，會突出顯示推薦作物名稱。

隨機產生資料的範圍：

| 欄位 | 隨機範圍 |
| --- | --- |
| `N` | 0-140 |
| `P` | 5-145 |
| `K` | 5-205 |
| `temperature` | 8.0-44.0 °C |
| `humidity` | 14.0-100.0 % |
| `ph` | 3.50-9.90 |
| `rainfall` | 20.0-300.0 mm |

### Disease Detection 頁面

植物病害辨識頁面可上傳植物葉片圖片，模型會回傳預測病害類別與信心分數。

- 支援圖片格式：PNG、JPG、JPEG、WEBP。
- 上傳圖片後會在頁面中顯示預覽。
- 表單錯誤提示與按鈕文字也支援中英文切換。

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

## 網頁啟動方式

請先確認已完成環境建立與套件安裝，並且模型檔案已放在指定位置：

```text
models/
  crop_model.pkl
  plant/
    best_vit_plant_disease_model.keras
    class_names.json
```

啟動 Flask 網頁：

```bash
python backend/app.py
```

如果使用專案內的虛擬環境，也可以直接執行：

```bash
.venv/bin/python backend/app.py
```

啟動後在瀏覽器開啟：

```text
http://127.0.0.1:5000
```

可使用的頁面：

- 首頁：`http://127.0.0.1:5000/`
- 作物推薦：`http://127.0.0.1:5000/crop`
- 植物病害辨識：`http://127.0.0.1:5000/disease`

注意：如果 `crop_model.pkl` 或植物病害模型 `.keras` 檔案不存在，網頁啟動時會因為無法載入模型而失敗。
