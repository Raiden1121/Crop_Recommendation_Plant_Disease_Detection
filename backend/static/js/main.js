document.addEventListener("DOMContentLoaded", () => {
    setupLanguage();
    setupCropForm();
    setupDiseaseForm();
    setupRevealState();
});

const translations = {
    en: {
        "nav.home": "Home",
        "nav.crop": "Crop Recommendation",
        "nav.disease": "Disease Detection",
        "common.backHome": "← Back to Home",
        "result.label": "Prediction Result",
        "home.eyebrow": "Agriculture x Machine Learning",
        "home.title": "Crop Recommendation & Plant Disease Detection",
        "home.intro": "A Flask-based agricultural intelligence system that combines structured environmental data with plant image classification to support smarter cultivation decisions and faster disease screening.",
        "home.tryCrop": "Try Crop Recommendation",
        "home.tryDisease": "Try Disease Detection",
        "home.inputs": "Inputs",
        "home.metricCropTitle": "7 Soil & Climate Features",
        "home.metricCropDesc": "N, P, K, temperature, humidity, pH, rainfall",
        "home.visionModel": "Vision Model",
        "home.metricDiseaseTitle": "Leaf Image Classification",
        "home.metricDiseaseDesc": "Upload a plant leaf image to identify possible disease classes.",
        "home.modules": "Project Modules",
        "home.workflowTitle": "Two Core ML Workflows",
        "home.workflowDesc": "Separate interfaces for tabular prediction and image-based diagnosis, designed for class presentation and live demo use.",
        "home.cropCardDesc": "Enter soil nutrient values and environmental conditions, then let the trained model recommend a suitable crop for planting.",
        "home.cropPoint1": "Structured data input form",
        "home.cropPoint2": "Fast prediction response",
        "home.cropPoint3": "Clear recommended crop result card",
        "home.openCrop": "Open Crop Page",
        "home.diseaseCardDesc": "Upload a leaf image and use the trained CNN model to classify plant disease categories with confidence feedback.",
        "home.diseasePoint1": "Image upload with preview",
        "home.diseasePoint2": "Disease name and confidence score",
        "home.diseasePoint3": "Designed for demo-friendly presentation",
        "home.openDisease": "Open Disease Page",
        "crop.eyebrow": "Tabular ML Prediction",
        "crop.title": "Crop Recommendation",
        "crop.intro": "Input soil nutrients and environmental conditions to predict a suitable crop from the trained machine learning model.",
        "crop.requiredTitle": "Required Features",
        "crop.requiredDesc": "The model expects seven numeric values describing soil fertility and growing conditions.",
        "crop.featureN": "Nitrogen ratio in soil",
        "crop.featureP": "Phosphorous ratio in soil",
        "crop.featureK": "Potassium ratio in soil",
        "crop.featureTemp": "degree Celsius (°C)",
        "crop.featureHumidity": "relative humidity (%)",
        "crop.featurePh": "soil pH value",
        "crop.featureRainfall": "rainfall in mm",
        "crop.recommended": "Recommended Crop",
        "crop.resultDesc": "The recommendation is generated from the submitted soil and climate values.",
        "crop.formTitle": "Enter Environmental Data",
        "crop.formDesc": "All fields are required and should contain numeric values.",
        "crop.nHelp": "Ratio of Nitrogen content in soil.",
        "crop.pHelp": "Ratio of Phosphorous content in soil.",
        "crop.kHelp": "Ratio of Potassium content in soil.",
        "crop.temperatureHelp": "Temperature in degree Celsius.",
        "crop.humidityHelp": "Relative humidity in %.",
        "crop.phHelp": "pH value of the soil. Recommended range: 0 to 14.",
        "crop.rainfallHelp": "Rainfall in mm.",
        "crop.submit": "Get Recommendation",
        "crop.randomize": "Generate Random Data",
        "crop.loading": "Predicting...",
        "unit.ratio": "(ratio)",
        "unit.value": "(value)",
        "disease.eyebrow": "Computer Vision Prediction",
        "disease.title": "Plant Disease Detection",
        "disease.intro": "Upload a plant leaf image and let the trained deep learning model classify the disease category with confidence feedback.",
        "disease.guidelinesTitle": "Upload Guidelines",
        "disease.guidelinesDesc": "Use a clear leaf photo with visible disease patterns. Supported formats: PNG, JPG, JPEG, WEBP.",
        "disease.predicted": "Predicted Disease:",
        "disease.formTitle": "Upload Plant Leaf Image",
        "disease.formDesc": "Select one image file to run disease classification.",
        "disease.chooseImage": "Choose an image",
        "disease.chooseDesc": "Click to browse or replace the selected image",
        "disease.preview": "Image preview will appear here.",
        "disease.submit": "Detect Disease",
        "disease.loading": "Analyzing...",
        "validation.required": "{field} is required.",
        "validation.number": "{field} must be a number.",
        "validation.phRange": "ph should be between 0 and 14.",
        "validation.fixFields": "Please fix the highlighted fields before submitting.",
        "validation.imageRequired": "Please choose an image before submitting.",
        "validation.imageOnly": "Only image files are allowed.",
        "validation.selectedImage": "Selected file must be an image.",
    },
    zh: {
        "nav.home": "首頁",
        "nav.crop": "作物推薦",
        "nav.disease": "植物病害辨識",
        "common.backHome": "← 回到首頁",
        "result.label": "預測結果",
        "home.eyebrow": "農業 x 機器學習",
        "home.title": "作物推薦與植物病害辨識",
        "home.intro": "這是一個以 Flask 建立的農業智慧系統，結合環境數值資料與植物影像分類，協助做出更好的栽培決策並加速病害篩檢。",
        "home.tryCrop": "試用作物推薦",
        "home.tryDisease": "試用病害辨識",
        "home.inputs": "輸入資料",
        "home.metricCropTitle": "7 項土壤與氣候特徵",
        "home.metricCropDesc": "N、P、K、溫度、濕度、pH、降雨量",
        "home.visionModel": "影像模型",
        "home.metricDiseaseTitle": "葉片影像分類",
        "home.metricDiseaseDesc": "上傳植物葉片影像，辨識可能的病害類別。",
        "home.modules": "專案模組",
        "home.workflowTitle": "兩個核心 ML 流程",
        "home.workflowDesc": "分別提供表格資料預測與影像診斷介面，適合課堂展示與現場 Demo。",
        "home.cropCardDesc": "輸入土壤養分與環境條件，讓訓練好的模型推薦適合作物。",
        "home.cropPoint1": "結構化數值輸入表單",
        "home.cropPoint2": "快速取得預測結果",
        "home.cropPoint3": "清楚醒目的推薦結果卡片",
        "home.openCrop": "開啟作物推薦",
        "home.diseaseCardDesc": "上傳葉片影像，使用訓練好的 CNN 模型分類植物病害並顯示信心分數。",
        "home.diseasePoint1": "影像上傳與預覽",
        "home.diseasePoint2": "病害名稱與信心分數",
        "home.diseasePoint3": "適合展示的操作介面",
        "home.openDisease": "開啟病害辨識",
        "crop.eyebrow": "表格資料 ML 預測",
        "crop.title": "作物推薦",
        "crop.intro": "輸入土壤養分與環境條件，使用訓練好的機器學習模型預測適合種植的作物。",
        "crop.requiredTitle": "必要特徵",
        "crop.requiredDesc": "模型需要七個數值，描述土壤肥力與作物生長環境。",
        "crop.featureN": "土壤氮含量比例",
        "crop.featureP": "土壤磷含量比例",
        "crop.featureK": "土壤鉀含量比例",
        "crop.featureTemp": "攝氏溫度 (°C)",
        "crop.featureHumidity": "相對濕度 (%)",
        "crop.featurePh": "土壤 pH 值",
        "crop.featureRainfall": "降雨量 (mm)",
        "crop.recommended": "推薦作物",
        "crop.resultDesc": "此建議是根據送出的土壤與氣候數值產生。",
        "crop.formTitle": "輸入環境資料",
        "crop.formDesc": "所有欄位皆為必填，且需輸入數值。",
        "crop.nHelp": "土壤氮含量比例。",
        "crop.pHelp": "土壤磷含量比例。",
        "crop.kHelp": "土壤鉀含量比例。",
        "crop.temperatureHelp": "溫度，單位為攝氏度。",
        "crop.humidityHelp": "相對濕度，單位為 %。",
        "crop.phHelp": "土壤 pH 值，建議範圍為 0 到 14。",
        "crop.rainfallHelp": "降雨量，單位為 mm。",
        "crop.submit": "取得推薦",
        "crop.randomize": "隨機產生資料",
        "crop.loading": "預測中...",
        "unit.ratio": "(比例)",
        "unit.value": "(數值)",
        "disease.eyebrow": "電腦視覺預測",
        "disease.title": "植物病害辨識",
        "disease.intro": "上傳植物葉片影像，讓訓練好的深度學習模型分類病害類別並回傳信心分數。",
        "disease.guidelinesTitle": "上傳建議",
        "disease.guidelinesDesc": "請使用清楚、病徵可見的葉片照片。支援格式：PNG、JPG、JPEG、WEBP。",
        "disease.predicted": "預測病害：",
        "disease.formTitle": "上傳植物葉片影像",
        "disease.formDesc": "選擇一張影像檔進行病害分類。",
        "disease.chooseImage": "選擇影像",
        "disease.chooseDesc": "點擊瀏覽或替換目前選取的影像",
        "disease.preview": "影像預覽會顯示在這裡。",
        "disease.submit": "辨識病害",
        "disease.loading": "分析中...",
        "validation.required": "請填寫 {field}。",
        "validation.number": "{field} 必須是數值。",
        "validation.phRange": "ph 應介於 0 到 14。",
        "validation.fixFields": "送出前請先修正標示的欄位。",
        "validation.imageRequired": "送出前請先選擇一張影像。",
        "validation.imageOnly": "僅允許上傳影像檔。",
        "validation.selectedImage": "選取的檔案必須是影像。",
    },
};

let currentLang = "en";

function setupLanguage() {
    currentLang = localStorage.getItem("agriMLLanguage") || "en";
    applyLanguage(currentLang);

    document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
        button.addEventListener("click", () => {
            currentLang = currentLang === "en" ? "zh" : "en";
            localStorage.setItem("agriMLLanguage", currentLang);
            applyLanguage(currentLang);
        });
    });
}

function applyLanguage(lang) {
    const dictionary = translations[lang] || translations.en;
    document.documentElement.lang = lang === "zh" ? "zh-Hant" : "en";

    document.querySelectorAll("[data-i18n]").forEach((element) => {
        const key = element.dataset.i18n;
        if (dictionary[key]) element.textContent = dictionary[key];
    });

    document.querySelectorAll("[data-i18n-loading]").forEach((element) => {
        const key = element.dataset.i18nLoading;
        if (dictionary[key]) element.dataset.loadingText = dictionary[key];
    });

    document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
        button.textContent = lang === "en" ? "中文" : "EN";
        button.setAttribute("aria-label", lang === "en" ? "Switch to Chinese" : "切換為英文");
    });
}

function t(key, replacements = {}) {
    const template = translations[currentLang]?.[key] || translations.en[key] || key;
    return Object.entries(replacements).reduce((message, [name, value]) => {
        return message.replace(`{${name}}`, value);
    }, template);
}

function setupRevealState() {
    document.querySelectorAll(".reveal").forEach((element) => {
        element.style.willChange = "transform, opacity";
    });
}

function setupCropForm() {
    const form = document.getElementById("cropForm");
    if (!form) return;

    const fields = [
        { id: "N", name: "N" },
        { id: "P", name: "P" },
        { id: "K", name: "K" },
        { id: "temperature", name: "temperature" },
        { id: "humidity", name: "humidity" },
        { id: "ph", name: "ph", min: 0, max: 14 },
        { id: "rainfall", name: "rainfall" },
    ];

    form.addEventListener("submit", (event) => {
        let valid = true;
        clearFormMessage("cropFormError");

        fields.forEach((field) => {
            const input = document.getElementById(field.id);
            const value = input.value.trim();

            if (value === "") {
                showFieldError(input, t("validation.required", { field: field.name }));
                valid = false;
                return;
            }

            if (Number.isNaN(Number(value))) {
                showFieldError(input, t("validation.number", { field: field.name }));
                valid = false;
                return;
            }

            if (field.id === "ph") {
                const numericValue = Number(value);
                if (numericValue < field.min || numericValue > field.max) {
                    showFieldError(input, t("validation.phRange"));
                    valid = false;
                    return;
                }
            }

            clearFieldError(input);
        });

        if (!valid) {
            event.preventDefault();
            setFormMessage("cropFormError", t("validation.fixFields"));
            return;
        }

        setButtonLoading(form.querySelector(".submit-btn"), true);
    });

    fields.forEach((field) => {
        const input = document.getElementById(field.id);
        input.addEventListener("input", () => clearFieldError(input));
    });

    const randomButton = document.getElementById("randomCropDataBtn");
    if (randomButton) {
        randomButton.addEventListener("click", () => {
            fillRandomCropData();
            fields.forEach((field) => clearFieldError(document.getElementById(field.id)));
            clearFormMessage("cropFormError");
        });
    }
}

function fillRandomCropData() {
    const randomValues = {
        N: randomInteger(0, 140),
        P: randomInteger(5, 145),
        K: randomInteger(5, 205),
        temperature: randomDecimal(8, 44, 1),
        humidity: randomDecimal(14, 100, 1),
        ph: randomDecimal(3.5, 9.9, 2),
        rainfall: randomDecimal(20, 300, 1),
    };

    Object.entries(randomValues).forEach(([id, value]) => {
        const input = document.getElementById(id);
        if (input) input.value = value;
    });
}

function randomInteger(min, max) {
    return String(Math.floor(Math.random() * (max - min + 1)) + min);
}

function randomDecimal(min, max, precision) {
    return (Math.random() * (max - min) + min).toFixed(precision);
}

function setupDiseaseForm() {
    const form = document.getElementById("diseaseForm");
    if (!form) return;

    const fileInput = document.getElementById("plant_image");
    const previewImage = document.getElementById("imagePreview");
    const previewBox = document.getElementById("previewBox");
    const placeholder = previewBox.querySelector(".preview-placeholder");
    const fileInfo = document.getElementById("fileInfo");
    const uploadPanel = document.getElementById("uploadPanel");

    fileInput.addEventListener("change", () => {
        clearFormMessage("diseaseFormError");
        uploadPanel.classList.remove("has-error");

        const file = fileInput.files[0];
        if (!file) {
            resetPreview(previewImage, placeholder, fileInfo);
            return;
        }

        if (!file.type.startsWith("image/")) {
            fileInput.value = "";
            resetPreview(previewImage, placeholder, fileInfo);
            uploadPanel.classList.add("has-error");
            setFormMessage("diseaseFormError", t("validation.selectedImage"));
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewImage.classList.remove("hidden");
            placeholder.classList.add("hidden");
            fileInfo.textContent = `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        };
        reader.readAsDataURL(file);
    });

    form.addEventListener("submit", (event) => {
        clearFormMessage("diseaseFormError");
        uploadPanel.classList.remove("has-error");

        const file = fileInput.files[0];
        if (!file) {
            event.preventDefault();
            uploadPanel.classList.add("has-error");
            setFormMessage("diseaseFormError", t("validation.imageRequired"));
            return;
        }

        if (!file.type.startsWith("image/")) {
            event.preventDefault();
            uploadPanel.classList.add("has-error");
            setFormMessage("diseaseFormError", t("validation.imageOnly"));
            return;
        }

        setButtonLoading(form.querySelector(".submit-btn"), true);
    });
}

function showFieldError(input, message) {
    const group = input.closest(".form-group");
    const error = group.querySelector(".field-error");
    group.classList.add("has-error");
    error.textContent = message;
}

function clearFieldError(input) {
    const group = input.closest(".form-group");
    const error = group.querySelector(".field-error");
    group.classList.remove("has-error");
    error.textContent = "";
}

function setFormMessage(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) element.textContent = message;
}

function clearFormMessage(elementId) {
    setFormMessage(elementId, "");
}

function setButtonLoading(button, isLoading) {
    if (!button) return;

    const originalText = button.dataset.originalText || button.textContent.trim();
    button.dataset.originalText = originalText;

    if (isLoading) {
        button.classList.add("is-loading");
        button.disabled = true;
        button.textContent = button.dataset.loadingText || "Loading...";
    } else {
        button.classList.remove("is-loading");
        button.disabled = false;
        button.textContent = originalText;
    }
}

function resetPreview(previewImage, placeholder, fileInfo) {
    previewImage.src = "#";
    previewImage.classList.add("hidden");
    placeholder.classList.remove("hidden");
    fileInfo.textContent = "";
}
