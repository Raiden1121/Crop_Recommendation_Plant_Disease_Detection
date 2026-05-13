document.addEventListener("DOMContentLoaded", () => {
    setupCropForm();
    setupDiseaseForm();
    setupRevealState();
});

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
                showFieldError(input, `${field.name} is required.`);
                valid = false;
                return;
            }

            if (Number.isNaN(Number(value))) {
                showFieldError(input, `${field.name} must be a number.`);
                valid = false;
                return;
            }

            if (field.id === "ph") {
                const numericValue = Number(value);
                if (numericValue < field.min || numericValue > field.max) {
                    showFieldError(input, "ph should be between 0 and 14.");
                    valid = false;
                    return;
                }
            }

            clearFieldError(input);
        });

        if (!valid) {
            event.preventDefault();
            setFormMessage("cropFormError", "Please fix the highlighted fields before submitting.");
            return;
        }

        setButtonLoading(form.querySelector(".submit-btn"), true);
    });

    fields.forEach((field) => {
        const input = document.getElementById(field.id);
        input.addEventListener("input", () => clearFieldError(input));
    });
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
            setFormMessage("diseaseFormError", "Selected file must be an image.");
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
            setFormMessage("diseaseFormError", "Please choose an image before submitting.");
            return;
        }

        if (!file.type.startsWith("image/")) {
            event.preventDefault();
            uploadPanel.classList.add("has-error");
            setFormMessage("diseaseFormError", "Only image files are allowed.");
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
