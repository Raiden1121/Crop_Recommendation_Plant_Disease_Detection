import numpy as np
import pickle
import cv2
import json
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
directory_root = PROJECT_ROOT / "data" / "plant_disease" / "PlantVillage"
models_dir = PROJECT_ROOT / "models"
width=256
height=256
depth=3

def print_training_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("[INFO] Training device: GPU")
        for index, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            device_name = details.get("device_name", gpu.name)
            compute_capability = details.get("compute_capability")
            capability_text = (
                f", compute capability: {compute_capability}"
                if compute_capability
                else ""
            )
            print(f"[INFO] GPU {index}: {device_name}{capability_text}")
    else:
        cpus = tf.config.list_physical_devices("CPU")
        cpu_names = ", ".join(cpu.name for cpu in cpus) or "CPU"
        print(f"[INFO] Training device: CPU ({cpu_names})")

#Function to convert images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
#Fetch images from directory
image_list, label_list = [], []
try:
    print_training_device()
    print("[INFO] Loading images ...")
    class_folders = [
        folder for folder in directory_root.iterdir()
        if folder.is_dir() and folder.name != ".DS_Store"
    ]

    for class_folder in class_folders:
        print(f"[INFO] Processing {class_folder.name} ...")
        plant_disease_image_list = [
            image_path for image_path in class_folder.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        for image_path in plant_disease_image_list[:200]:
            converted_image = convert_image_to_array(str(image_path))
            if converted_image is not None and converted_image.size > 0:
                image_list.append(converted_image)
                label_list.append(class_folder.name)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

#Get Size of Processed Image
image_size = len(image_list)

#Transform Image Labels uisng Scikit Learn's LabelBinarizer
models_dir.mkdir(parents=True, exist_ok=True)
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
with open(models_dir / "label_transform.pkl", "wb") as label_file:
    pickle.dump(label_binarizer, label_file)
n_classes = len(label_binarizer.classes_)
with open(models_dir / "class_names.json", "w", encoding="utf-8") as class_file:
    json.dump(label_binarizer.classes_.tolist(), class_file, indent=2)

print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()
opt = Adam(learning_rate=INIT_LR)
# distribution
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the network
print("[INFO] training network...")

history = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

# save the model to disk
print("[INFO] Saving model...")
model.save(models_dir / "plant_disease_model.keras")
