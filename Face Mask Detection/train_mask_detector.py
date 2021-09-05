# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

DIRECTORY = "./dataset"
CATEGORIES = ["mask", "no_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(train_X, test_X, train_Y, test_Y) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network
base_model = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# defining the top layers of the model
top_layer = base_model.output
top_layer = AveragePooling2D(pool_size=(7, 7))(top_layer)
top_layer = Flatten(name="flatten")(top_layer)
top_layer = Dense(128, activation="relu")(top_layer)
top_layer = Dropout(0.5)(top_layer)
top_layer = Dense(2, activation="softmax")(top_layer)

# Full Model
model = Model(inputs=base_model.input, outputs=top_layer)

# Making base_model layers non-trainable
for layer in base_model.layers:
	layer.trainable = False

# compiling model
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# training model
H = model.fit(
	aug.flow(train_X, train_Y, batch_size=BATCH_SIZE),
	steps_per_epoch=len(train_X) // BATCH_SIZE,
	validation_data=(test_X, test_Y),
	validation_steps=len(test_X) // BATCH_SIZE,
	epochs=EPOCHS)

# making predictions
preds = model.predict(test_X, batch_size=BATCH_SIZE)
preds = np.argmax(preds, axis=1)

# show a nicely formatted classification report
print(classification_report(test_Y.argmax(axis=1), preds,
	target_names=encoder.classes_))

# saving model
model.save("mask_detector.model", save_format="h5")

# plotting
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")