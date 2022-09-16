import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping



# Load the data
train_ds = image_dataset_from_directory(
    "../data/chest-xray-dummy/train",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode='grayscale'
)

val_ds = image_dataset_from_directory(
    "../data/chest-xray-dummy/val",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode='grayscale'
)

test_ds = image_dataset_from_directory(
    "../data/chest-xray-dummy/test",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode='grayscale'
)

class_names = train_ds.class_names
print("Class names:", class_names)

model = Sequential([
    Input(shape=(224, 224, 1)),
    Rescaling(1. / 255),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names)-1, activation='sigmoid')

])

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping]
)

# Evaluate the model
print("Evaluate")
model.evaluate(test_ds)


