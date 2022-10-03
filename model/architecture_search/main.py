from create_directory import create_diretory
from load_data import load_data
from save_model_info import save_model_info
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# Set variables
NAME_OF_RUN = "1"
TRAIN_PATH = "../../data/chest-xray-dummy/train"
VAL_PATH = "../../data/chest-xray-dummy/val"
TEST_PATH = "../../data/chest-xray-dummy/test"
IMG_SHAPE = (224, 224, 1)
BATCH_SIZE = 32

# Create directory for this run
this_run_dir_path = create_diretory(NAME_OF_RUN, True)

# Load the data
train_ds, val_ds, test_ds, class_names =  load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, BATCH_SIZE)

# Create model
model = Sequential([
    Input(shape=IMG_SHAPE),
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

# Save model info
save_model_info(model, this_run_dir_path)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )


model.summary()

# Set callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3
    )
logs_dir = "{}/tensorboard/{}".format(this_run_dir_path, model.name)
tensorboard = TensorBoard(
    log_dir=logs_dir
    )

# Train model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[early_stopping, tensorboard]
    )








