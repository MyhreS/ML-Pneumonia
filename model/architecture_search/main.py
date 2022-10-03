from create_directory import create_diretory
from load_data import load_data
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Set variables
NAME_OF_RUN = "4"
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

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

model.summary()



