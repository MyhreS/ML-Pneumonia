from model.architecture_search.utils.create_directory import create_directory
from model.architecture_search.utils.load_data import load_data
from model.architecture_search.utils.save_model_info import save_model_info
from model.architecture_search.utils.save_results import save_results
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

"""
This script has been used to manually do architecture search. 

It is a script that has been changed for every search, based on the previous search. It created a directory
for every run, named NAME_OF_RUN. In this directory, the model architecture is saved, its testing results
and a txt file to make comments.  
"""


# Set variables
NAME_OF_RUN = "1"
TRAIN_PATH = "../../data/chest-xray-augmented/train"
VAL_PATH = "../../data/chest-xray-augmented/val"
TEST_PATH = "../../data/chest-xray-augmented/test"
IMG_SHAPE = (224, 224, 1)
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
EPOCHS = 50

# Create directory for this run
this_run_dir_path = create_directory(NAME_OF_RUN, False)

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
    optimizer=Adam(learning_rate=LEARNING_RATE),
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
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, tensorboard]
    )

# Evaluate model
print("Evaluate model")
test_results = model.evaluate(test_ds)

# Save last epoch and test results
save_results(history, test_results, LEARNING_RATE, BATCH_SIZE, this_run_dir_path)














