import os
import pathlib
from model.hyperparameter_tuning.utils.load_data import load_data
from model.hyperparameter_tuning.utils.save_results import save_results
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import shutil
from tensorflow.keras.applications import InceptionV3

"""
This script has been used to automatically perform hyperparameter tuning.

It was set to run for all the iterations and save the history to tensorBoard and the testing results to a txt file.
It iterates through 35ish different combinations of hyperparameters for the best performing model from the architecture search.
"""

TRAIN_PATH = "../../data/chest-xray-dummy/train"
VAL_PATH = "../../data/chest-xray-dummy/val"
TEST_PATH = "../../data/chest-xray-dummy/test"
IMG_SHAPE_FT = (224, 224, 3)

EPOCHS = 30
LEARNING_RATE = 0.000005
CONV_FILTERs = [16, 32, 64]
DENSE_FILTERs = [64, 128]

# Cleaning directory
current_directory = pathlib.Path(__file__).parent.absolute()
runs_dir = os.path.join(current_directory, "model_runs")
if os.path.exists(runs_dir):
    shutil.rmtree(runs_dir)
os.mkdir(runs_dir)

# Set some parameters
best_result_model_name = ""
best_result = 0


"""
Hyperparameter tuning the model from approach two
"""

train_ds, val_ds, test_ds, class_names =  load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE_FT, 32)

extractor = InceptionV3(input_shape=IMG_SHAPE_FT,
                                include_top=False,
                                weights='imagenet')
extractor.trainable = False
layer_name = 'mixed2'

extractor_cut = Model(inputs=extractor.input, outputs=extractor.get_layer(layer_name).output)

i = 0
# Two layers of conv
for first_conv_layer_filter in CONV_FILTERs:
    for second_conv_layer_filter in CONV_FILTERs:
        # Two layers of dense
        for first_dense_layer_filter in DENSE_FILTERs:
            for second_dense_layer_filter in DENSE_FILTERs:

                print("\n\nIteration: {}\n".format(i))
                i += 1

                name = "ft_conv-{}-{}-dense-{}-{}".format(first_conv_layer_filter, second_conv_layer_filter, first_dense_layer_filter, second_dense_layer_filter)

                model = Sequential([
                    extractor_cut,
                    Conv2D(first_conv_layer_filter, 3, activation='relu'),
                    Dropout(0.2),
                    Conv2D(second_conv_layer_filter, 3, activation='relu'),
                    Flatten(),
                    Dense(first_dense_layer_filter, activation='relu'),
                    Dropout(0.2),
                    Dense(second_dense_layer_filter, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(
                    optimizer=Adam(learning_rate=LEARNING_RATE),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                model._name = name

                model.summary()

                # Set callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=3
                )
                logs_dir = "{}/tensorboard/{}".format('model_runs', model.name)
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
                test_results = model.evaluate(test_ds)
                print("Test results: Loss: {}, Accuracy: {}".format(test_results[0], test_results[1]))

                # Save results
                save_results(test_results, model.name, runs_dir)

                if test_results[1] > best_result:
                    best_result_model_name = model.name
                    best_result = test_results[1]




