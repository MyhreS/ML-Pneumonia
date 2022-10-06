import os
import pathlib
from load_data import load_data
from save_results import save_results
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import shutil
from tensorflow.keras.applications import InceptionV3

TRAIN_PATH = "../../data/chest-xray-dummy/train"
VAL_PATH = "../../data/chest-xray-dummy/val"
TEST_PATH = "../../data/chest-xray-dummy/test"
IMG_SHAPE = (224, 224, 1)
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

# Load data
train_ds, val_ds, test_ds, class_names =  load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, 32)

"""
Hyperparameter tuning the model from approach one
"""

i = 0
# Four layers of dense
for first_conv_layer_filter in CONV_FILTERs:
    for second_conv_layer_filter in CONV_FILTERs:
        for third_conv_layer_filter in CONV_FILTERs:
            for fourth_conv_layer_filter in CONV_FILTERs:
                # Two layers of dense
                for first_dense_layer_filter in DENSE_FILTERs:
                    for second_dense_layer_filter in DENSE_FILTERs:
                        print("\n\nIteration: {}\n".format(i))
                        i += 1
                        name = "conv-{}-{}-{}-{}-dense-{}-{}".format(first_conv_layer_filter, second_conv_layer_filter, third_conv_layer_filter, fourth_conv_layer_filter, first_dense_layer_filter, second_dense_layer_filter)

                        model = Sequential([
                            Input(shape=IMG_SHAPE),
                            Rescaling(1. / 255),
                            Conv2D(first_conv_layer_filter, 3, activation='relu'),
                            MaxPooling2D(),
                            Conv2D(second_conv_layer_filter, 3, activation='relu'),
                            MaxPooling2D(),
                            Conv2D(third_conv_layer_filter, 3, activation='relu'),
                            Dropout(0.2),
                            MaxPooling2D(),
                            Conv2D(fourth_conv_layer_filter, 3, activation='relu'),
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


"""
Hyperparameter tuning the model from approach two
"""



extractor = InceptionV3(input_shape=IMG_SHAPE_FT,
                                include_top=False,
                                weights='imagenet')
extractor.trainable = False
layer_name = 'mixed2'

extractor_cut = Model(inputs=extractor.input, outputs=extractor.get_layer(layer_name).output)


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

# Save best result to file
with open("{}/best_result.txt".format(runs_dir), "w") as f:
    f.write("Best result model: {}\n".format(best_result_model_name))
    f.write("Best result: {}".format(best_result))






