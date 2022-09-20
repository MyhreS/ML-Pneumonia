import numpy as np

import load_data, generate_models, train_model, prerun_check, save_model, create_model

CHECK_GPU = True
DELETE_PREVIOUS_RUNS = True
TRAIN_PATH = "../data/chest-xray-dummy/train"
TEST_PATH = "../data/chest-xray-dummy/test"
VAL_PATH = "../data/chest-xray-dummy/val"
IMG_SHAPE = (224, 224, 1)
EPOCHS = 10
PATIENCE = 3
LEANING_RATE = 0.00005
BATCH_SIZE = 32

# Arcitecture search
CONV_LAYERS = [3, 5, 7]
DENSE_LAYERS = [1, 2, 3, 4]

# Hyperparameter search
BATCH_SIZES = [32, 128, 256]
LEARNING_RATES = [0.0001, 0.001, 0.01]
CONV_LAYERS_FILTERS = [32, 64, 128]
DENSE_LAYERS_FILTERS = [32, 64, 128]




name_of_run = prerun_check.prerun_check(check_gpu=CHECK_GPU, delete_previous_runs=DELETE_PREVIOUS_RUNS)

train_ds, val_ds, test_ds, class_names = load_data.load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, BATCH_SIZE)

generated_models = generate_models.generate_models(CONV_LAYERS, DENSE_LAYERS)

for generated_model in generated_models:
    model = create_model.create_model(generated_model, IMG_SHAPE, class_names, LEANING_RATE)
    if model is not None:
        model.summary()
        train_model.train_model(model, train_ds, val_ds, EPOCHS, PATIENCE, name_of_run)
        print("Evaluating model")
        test_results = model.evaluate(test_ds)
        save_model.save_model(model, name_of_run)






