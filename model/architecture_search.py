import load_data, generate_architecture_models, train_model, prerun_check, save_model, create_model

CHECK_GPU = True
DELETE_PREVIOUS_RUNS = True
SAVE_DIR = "architecture_search"
TRAIN_PATH = "../data/chest-xray-augmented/train"
TEST_PATH = "../data/chest-xray-augmented/test"
VAL_PATH = "../data/chest-xray-augmented/val"
IMG_SHAPE = (224, 224, 1)
EPOCHS = 2
PATIENCE = 3
LEARNING_RATE = 0.00005
BATCH_SIZE = 32

# Adjust size of architectures that are included in the search
PARAMS_LOWER_BOUND = 100_000
PARAMS_UPPER_BOUND = 5_000_000

# Arcitecture search sizes
CONV_LAYERS = [3, 5, 7]
DENSE_LAYERS = [1, 2, 3, 4]




name_of_run = prerun_check.prerun_check(save_dir=SAVE_DIR, check_gpu=CHECK_GPU, delete_previous_runs=DELETE_PREVIOUS_RUNS)

train_ds, val_ds, test_ds, class_names = load_data.load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, BATCH_SIZE)

generated_models = generate_architecture_models.generate_models(CONV_LAYERS, DENSE_LAYERS)

for generated_model in generated_models:
    model = create_model.create_model(generated_model, IMG_SHAPE, class_names, LEARNING_RATE, BATCH_SIZE, PARAMS_LOWER_BOUND, PARAMS_UPPER_BOUND)
    if model is not None:
        model.summary()
        train_model.train_model(model, train_ds, val_ds, EPOCHS, PATIENCE, name_of_run, SAVE_DIR)
        print("Evaluating model")
        test_results = model.evaluate(test_ds)

        save_model.save_model(model, name_of_run, test_results, SAVE_DIR)






