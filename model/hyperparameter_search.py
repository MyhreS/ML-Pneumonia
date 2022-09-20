import load_data, generate_hyperparameter_models, train_model, prerun_check, save_model, create_model

CHECK_GPU = True
DELETE_PREVIOUS_RUNS = True
SAVE_DIR = "hyperparameter_search"
TRAIN_PATH = "../data/chest-xray-dummy/train"
TEST_PATH = "../data/chest-xray-dummy/test"
VAL_PATH = "../data/chest-xray-dummy/val"
IMG_SHAPE = (224, 224, 1)
EPOCHS = 2
PATIENCE = 3

# Hyperparameter_search sizes
Conv_layer_filters = [16, 32, 64]
Dense_layer_sizes = [128, 256, 512]
LEANING_RATEs = [0.00005, 0.0001, 0.0005, 0.001]
BATCH_SIZEs = [16, 32, 64]




name_of_run = prerun_check.prerun_check(save_dir=SAVE_DIR, check_gpu=CHECK_GPU, delete_previous_runs=DELETE_PREVIOUS_RUNS)

train_ds, val_ds, test_ds, class_names = load_data.load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, BATCH_SIZE)

generated_models = generate_hyperparameter_models.generate_models(Conv_layer_filters, Dense_layer_sizes, LEANING_RATEs, BATCH_SIZEs)

for generated_model in generated_models:
    model = create_model.create_model(generated_model, IMG_SHAPE, class_names, LEARNING_RATE)
    if model is not None:
        model.summary()
        train_model.train_model(model, train_ds, val_ds, EPOCHS, PATIENCE, name_of_run, SAVE_DIR)
        print("Evaluating model")
        test_results = model.evaluate(test_ds)

        save_model.save_model(model, name_of_run, test_results, SAVE_DIR)






