import load_data, hyperparameter_tune_model, train_model, prerun_check, save_model, create_model

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
CONV_LAYER_FILTERs = [16, 32, 64]
LEARNING_RATEs = [0.00005, 0.0001, 0.0005, 0.001]
BATCH_SIZEs = [16, 32, 64]
OPTIMIZERs = ["adam", "sgd", "rmsprop"]
ACTIVATIONs = ["relu", "tanh"]
LOSS_FUNCTIONs = ["binary_crossentropy", "mean_squared_error"]


name_of_run = prerun_check.prerun_check(save_dir=SAVE_DIR, check_gpu=CHECK_GPU, delete_previous_runs=DELETE_PREVIOUS_RUNS)

for batch_size in BATCH_SIZEs:
    train_ds, val_ds, test_ds, class_names = load_data.load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, batch_size)

    for first_conv_layer_filter in CONV_LAYER_FILTERs:
        for second_conv_layer_filter in CONV_LAYER_FILTERs:
            for third_conv_layer_filter in CONV_LAYER_FILTERs:
                for optimizer in OPTIMIZERs:
                    for activation in ACTIVATIONs:
                        for loss_function in LOSS_FUNCTIONs:
                            for learning_rate in LEARNING_RATEs:
                                model = hyperparameter_tune_model.create_model(first_conv_layer_filter, second_conv_layer_filter, third_conv_layer_filter, optimizer, activation, loss_function, learning_rate, IMG_SHAPE)
                                model._name += "_batchsize-{}".format(batch_size)
                                model.summary()
                                train_model.train_model(model, train_ds, val_ds, EPOCHS, PATIENCE, name_of_run, SAVE_DIR)
                                print("Evaluating model")
                                test_results = model.evaluate(test_ds)
                                save_model.save_model(model, name_of_run, test_results, SAVE_DIR)






