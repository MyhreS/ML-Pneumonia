import load_data, create_models, train_model, prerun_check, save_model

CHECK_GPU = True
DELETE_PREVIOUS_RUNS = True
TRAIN_PATH = "../data/chest-xray-dummy/train"
TEST_PATH = "../data/chest-xray-dummy/test"
VAL_PATH = "../data/chest-xray-dummy/val"
BATCH_SIZE = 32
IMG_SHAPE = (224, 224, 1)
EPOCHS = 2
PATIENCE = 3




name_of_run = prerun_check.prerun_check(check_gpu=CHECK_GPU, delete_previous_runs=DELETE_PREVIOUS_RUNS)

train_ds, val_ds, test_ds, class_names = load_data.load_data("../data/chest-xray-dummy/train", "../data/chest-xray-dummy/val", "../data/chest-xray-dummy/test", IMG_SHAPE, BATCH_SIZE)

models = create_models.create_models(IMG_SHAPE, class_names)

for model in models:
    model.summary()
    model = train_model.train_model(model, train_ds, val_ds, epochs=EPOCHS, patience=PATIENCE, name_of_run=name_of_run)
    print("Evaluating model")
    test_results = model.evaluate(test_ds)

    save_model.save_model(model, name_of_run=name_of_run)









