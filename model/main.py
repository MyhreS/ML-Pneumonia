import load_data, create_models, train_model, prerun_check, save_model
# Generate a unique name for this run

name_of_run = prerun_check.prerun_check(check_gpu=True, delete_previous_runs=False)

train_ds, val_ds, test_ds, class_names = load_data.load_data("../data/chest-xray-dummy/train", "../data/chest-xray-dummy/val", "../data/chest-xray-dummy/test", (224, 224), 32, 'grayscale')

models = create_models.create_models((224, 224, 1), class_names)

for model in models:
    model.summary()
    model = train_model.train_model(model, train_ds, val_ds, epochs=2, patience=3, name_of_run=name_of_run)
    print("Evaluating model")
    test_results = model.evaluate(test_ds)

    save_model.save_model(model, name_of_run=name_of_run)









