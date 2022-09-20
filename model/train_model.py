from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def train_model(model, train_ds, val_ds, epochs=10, patience=3, name_of_run="", save_dir="runs"):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    logs_dir = "{}/run-{}/logs/{}".format(save_dir, name_of_run, model.name)
    tensorboard = TensorBoard(log_dir=logs_dir) # Use command: tensorboard --logdir logs
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, tensorboard]
    )
    return model
