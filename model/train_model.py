from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def train_model(model, train_ds, val_ds, epochs=10, patience=3):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    tensorboard = TensorBoard(log_dir="logs/{}".format(model._name)) # Use command: tensorboard --logdir logs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, tensorboard]
    )
    return model, history
