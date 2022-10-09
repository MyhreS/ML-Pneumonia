
def save_results(history, test_results, learning_rate, batch_size, this_run_dir_path):
    last_epoch_loss = history.history['loss'][-1]
    last_epoch_accuracy = history.history['accuracy'][-1]
    last_epoch_val_loss = history.history['val_loss'][-1]
    last_epoch_val_accuracy = history.history['val_accuracy'][-1]

    amount_of_epochs = len(history.history['loss'])

    # Save last epoch and test_results to txt file
    with open("{}/results.txt".format(this_run_dir_path), "a") as f:
        # Write learning rate, batch size and amount of epochs in one line
        f.write("Info: learning_rate: {}, batch_size: {}, amount_of_epochs: {}\n".format(learning_rate, batch_size, amount_of_epochs))
        f.write("Last epoch: loss: {}, val_loss: {}, acc: {}, val_acc: {}\n".format(last_epoch_loss, last_epoch_val_loss, last_epoch_accuracy, last_epoch_val_accuracy))
        f.write("Test results: Loss: {}, Accuracy: {}\n".format(test_results[0], test_results[1]))

