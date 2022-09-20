import shutil
import os

def save_model(model, name_of_run, test_results, save_dir="runs"):

    if not os.path.exists("{}/run-{}/models".format(save_dir, name_of_run)):
        os.mkdir("{}/run-{}/models".format(save_dir, name_of_run))

    # Save the model
    model.save("{}/run-{}/models/{}.h5".format(save_dir, name_of_run, model.name))

    # Save the test results to a txt file. Every line is a different model
    with open("{}/run-{}/test-results.txt".format(save_dir, name_of_run), "a") as f:
        f.write("Loss: {}, Accuracy: {}, Name: {}\n".format(test_results[0], test_results[1], model.name))


