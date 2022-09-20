import shutil
import os

def save_model(model, name_of_run, test_results):

    if not os.path.exists("runs/run-{}/models".format(name_of_run)):
        os.mkdir("runs/run-{}/models".format(name_of_run))

    # Save the model
    model.save("runs/run-{}/models/{}.h5".format(name_of_run, model.name))

    # Save the test results to a txt file. Every line is a different model
    with open("runs/run-{}/test-results.txt".format(name_of_run), "a") as f:
        f.write("Loss: {}, Accuracy: {}, Name: {}\n".format(test_results[0], test_results[1], model.name))


