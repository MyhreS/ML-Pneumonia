import shutil
import os

def save_model(model, name_of_run=""):
    if not os.path.exists("runs/run-{}/models".format(name_of_run)):
        os.mkdir("runs/run-{}/models".format(name_of_run))

    # Save the model
    model.save("runs/run-{}/models/{}.h5".format(name_of_run, model.name))

