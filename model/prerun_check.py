import os
import tensorflow as tf
import shutil


def prerun_check(check_gpu = True, delete_previous_runs = False):
    name_of_run = ""
    # Check if runs folder exists
    if not os.path.exists("runs"):
        os.mkdir("runs")
        name_of_run = "run_1"
    elif delete_previous_runs:
        shutil.rmtree("runs")
        os.mkdir("runs")
    if os.path.exists("runs"):
        # Get the number of runs
        num_of_runs = len(os.listdir("runs"))
        name_of_run = str(num_of_runs)




    # Make a folder for this run
    os.mkdir("runs/run-{}".format(name_of_run))

    # Check if GPU is available
    if check_gpu:
        print("--GPU--")
        print(tf.test.gpu_device_name())

    return name_of_run




