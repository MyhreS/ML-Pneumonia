import os
import tensorflow as tf
import shutil


def prerun_check(name_of_run, check_gpu = True, delete_previous_runs = False):
    # Check if runs folder exists
    if not os.path.exists("runs"):
        os.mkdir("runs")
    elif delete_previous_runs:
        shutil.rmtree("runs")
        os.mkdir("runs")
    # Make a folder for this run
    os.mkdir("runs/run-{}".format(name_of_run))

    # Check if GPU is available
    if check_gpu:
        print("--GPU--")
        print(tf.test.gpu_device_name())




