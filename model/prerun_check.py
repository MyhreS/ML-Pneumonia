import os
import tensorflow as tf
import shutil


def prerun_check(save_dir="runs", check_gpu = True, delete_previous_runs = False):
    name_of_run = ""
    # Check if runs folder exists
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        name_of_run = "run_1"
    elif delete_previous_runs:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    if os.path.exists(save_dir):
        # Get the number of runs
        num_of_runs = len(os.listdir(save_dir))
        name_of_run = str(num_of_runs)




    # Make a folder for this run
    os.mkdir("{}/run-{}".format(save_dir, name_of_run))

    # Check if GPU is available
    if check_gpu:
        print(tf.test.gpu_device_name())

    return name_of_run




