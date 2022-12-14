import os
import pathlib
import shutil

def create_directory(name_of_run, delete_similiar_run=False):
    # Absolute path to this directory
    current_directory = pathlib.Path(__file__).parent.absolute()
    runs_dir = os.path.join(current_directory, "../model_runs")

    # Create a "model_runs" directory if it does not already exist
    if not os.path.exists(runs_dir):
        os.makedirs("../model_runs")



    this_run_dir = os.path.join(runs_dir, name_of_run)
    # Delete the directory if it already exists
    if os.path.exists(this_run_dir):
        if delete_similiar_run:
            shutil.rmtree(this_run_dir)

    # Create subdirectory for this run
    if not os.path.exists(this_run_dir):
        os.makedirs(this_run_dir)
    else:
        # Cast error that stops the program
        raise Exception("Directory/run already exists. Rename the run or delete the directory.")

    # Create a txt file storing the models results
    results_file = os.path.join(this_run_dir, "results.txt")
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("")

    # Create subsubdirectory for saving the tensorboard logs
    tensorboard_dir = os.path.join(this_run_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # Create subsubdiretory for saving the info about the model
    model_info_file = os.path.join(this_run_dir, "model_info.txt")
    if not os.path.exists(model_info_file):
        with open(model_info_file, "w") as f:
            f.write("")

    # Create a txt file for my comments
    comments_file = os.path.join(this_run_dir, "comments.txt")
    if not os.path.exists(comments_file):
        with open(comments_file, "w") as f:
            f.write("")

    return this_run_dir