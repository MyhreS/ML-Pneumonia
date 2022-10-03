import os
import pathlib
import shutil

def create_diretory(name_of_run):
    # Create a "runs" directory if it does not already exist

    current_directory = pathlib.Path(__file__).parent.absolute()
    runs_dir = os.path.join(current_directory, "model_runs")
    if not os.path.exists(runs_dir):
        os.makedirs("model_runs")

    # Create subdirectory for this run

    this_run_dir = os.path.join(runs_dir, name_of_run)
    print(this_run_dir)
    if not os.path.exists(this_run_dir):
        os.makedirs(this_run_dir)
    else:
        # Cast error that stops the program
        raise Exception("Directory/run already exists. Rename the run or delete the directory.")

    # Create subdirectory storing the models results
    results_dir = os.path.join(this_run_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create subsubdirectory for saving the tensorboard logs
    tensorboard_dir = os.path.join(this_run_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # Create subsubdiretory for saving the info about the model
    model_info_dir = os.path.join(this_run_dir, "model_info")
    if not os.path.exists(model_info_dir):
        os.makedirs(model_info_dir)

    # Create subsubdiretory for my comments on the model
    comments_dir = os.path.join(this_run_dir, "comments")
    if not os.path.exists(comments_dir):
        os.makedirs(comments_dir)
