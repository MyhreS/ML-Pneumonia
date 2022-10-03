
def save_model_info(model, path_to_run_dir):
    """Save the model info to a txt file"""
    with open("{}/model_info.txt".format(path_to_run_dir), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))