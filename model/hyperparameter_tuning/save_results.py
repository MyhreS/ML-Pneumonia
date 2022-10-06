import os

def save_results(test_results, name_of_model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    results_file = os.path.join(path, "results.txt")
    with open(results_file, "a") as f:
        f.write("Loss: {}, acc: {}\n".format(name_of_model, test_results))

