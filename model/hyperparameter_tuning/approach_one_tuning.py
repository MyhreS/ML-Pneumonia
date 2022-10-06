from load_data import load_data
from save_model_info import save_model_info
from save_results import save_results

TRAIN_PATH = "../../data/chest-xray-dummy/train"
VAL_PATH = "../../data/chest-xray-dummy/val"
TEST_PATH = "../../data/chest-xray-dummy/test"
IMG_SHAPE = (224, 224, 1)

BATCH_SIZEs = [32, 64, 128]

for batch_size in BATCH_SIZEs:
    train_ds, val_ds, test_ds, class_names = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, batch_size)
