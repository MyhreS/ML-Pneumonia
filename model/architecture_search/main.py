from create_directory import create_diretory
from load_data import load_data

NAME_OF_RUN = "1"
TRAIN_PATH = "../../data/chest-xray-dummy/train"
VAL_PATH = "../../data/chest-xray-dummy/val"
TEST_PATH = "../../data/chest-xray-dummy/test"
IMG_SHAPE = (224, 224, 1)
BATCH_SIZE = 32

# Create directory for this run
create_diretory(NAME_OF_RUN)

# Load the data
train_ds, val_ds, test_ds, class_names =  load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, IMG_SHAPE, BATCH_SIZE)
print("Class names:", class_names)

