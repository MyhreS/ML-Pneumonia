from tensorflow.keras.utils import image_dataset_from_directory

def load_data(train_path, val_path, test_path, img_shape, batch_size):
    if img_shape[2] == 1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'

    """Load the data from the given paths and return the datasets"""
    train_ds = image_dataset_from_directory(
        train_path,
        seed=123,
        image_size=(img_shape[0], img_shape[1]),
        batch_size=batch_size,
        color_mode=color_mode
    )
    val_ds = image_dataset_from_directory(
        val_path,
        seed=123,
        image_size=(img_shape[0], img_shape[1]),
        batch_size=batch_size,
        color_mode=color_mode
    )
    test_ds = image_dataset_from_directory(
        test_path,
        seed=123,
        image_size=(img_shape[0], img_shape[1]),
        batch_size=batch_size,
        color_mode=color_mode
    )

    class_names = train_ds.class_names

    return train_ds, val_ds, test_ds, class_names