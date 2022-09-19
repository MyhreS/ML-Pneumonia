from tensorflow.keras.utils import image_dataset_from_directory

def load_data(train_path, val_path, test_path, image_size, batch_size, color_mode):
    train_ds = image_dataset_from_directory(
        train_path,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode
    )
    val_ds = image_dataset_from_directory(
        val_path,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode
    )
    test_ds = image_dataset_from_directory(
        test_path,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode
    )
    class_names = train_ds.class_names
    print("Class names:", class_names)
    return train_ds, val_ds, test_ds, class_names