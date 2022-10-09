from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np

test_ds = image_dataset_from_directory(
        "../../data/chest-xray-augmented/test",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        color_mode='rgb'
    )

# Load a model
model = load_model("../../saved_model/model.h5")
model.summary()

# Evaluate and predict
model.evaluate(test_ds)
predicitons = model.predict(test_ds)

# Save predictions to a file
import numpy as np
np.savetxt("predictions.csv", predicitons, delimiter=",")


"""
Some selected inputs task
"""
some_selected_inputs_ds = image_dataset_from_directory(
        "../../data/some-selected-inputs",
        image_size=(224, 224),
        batch_size=1,
        color_mode='rgb',
        shuffle=False
    )

print("\nSome selected inputs predictions:")

# Take one image at the time from dataset and print a prediction and label
for image, label in some_selected_inputs_ds:
    print("Prediction: ", model.predict(image))
    print("Actual: ", label.numpy())
    print("")





