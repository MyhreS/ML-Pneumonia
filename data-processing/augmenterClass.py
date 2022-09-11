import glob
import cv2
import shutil
import os

class ReshapeAndAugmenter:
    def reshape_and_augment(self, input_path, output_path):
        # Remove old folder if it exists and make new folder.
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        input_images_paths = glob.glob(input_path + "/*")

        # Reshape the images
        for input_image_path in input_images_paths:
            images_created = []
            # Read image
            image = cv2.imread(input_image_path)
            # Resize image
            reshaped_image = cv2.resize(image, (224, 224))
            # Add image to list
            images_created.append(reshaped_image)
            # Augment images
            images_created.extend(self.augment(reshaped_image))

            # Write image
            for image_created in images_created:
                output_image_path = output_path + "/" + os.path.basename(input_image_path)
                cv2.imwrite(output_image_path, image_created)
        print("Augment finished")


    def augment(self, reshaped_images):
        images_augmented = []

        return images_augmented

