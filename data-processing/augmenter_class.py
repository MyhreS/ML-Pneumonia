import glob
import itertools
import cv2
import shutil
import os
from imgaug import augmenters as iaa

"""
This class is used to resize and augment images and save them to a new directory.
"""
class ReshaperAndAugmenter:
    def __init__(self):
        self.augment_combinations = self.calculate_combinations()

    """
    This function returnes the different types of augmentation that will be used.
    """
    def get_augmentation_types(self):
        return [
            iaa.SaltAndPepper(0.05),
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.GaussianBlur(sigma=(0, 3.0)),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.Affine(rotate=(-90, 90)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Pad(px=(0, 50), keep_size=False),
            iaa.Crop(percent=(0, 0.1)),
            iaa.PerspectiveTransform(scale=(0.01, 0.1))
        ]

    """
    This function calculates the different combinations of the augmenation types. All the combinations
    will be used in the augmentation of the images.
    """
    def calculate_combinations(self):
        augment_types = self.get_augmentation_types()
        # Create dummylist with index 1 to 10 using range
        dummy_list = list(range(0, len(augment_types)))
        # Create all combinations the dummylist with 1 to 3 items
        dummy_combinations = []
        for i in range(1, 3):
            dummy_combinations.extend(list(itertools.combinations(dummy_list, i)))

        # Create sequences with the combinations
        sequences = []
        for dummy_combination in dummy_combinations:
            sequence = []
            for i in dummy_combination:
                sequence.append(augment_types[i])
            sequences.append(iaa.Sequential(sequence))
        return sequences


    """
    When called uppon, this function will reshape and augment the images in the input folder and save them to the output folder.
    """
    def reshape_and_augment(self, input_path, output_path):
        # Remove old folder if it exists and make new folder.
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        input_images_paths = glob.glob(input_path + "/*")

        name = 0 # Name of the image
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
                output_image_path = output_path + str(name) + ".png" # Path to write image
                cv2.imwrite(output_image_path, image_created) # Write image
                name += 1 # Increment name

                # For every 100 images, print a dot to show progress
                if name % 100 == 0:
                    print(output_image_path)
        print("Augment finished")

    """
    This function will apply augmentation to an image and return a list of augmenged images from the image.
    """
    def augment(self, reshaped_image):
        images_augmented = []
        # Augmenting types
        for augment_combination in self.augment_combinations:
            images_augmented.append(augment_combination(image=reshaped_image))

        return images_augmented

