from augmenter_class import ReshaperAndAugmenter

"""
This script is used to reshape and augment the images inside the train and validation folders. 
It calls reshape and augmenting functions on the images in train and val folders.
"""


aug = ReshaperAndAugmenter()
# Reshape and augment the data
aug.reshape_and_augment("../data/chest-xray-balanced/train/NORMAL/", "../chest-xray-augmented/train/NORMAL/")
aug.reshape_and_augment("../data/chest-xray-balanced/train/PNEUMONIA/", "../chest-xray-augmented/train/PNEUMONIA/")
aug.reshape_and_augment("../data/chest-xray-balanced/val/NORMAL/", "../chest-xray-augmented/val/NORMAL/")
aug.reshape_and_augment("../data/chest-xray-balanced/val/PNEUMONIA/", "../chest-xray-augmented/val/PNEUMONIA/")

# The test images are manually moved from the balanced test folder to the augmented test folder.
