from augmenterClass import ReshapeAndAugmenter

aug = ReshapeAndAugmenter()

# Reshape and augment the data
aug.reshape_and_augment("../chest-xray-balanced/val/NORMAL/", "../chest-xray-augmented/val/NORMAL/")

