import cv2
import glob

def rechape_image(image_path, size):
    image = cv2.imread(image_path)
    # Resize image
    reshaped_image = cv2.resize(image, size)
    return reshaped_image

input_path = "../data/chest-xray-balanced/test/PNEUMONIA"
output_path = "../data/chest-xray-augmented-512/test/PNEUMONIA"
# Get paths to images
input_images_paths = glob.glob(input_path + "/*")

# Reshape the images
for input_image_path in input_images_paths:
    # Reshape image
    image = rechape_image(input_image_path, (512, 512))
    # Write image
    output_image_path = output_path + "/" + input_image_path.split("\\")[-1]
    cv2.imwrite(output_image_path, image) # Write image
