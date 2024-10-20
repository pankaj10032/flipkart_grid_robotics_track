# import os
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# # Function to apply augmentation on images in a folder
# def augment_images(input_folder, output_folder, augment_count=5):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Define the image augmentation generator
#     datagen = ImageDataGenerator(
#         rotation_range=40,      # Rotate the image by 0-40 degrees
#         width_shift_range=0.2,  # Shift the image horizontally by 0-20% of the width
#         height_shift_range=0.2, # Shift the image vertically by 0-20% of the height
#         shear_range=0.2,        # Shear transformation
#         zoom_range=0.2,         # Zoom in/out
#         horizontal_flip=True,   # Flip the image horizontally
#         fill_mode='nearest'     # Fill missing pixels after transformation
#     )

#     # Loop through all images in the folder
#     for image_file in os.listdir(input_folder):
#         # Load the image
#         image_path = os.path.join(input_folder, image_file)
#         img = load_img(image_path)  # Load image using Keras
#         x = img_to_array(img)       # Convert the image to a numpy array
#         x = np.expand_dims(x, axis=0)  # Add batch dimension

#         # Generate augmented images and save them
#         i = 0
#         for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder,
#                                   save_prefix='aug', save_format='jpg'):
#             i += 1
#             if i >= augment_count:  # Generate specified number of augmentations
#                 break

# # Folder paths
# input_folder = 'Flipkart Dataset/Almonds drops'   # Folder containing original images
# output_folder = 'latest' # Folder to save augmented images

# # Call the function to augment images
# augment_images(input_folder, output_folder, augment_count=5)


# import os
# import json
# import numpy as np
# import easyocr
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from PIL import Image

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Function to apply augmentation and save text in JSON for each image
# def augment_images_with_text(input_folder, output_folder, json_file, augment_count=5):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Define the image augmentation generator
#     datagen = ImageDataGenerator(
#         rotation_range=40,      # Rotate the image by 0-40 degrees
#         width_shift_range=0.2,  # Shift the image horizontally by 0-20% of the width
#         height_shift_range=0.2, # Shift the image vertically by 0-20% of the height
#         shear_range=0.2,        # Shear transformation
#         zoom_range=0.2,         # Zoom in/out
#         horizontal_flip=True,   # Flip the image horizontally
#         fill_mode='nearest'     # Fill missing pixels after transformation
#     )

#     # Dictionary to store text data for each image
#     image_texts = {}

#     # Loop through all images in the folder
#     for image_file in os.listdir(input_folder):
#         image_path = os.path.join(input_folder, image_file)
        
#         # Check if the file is an image (you can extend this for other formats)
#         if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue

#         # Load and process the image
#         img = load_img(image_path)
#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)  # Add batch dimension
        
#         # Perform OCR on the original image and store the extracted text
#         image_text = reader.readtext(image_path, detail=0)
#         text = ' '.join(image_text)  # Convert list of text to a single string
#         image_texts[image_file] = text  # Store the text for the image in the dictionary

#         # Generate augmented images and associate the same text
#         i = 0
#         for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder,
#                                   save_prefix=f'aug_{image_file.split(".")[0]}', save_format='jpg'):
#             i += 1
#             augmented_image_filename = f'aug_{image_file.split(".")[0]}_{i}.jpg'
#             image_texts[augmented_image_filename] = text  # Store the same text for augmented images
#             if i >= augment_count:  # Generate specified number of augmentations
#                 break

#     # Save the image text data to a JSON file
#     with open(json_file, 'w') as f:
#         json.dump(image_texts, f, indent=4)

# # Folder paths
# input_folder = 'Flipkart Dataset/Almonds drops'  # Folder containing original images
# output_folder = 'latest'  # Folder to save augmented images
# json_file = 'image_texts.json'  # JSON file to store image-text data

# # Call the function to augment images and store text data
# augment_images_with_text(input_folder, output_folder, json_file, augment_count=5)


import os
import json
import numpy as np
import easyocr
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to apply augmentation and save text in JSON for all subfolders and images
def augment_images_in_folders(root_folder, output_root, json_file, augment_count=10):
    # Create output root folder if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Dictionary to store text data for each image
    image_texts = {}

    # Loop through each subfolder in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # Skip if it's not a folder
        if not os.path.isdir(subfolder_path):
            continue
        
        # Create corresponding output subfolder
        output_subfolder = os.path.join(output_root, subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Define the image augmentation generator
        datagen = ImageDataGenerator(
            rotation_range=40,      # Rotate the image by 0-40 degrees
            width_shift_range=0.2,  # Shift the image horizontally by 0-20% of the width
            height_shift_range=0.2, # Shift the image vertically by 0-20% of the height
            shear_range=0.2,        # Shear transformation
            zoom_range=0.2,         # Zoom in/out
            horizontal_flip=True,   # Flip the image horizontally
            fill_mode='nearest'     # Fill missing pixels after transformation
        )

        # Loop through all images in the subfolder
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)
            
            # Check if the file is an image (you can extend this for other formats)
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load and process the image
            img = load_img(image_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)  # Add batch dimension

            # Perform OCR on the original image and store the extracted text
            image_text = reader.readtext(image_path, detail=0)
            text = ' '.join(image_text)  # Convert list of text to a single string
            image_texts[image_file] = text  # Store the text for the image in the dictionary

            # Generate augmented images and associate the same text
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_subfolder,
                                      save_prefix=f'aug_{image_file.split(".")[0]}', save_format='jpg'):
                i += 1
                augmented_image_filename = f'aug_{image_file.split(".")[0]}_{i}.jpg'
                image_texts[augmented_image_filename] = text  # Store the same text for augmented images
                if i >= augment_count:  # Generate specified number of augmentations
                    break

    # Save the image text data to a JSON file
    with open(json_file, 'w') as f:
        json.dump(image_texts, f, indent=4)

# Folder paths
root_folder = 'Flipkart Dataset'  # Root folder containing multiple subfolders with images
output_root = 'Augmented Dataset'  # Root folder to save augmented images for each subfolder
json_file = 'image_texts.json'  # JSON file to store image-text data

# Call the function to augment images and store text data for all folders
augment_images_in_folders(root_folder, output_root, json_file, augment_count=10)
