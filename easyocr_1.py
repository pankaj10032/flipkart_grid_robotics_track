import easyocr

# Initialize the reader with the languages you want (e.g., English)
reader = easyocr.Reader(['en'])

# Path to your image
image_path = 'Flipkart Dataset/Pears/51o2w8K5CWL._SX522_.jpg'

# Extract text from the image
results = reader.readtext(image_path)

# Print the results
for result in results:
    # result[1] contains the detected text
    print(result[1])
