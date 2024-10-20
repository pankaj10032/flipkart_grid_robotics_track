import os
import numpy as np
import json
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Embedding, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load image and text data from the dataset
def load_data_and_text(image_root_folder, json_file):
    images = []
    texts = []
    labels = []

    # Load OCR text from the JSON file
    with open(json_file, 'r') as f:
        image_texts = json.load(f)
    
    label_map = {label: idx for idx, label in enumerate(os.listdir(image_root_folder))}

    # Loop through all subfolders (representing classes)
    for subfolder in os.listdir(image_root_folder):
        subfolder_path = os.path.join(image_root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        label_idx = label_map[subfolder]

        # Loop through all images in the subfolder
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                img = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
                img_array = img_to_array(img)
                img_array = img_array / 255.0  # Normalize
                images.append(img_array)

                # Load text
                text = image_texts.get(image_file, "")
                texts.append(text)

                # Assign label (based on the subfolder)
                labels.append(label_idx)

    return np.array(images), texts, np.array(labels)

# Prepare image model (CNN for image features)
def build_image_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)  # Global average pooling instead of flatten
    image_model = Model(inputs=base_model.input, outputs=x)
    return image_model

# Prepare text model (RNN for text features)
def build_text_model(vocab_size, max_length, embedding_dim=100):
    text_input = Input(shape=(max_length,), name="text_input")
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(text_input)
    rnn_out = GRU(128)(embedding)  # Use GRU or LSTM
    return text_input, rnn_out

# Combine image and text features and build the final model
def build_combined_model(image_model, text_input, text_output, num_classes):
    # Combine image and text features
    combined = Concatenate()([image_model.output, text_output])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    final_output = Dense(num_classes, activation='softmax')(combined)

    # Build the final model
    model = Model(inputs=[image_model.input, text_input], outputs=final_output)
    return model

# Load and preprocess data
image_root_folder = 'Augmented Dataset'  # Folder containing multiple subfolders (each subfolder is a class)
json_file = 'image_texts.json'

# Load images, texts, and labels
images, texts, labels = load_data_and_text(image_root_folder, json_file)

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=10000)  # Maximum vocabulary size
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = 100  # Maximum length for padding sequences
text_data = pad_sequences(sequences, maxlen=max_length)

# One-hot encode labels
num_classes = len(set(labels))
print(num_classes)
labels = to_categorical(labels, num_classes=num_classes)

# Split the data into training and testing sets
X_train_img, X_test_img, X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    images, text_data, labels, test_size=0.2, random_state=42
)

# Build models
image_model = build_image_model()
text_input, text_output = build_text_model(vocab_size=10000, max_length=max_length)

# Build the combined model
model = build_combined_model(image_model, text_input, text_output, num_classes=num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit([X_train_img, X_train_txt], y_train, epochs=10, batch_size=32, validation_data=([X_test_img, X_test_txt], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test_img, X_test_txt], y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
