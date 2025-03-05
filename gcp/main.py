from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

# Constants
BUCKET_NAME = "potato-tf-models-3"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL_PATH = "/tmp/model"  # Temporary local path to store downloaded model files

# Initialize model variable
model = None

# Function to download model files
def download_model_files():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # List of model files to download
    model_files = [
        "models/variables/variables.data-00000-of-00001",
        "models/variables/variables.index",
        "models/saved_model.pb"
    ]

    # Download each file
    for file_name in model_files:
        blob = bucket.blob(file_name)
        destination_path = f"{MODEL_PATH}/{file_name.split('/')[-1]}"
        blob.download_to_filename(destination_path)

# Function to load the model
def load_model():
    global model
    if model is None:
        download_model_files()
        model = tf.keras.models.load_model(MODEL_PATH)

# Prediction function
def predict(request):
    # Load model
    load_model()
    
    # Assuming the image is sent as a file in the request
    image_file = request.files["file"]
    image = Image.open(image_file).resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    # Return the prediction result
    return {"predicted_class": predicted_class}