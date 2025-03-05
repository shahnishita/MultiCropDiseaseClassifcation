from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import logging

main = FastAPI()

# Define constants
endpoint = "http://localhost:8501/v1/models/potato_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@main.get("/ping")
async def ping():
    return "Hello, I am alive"

# Helper function to read image data
def read_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((224, 224))  # Resize to model's input dimensions
    image = np.array(image) / 255.0  # Normalize if needed
    return image.astype(np.float32)  # Ensure dtype matches model input

@main.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Convert the file into an image array
        image = read_as_image(await file.read())
        json_data = {"instances": [image.tolist()]}

        # Make a request to TensorFlow Serving
        response = requests.post(endpoint, json=json_data)
        
        # Check if the response was successful
        if response.status_code != 200:
            logging.error(f"Error from TensorFlow Serving: {response.text}")
            raise HTTPException(status_code=500, detail="Error from TensorFlow Serving")

        # Parse the response
        response_data = response.json()
        
        # Check if 'predictions' key is present in the response
        if "predictions" not in response_data:
            logging.error(f"Unexpected response format: {response_data}")
            raise HTTPException(status_code=500, detail="Unexpected response format from TensorFlow Serving")
        
        # Process the response and find the predicted class
        prediction = response_data["predictions"][0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"Connection error: {e}")
        raise HTTPException(status_code=500, detail="Could not connect to TensorFlow Serving")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction")

if __name__ == "__main__":
    uvicorn.run(main, host="localhost", port=8000)
