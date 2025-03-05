from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


main = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
main.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load your model
MODEL = tf.keras.models.load_model("../models/1")


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@main.get("/ping")
async def ping():
    return "Hello, I am alive"

# Helper function to read image data
def read_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@main.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Convert the file into an image array
    image = read_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Make the prediction
    prediction = MODEL.predict(img_batch)

    #get the predicted class and confidence
    predicted_class_index = np.argmax(prediction,axis=-1)[0]
    confidence = np.max(prediction,axis=-1)[0]

    #prepare the response

    return{
        "class":CLASS_NAMES[predicted_class_index],
        "confidence":float(confidence)
    }
    

if __name__ == "__main__":
    uvicorn.run(main, host="localhost", port=8005)