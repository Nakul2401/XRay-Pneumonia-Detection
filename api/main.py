from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI app
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the trained TensorFlow model
MODEL_PATH = r"C:\Users\nakul\Desktop\Pneumonia\saved_model\3"  # Path to the folder containing saved_model.pb
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize the image to 256x256
    image = image.resize((256, 256))
    # Convert image to array and scale values between 0 and 1
    image_array = np.array(image)
    # Add batch dimension (1, 256, 256, 3)
    if len(image_array.shape) == 2:  # If grayscale, convert to 3 channels
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 1:  # If single channel, convert to 3 channels
        image_array = np.concatenate([image_array] * 3, axis=-1)
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Open and process the uploaded image
        image = Image.open(file.file).convert("RGB")
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        })

# Start the server
# Use the following command to run the server: uvicorn script_name:app --reload
 

if __name__=="__main__":
    uvicorn.run(app , host='localhost', port=8080)
