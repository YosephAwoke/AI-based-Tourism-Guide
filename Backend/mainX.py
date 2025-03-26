
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import os

import tensorflow as tf

from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Ensure TensorFlow environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model safely
try:
    model = load_model('Hg_MNetV2_2new.h5', compile=False )
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")
print(model.input_shape)

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names
class_names = [
    'Axum Obelisk', 'Bete Amanueal rock hewn church', 'Blue Nile falls', 'Dallol',
    'Debre Dammo Monastery', 'Debre birhan silaseie church', 'Fasil ginbe',
    'Fasiledes bath', 'Gobatit bridge', 'Konso', 'Roha Lalibela',
    'Simien mountains national park', 'Stele of tiya', 'Walled city of harar'
]

# Load heritage information from JSON file
try:
    with open('HeritagesInfo.json', 'r', encoding='utf-8') as file:
        HeritageDescription = json.load(file)
except Exception as e:
    raise RuntimeError(f"Failed to load JSON file: {e}")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize to 224x224 (assuming MobileNetV2 input size)
        img = img.resize((299, 299))
        img_array = np.array(img)

        # Preprocess input for MobileNetV2
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds[0])
        predicted_class_name = class_names[predicted_class]

        # Retrieve heritage info from JSON file
        heritage_info = next(
            (h for h in HeritageDescription["Heritages"] if h["Name"] == predicted_class_name),
            None
        )

        if heritage_info:
            heritage_details = {
                "local_name": heritage_info.get("Local Name", "N/A"),
                "location": heritage_info.get("Location", "N/A"),
                "year_of_construction": heritage_info.get("Year of construction", "N/A"),
                "description": heritage_info.get("Description", "N/A")
            }
        else:
            heritage_details = "No heritage information available"

        return {
            "predicted_class": predicted_class_name,
            "HeritageDescription": heritage_details
        }

    except Exception as e:
        return {"error": str(e)}
