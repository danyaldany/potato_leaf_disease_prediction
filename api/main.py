from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
)

# Load the model
try:
    MODEL = tf.keras.models.load_model(r"D:\Code\python\Potato_Leaf_Diseas\saved_model\2")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Make sure label order matches training
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


# Image reading function
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((255, 255))  # Change to (224, 224) if model expects that
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        print(f"📷 Image shape: {image.shape}, mean pixel: {image.mean()}")
        return image
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        raise


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

        predictions = MODEL.predict(img_batch)
        print(f"🔍 Raw predictions: {predictions}")

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
