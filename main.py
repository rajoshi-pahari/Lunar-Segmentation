import io
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from utils import process_input_image, map_colors
from tensorflow import keras

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained segmentation model
MODEL_PATH = "./LunarModel(1).h5"  # Update with your actual model path
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Error: {e}")

@app.get("/")
async def root():
    """
    Root endpoint to verify server status.
    """
    return {"message": "Lunar Segmentation Server is running"}

@app.post("/segment/")
async def segment_image(file: UploadFile):
    """
    Endpoint to segment the uploaded lunar image.

    Args:
    - file (UploadFile): The uploaded image file.

    Returns:
    - StreamingResponse: The segmented image in PNG format.
    """
    try:
        # Read and preprocess the uploaded image
        image_bytes = await file.read()
        processed_image = process_input_image(io.BytesIO(image_bytes))

        # Add a batch dimension for prediction
        input_tensor = np.expand_dims(processed_image, axis=0)

        # Predict segmentation mask using the model
        predictions = model.predict(input_tensor)
        mask = np.argmax(predictions, axis=-1)[0]  # Get the predicted class for each pixel

        # Map classes to colors
        color_mapped_image = map_colors()[mask]

        # Convert to PIL Image
        segmented_image_pil = Image.fromarray(color_mapped_image)

        # Save the image to a BytesIO object for streaming
        img_byte_arr = io.BytesIO()
        segmented_image_pil.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Return the image as a response
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {e}")