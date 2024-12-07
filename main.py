import io
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from tensorflow import keras
from utils import input_image, map_colours

# Initialize the FastAPI app
app = FastAPI()

# Constants
MODEL_PATH = "./LunarModel(1).h5"  # Update with your actual model path

# Load the model at startup
def load_model(model_path):
    """
    Load the pre-trained model from the given path.
    """
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")

model = load_model(MODEL_PATH)

# Utility functions
def preprocess_image(file_data):
    """
    Preprocess the uploaded image for model inference.
    
    Args:
    - file_data (BytesIO): The raw image data.

    Returns:
    - np.ndarray: The processed image ready for prediction.
    """
    return input_image(file_data)

def generate_mask(input_tensor):
    """
    Generate the segmentation mask for the given input.

    Args:
    - input_tensor (np.ndarray): The preprocessed image with a batch dimension.

    Returns:
    - np.ndarray: The predicted segmentation mask.
    """
    predictions = model.predict(input_tensor)
    return np.argmax(predictions, axis=-1)[0]

def convert_to_response(mask):
    """
    Convert the segmentation mask to a colour-mapped image and return it as a StreamingResponse.

    Args:
    - mask (np.ndarray): The segmentation mask.

    Returns:
    - StreamingResponse: The colour-mapped image as a PNG file.
    """
    # Map classes to colours
    colour_mapped_image = map_colours()[mask]

    # Convert to PIL Image
    segmented_img_pil = Image.fromarray(colour_mapped_image)

    # Save to BytesIO for streaming
    img_byte_arr = io.BytesIO()
    segmented_img_pil.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

# API Endpoints
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
        img_bytes = await file.read()
        processed_img = preprocess_image(io.BytesIO(img_bytes))

        # Add batch dimension for prediction
        input_tensor = np.expand_dims(processed_img, axis=0)

        # Generate segmentation mask
        mask = generate_mask(input_tensor)

        # Convert mask to a response
        return convert_to_response(mask)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {e}")
