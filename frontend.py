import streamlit as st
import requests
from PIL import Image
import io

# FastAPI backend endpoint URL
API_URL = "http://localhost:8000/segment/"

# Set up Streamlit page configuration
st.set_page_config(page_title="Lunar Image Segmentation", page_icon="🌕", layout="wide")

# Background CSS for adding a background image
bg_image_url = "https://images.unsplash.com/photo-1451188214936-ec16af5ca155?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NDd8fGx1bmFyJTIwc3VyZmFjZXxlbnwwfHwwfHx8MA%3D%3D"  # Replace with your image URL or local path
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 style="color: lightgray; text-align: center;">🌙 Lunar Terrain Segmentation App</h1>', unsafe_allow_html=True)

# Initialize session state for storing images
if 'segmented_result' not in st.session_state:
    st.session_state['segmented_result'] = None
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

# Display the "Upload an Image" button and image uploader when clicked
image_file = st.file_uploader("Upload an image of the moon's surface", type=["jpg", "jpeg", "png", "bmp"])

if image_file is not None:
    # Save the uploaded image in session state
    st.session_state['uploaded_image'] = image_file

    # Display the uploaded image
    st.image(image_file, caption="Uploaded Image", use_container_width=True)

    # Display the "Segment Image" button after the image is uploaded
    segment_btn = st.button("Segment Image")

    # When the segment button is clicked, call the FastAPI backend for segmentation
    if segment_btn:
        # Send the uploaded image to the FastAPI backend for segmentation
        try:
            # Convert the uploaded image to bytes
            files = {'file': image_file.getvalue()}
            st.write("Segmenting....")
            response = requests.post(API_URL, files=files)

            st.write("Tried my best :')")

            # Check if the request was successful
            if response.status_code == 200:
                # Convert the segmented image from the response
                segmented_result = Image.open(io.BytesIO(response.content))
                st.session_state['segmented_result'] = segmented_result  # Store the segmented image in session state
            else:
                st.error("Error in segmentation: " + response.json().get("detail", "Unknown error"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error during request: {e}")

# Display the segmented image if available
if st.session_state['segmented_result'] is not None:
    st.image(st.session_state['segmented_result'], caption="Segmented Image", use_container_width=True)

# Add a footer to the page
footer_content = """
<div style='position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; text-align: center; padding: 10px;'>
    <p style='color: black; margin: 0;'>This project is developed by <b>Rajoshi Pahari</b> as part of the <b>Machine Learning for Astronomy</b> Training Program by Spartifical</p>
</div>
"""

# Inject the footer using markdown
st.markdown(footer_content, unsafe_allow_html=True)
