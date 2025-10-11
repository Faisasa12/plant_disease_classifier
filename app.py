import streamlit as st
from PIL import Image
import torch
import os

from models.model import MyCNN
from utils.transforms import get_transforms
from utils.predict import predict
from utils.visualize import get_grad_cam_image
from utils.load_model import load_model

from torchcam.methods import GradCAM


@st.cache_resource
def get_model():
    return load_model()

model, idx_to_class = get_model()
target_layer = model.conv_block4[-2] # ReLu layer
cam_extractor = GradCAM(model, target_layer=target_layer)
transform = get_transforms(train=False)


st.title("Plant Disease Classifier")
st.markdown("Upload a leaf image to detect disease and view model attention with Grad-CAM.")
st.markdown("**If 224 x 224 image is not used, then image will be resized**")

col1, col2 = st.columns(2)

EXAMPLE_IMAGE_DIR = "examples"

example_images = {
    "Tomato Leaf Mold correct": "0a9b3ff4-5343-4814-ac2c-fdb3613d4e4d___Crnl_L.Mold 6559.JPG",
    "Tomato Leaf Mold wrong": "1fa78650-4f81-4c8f-9190-91f1057d1158___Crnl_L.Mold 9083.JPG",
    "Potato healthy wrong": "0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG",
    "Potato healthy correct": "2ccb9ee9-faac-4d32-9af5-29497fa2e028___RS_HL 1837.JPG"
}

example_choice = st.selectbox("Or choose an example image", ["None"] + list(example_images.keys()))
is_example = False

if example_choice != "None":
    is_example = True
    example_path = os.path.join(EXAMPLE_IMAGE_DIR, example_images[example_choice])
    
    example_image = Image.open(example_path).convert("RGB")

    input_image = example_image
else:
    input_image = None
    
    
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    is_example = False    


if input_image:
    with col1:
        if is_example:
            st.image(example_image, caption=f"Example: {example_choice}", use_container_width=True)
            
        else:
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
        
    input_tensor = transform(input_image)
    confidence, class_idx, predicted_class = predict(model, input_tensor, idx_to_class)
    
    activation_map = cam_extractor(class_idx, model(input_tensor.unsqueeze(0)))
    
    image, heatmap_image = get_grad_cam_image(input_tensor.unsqueeze(0), activation_map)

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    
    with col2:
        st.image(heatmap_image, caption="Grad-CAM Heatmap", use_container_width=True)
