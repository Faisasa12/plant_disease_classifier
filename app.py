import streamlit as st
from PIL import Image
import torch

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
target_layer = model.conv_block4[-2]
cam_extractor = GradCAM(model, target_layer=target_layer)
transform = get_transforms(train=False)


st.title("Plant Disease Classifier")
st.markdown("Upload a leaf image to detect disease and view model attention with Grad-CAM.")
st.markdown("**If 224 x 224 image is not used, then image will be resized**")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image)
    confidence, class_idx, predicted_class = predict(model, input_tensor, idx_to_class)
    
    activation_map = cam_extractor(class_idx, model(input_tensor.unsqueeze(0)))
    
    image, heatmap_image = get_grad_cam_image(input_tensor.unsqueeze(0), activation_map)

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    

    st.image(heatmap_image, caption="Grad-CAM Heatmap", use_container_width=True)
