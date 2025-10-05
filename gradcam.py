import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import argparse

from models.model import MyCNN
from utils.transforms import get_transforms
from utils.visualize import show_grad_cam
from utils.load_model import load_model

parser = argparse.ArgumentParser(description="Run Grad-CAM on an input image")
parser.add_argument("image_path", type=str, help="Path to the input image")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, idx_to_class = load_model("model.pth")

model.to(device).eval()

target_layer = model.conv_block4[-2]  # ReLU layer

cam_extractor = GradCAM(model, target_layer=target_layer)


image_path = args.image_path
transform = get_transforms(train=False)
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

output = model(input_tensor)

top_p, top_class = output.topk(1, dim=1)

top_class = top_class.item()

activation_map = cam_extractor(top_class, output)
top_class = idx_to_class[top_class]

show_grad_cam(input_tensor=input_tensor, activation_map=activation_map, top_class=top_class)
