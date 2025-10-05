import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

from models.model import MyCNN
from utils.transforms import get_transforms
from utils.visualize import show_grad_cam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN(num_classes=15)
checkpoint = torch.load("checkpoint_epoch_10.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
idx_to_class = {idx: class_name for class_name, idx in model.class_to_idx.items()}

model.to(device).eval()

target_layer = model.conv_block4[-2]  # ReLU layer

cam_extractor = GradCAM(model, target_layer=target_layer)


image_path = r"data\PlantVillage\Pepper__bell___Bacterial_spot\0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"
transform = get_transforms(train=False)
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

output = model(input_tensor)

top_p, top_class = output.topk(1, dim=1)

top_class = top_class.item()

activation_map = cam_extractor(top_class, output)
top_class = idx_to_class[top_class]

show_grad_cam(input_tensor=input_tensor, activation_map=activation_map, top_class=top_class)
