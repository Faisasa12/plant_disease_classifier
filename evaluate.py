import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.transforms import get_transforms
from models.model import MyCNN
from utils.visualize import plot_conf_matrix
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms()
dataset_path = 'data/PlantVillage'

dataset = datasets.ImageFolder(dataset_path, transform=transform)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(0.7 * len(dataset)), int(0.2 * len(dataset)), int(0.1 * len(dataset))])
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = MyCNN(num_classes=len(dataset.classes))
model.load_state_dict(torch.load('checkpoint_epoch_10.pth')['model_state_dict'])
model.to(device)
model.eval()

criterion = nn.NLLLoss()
test_loss, test_acc = 0, 0
all_predictions, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        
        top_p, top_class = torch.exp(output).topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        test_acc += torch.mean(equals.type(torch.FloatTensor))
        test_loss += loss.item()
        
        all_predictions.extend(top_class.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Loss: {test_loss / len(test_loader):.3f}, Test Accuracy: {test_acc / len(test_loader):.3f}")
plot_conf_matrix(all_predictions, all_labels, dataset.classes)
