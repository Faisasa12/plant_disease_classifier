import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.transforms import get_transforms
from utils.get_data import get_datasets
from models.model import MyCNN
from utils.visualize import plot_conf_matrix
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms()

dataset, train_set, val_set, test_set = get_datasets()
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

checkpoint_pth = 'checkpoint_epoch_10.pth'
checkpoint = torch.load(checkpoint_pth)



model = MyCNN(num_classes=len(dataset.classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
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
        
        test_acc += torch.mean(equals.float()).item()
        test_loss += loss.item()
        
        all_predictions.extend(top_class.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Loss: {test_loss / len(test_loader):.3f}, Test Accuracy: {test_acc / len(test_loader):.3f}")
print('The Classification report: ')
print(classification_report(all_labels, all_predictions, target_names= dataset.classes))

plot_conf_matrix(all_predictions, all_labels, dataset.classes, fontsize=7)
