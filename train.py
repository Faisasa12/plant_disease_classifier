import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import csv

from models.model import MyCNN
from utils.transforms import get_transforms
from utils.get_data import get_datasets

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset, train_set, val_set, test_set = get_datasets()
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

model = MyCNN(num_classes=len(dataset.classes)).to(device)
model.class_to_idx = dataset.class_to_idx

criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003)

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output = model(images)
        
        loss = criterion(output, labels)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()

    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            
            loss = criterion(output, labels)
            
            top_p, top_class = torch.exp(output).topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            val_acc += torch.mean(equals.float()).item()
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss / len(train_loader):.3f}, "
          f"Val loss: {val_loss / len(val_loader):.3f}, Accuracy: {val_acc / len(val_loader):.3f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, f'checkpoint_epoch_{epoch+1}.pth')
    
    with open('train_log.csv', 'a') as file:
        writer = csv.writer(file)

        if epoch == 0:
            writer.writerow(["Epoch", 'TrainLoss', 'ValLoss', 'ValAcc'])
            
        writer.writerow([
            epoch + 1,
            train_loss / len(train_loader),
            val_loss / len(val_loader),
            val_acc / len(val_loader)
        ])