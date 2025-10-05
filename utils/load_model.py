from models.model import MyCNN
import torch

def load_model(pth= "checkpoint_epoch_10.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MyCNN(num_classes=15)
    checkpoint = torch.load(pth, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    
    return model, idx_to_class