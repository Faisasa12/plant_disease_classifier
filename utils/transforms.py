from torchvision import transforms


def get_mean():
    return [0.485, 0.456, 0.406]

def get_std():
    return [0.229, 0.224, 0.225]
    
def get_transforms():
    return transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=get_mean(),
                             std=get_std())
    ])