import torch

def predict(model, image_tensor, class_map, device='cpu'):
    model.to(device).eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    top_p, top_class = torch.exp(output).topk(1, dim=1)
    class_name = class_map[top_class.item()]
    return top_p.item(), top_class.item(), class_name