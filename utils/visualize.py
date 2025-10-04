import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.transforms import get_mean, get_std
import pandas as pd
from torchvision.transforms import ToPILImage
from torchcam.utils import overlay_mask

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))
    mean = np.array(get_mean())
    std = np.array(get_std())
    
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def plot_conf_matrix(preds, labels, classes, fontsize = 10):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10,10))
    
    disp.plot(ax=ax, xticks_rotation=90)
    
    ax.tick_params(axis='both', labelsize= fontsize)
    
    plt.subplots_adjust(bottom=0.4)
    plt.show()


def plot_training_curves(csv_file= 'train_log.csv'):
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(df['Epoch'], df['TrainLoss'], label='Train Loss')
    ax[0].plot(df['Epoch'], df['ValLoss'], label='Val Loss')
    ax[0].legend()
    ax[0].set_title("Loss Curves")

    ax[1].plot(df['Epoch'], df['ValAcc'], label='Validation Accuracy', color='green')
    ax[1].legend()
    ax[1].set_title("Accuracy Curve")

    plt.show()
    
def show_grad_cam(input_tensor, activation_map, top_class):
    to_pil = ToPILImage()
    input_image = to_pil(input_tensor.squeeze(0))


    result = overlay_mask(input_image, to_pil(activation_map[0].squeeze(0)), alpha=0.5)

    plt.imshow(result)
    plt.title(f"Grad-CAM - Predicted Class: {top_class}")
    plt.axis('off')
    plt.show()