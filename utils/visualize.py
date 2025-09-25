import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transforms import get_mean, get_std

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

def plot_conf_matrix(preds, labels, classes):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45)
    plt.show()
