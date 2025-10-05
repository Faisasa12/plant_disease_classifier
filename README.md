# Plant Disease Classifier (CNN-based)

A deep learning project for detecting **15 plant diseases** using leaf images from the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).  
The project includes model training, evaluation (with detailed metrics), and will soon support interpretability via **Grad-CAM** and a **Streamlit web app**.

---

## Project Goals

- Train a custom Convolutional Neural Network (CNN) on the PlantVillage dataset
- Evaluate the model with:
  - Classification report (precision, recall, F1)
  - Confusion matrix
  - Accuracy and loss curves
- Deploy a user-friendly web app with Streamlit *(coming soon)*

---

## Dataset

- **Source**: [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes**: 15 diseases across Tomato, Potato, Pepper
- **Splits**: 70% train, 20% val, 10% test  
- Images are resized and normalized using standard ImageNet stats.

---

## Model

A custom CNN with:
- 4 convolutional blocks (with BatchNorm + ReLU + MaxPool)
- 2 fully connected layers
- Dropout regularization
- Trained with **NLLLoss** and **AdamW** optimizer

---

## Results

- **Test Accuracy**: ~90%
- **Evaluation Metrics**:
  - Classification report (per class)
  - Confusion matrix
  - Accuracy & loss curves
- See `evaluate.py` and `train_log.csv` for details

---

## How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt

```

### 2. Train the model
```bash
python train.py
```

- Uses **cuda** if it is available, otherwise it will use **cpu**
- Trains for 10 epochs
- Saves model checkpoints to **checkpoint_epoch_{n}.pth**
- Logs training/validation loss and accuracy to **train_log.csv**


### 3. Evaluate the model
```bash
python evaluate.py
```

- Loads the model saved in **checkpoint_epoch_10.pth**
- Evaluates on test set
- Prints the classification report
- Shows the confusion matrix

### 4. Show what the model sees (gradcam)
```bash
python gradcam.py image/path.jpg
```

- **Warning** make sure that your path has no escape characters
- **Warning** if your path has spaces, wrap it in quotes
- Loads the model saved in **checkpoint_epoch_10.pth**
- Predicts the image given as an argument
- shows the original image
- shows the image overlayed with gradcam and the predicted class

## Streamlit app
```bash
streamlit run app.py
```

- Opens a local streamlit webpage to test the model
- You can upload images or choose some of the given examples
- Shows the gradcam of the image

## Coming soon
- A demo video


## Author
Faisal Almofadhi