# 🚗 Distracted Driver Detection using CNN and Transfer Learning (ResNet50)

This project implements **distracted driver activity classification** using:
1. A **custom Convolutional Neural Network (CNN)**.
2. A **fine-tuned ResNet50** transfer learning model.

It classifies images into 10 categories of driver activities such as safe driving, texting, talking on the phone, operating the radio, etc., based on the **[State Farm Distracted Driver Dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)**.

---

## 📂 Project Structure

```
imgs/
│
├── cnn_dataset/
│   ├── train/
│   │   ├── c0/ img1.jpg, img2.jpg ...
│   │   ├── c1/ ...
│   │   └── c9/
│   ├── val/
│   │   ├── c0/ ...
│   │   └── c9/
│   └── test/
│       ├── c0/ ...
│       └── c9/
│
├── resnet_dataset/
│   ├── train/
│   │   ├── c0/ ...
│   │   └── c9/
│   ├── val/
│   │   ├── c0/ ...
│   │   └── c9/
│   └── test/
│       ├── c0/ ...
│       └── c9/
```

---

## ⚙️ Data Preprocessing

### 1. Image Resizing
- **Simple CNN:** 128×128  
- **ResNet50:** 224×224

### 2. Data Augmentation (using `keras.Sequential` - Applied only during training phase)
- **Random Flip:** Horizontal flip
- **Random Rotation:** ±10%
- **Zoom:** ±10%
- **Contrast Variation:** ±10%
- **Translation:** ±10% height/width shift


### 3. Rescaling
- **CNN:** Pixel values scaled from `[0, 255]` → `[0.0, 1.0]`
- **ResNet50:** Preprocessed using `keras.applications.resnet50.preprocess_input`

---

## 🏗 Model Architectures

### Custom CNN
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for spatial reduction
- Dense layers for classification
- Dropout for regularization
- Output layer with `softmax` activation (10 classes)

### ResNet50 (Transfer Learning)
- Pretrained ResNet50 base (ImageNet weights)
- Top layers replaced with:
  - Flatten / Global Average Pooling
  - Fully connected dense layers (256 → 512 → 1024 neurons)
  - Dropout regularization
  - Final softmax layer for 10 classes
- Top ResNet layers fine-tuned

---

## 📊 Training Results

| Model        | Best Validation Accuracy | Notes |
|--------------|--------------------------|-------|
| CNN          | ~92%                     | Achieved with data augmentation |
| ResNet50 TL  | ~89%                     | Overfitting observed after ~3 epochs |

- EarlyStopping used to avoid overfitting
- ModelCheckpoint to save best model weights

---

## 🖥 Tkinter UI
A simple **GUI application** is included for end-users:
- **Select model**: Simple CNN or ResNet50
- **Upload an image**
- **Prediction output**: Shows whether the driver is distracted and the type of activity

---

## 📌 Future Improvements
- Add more aggressive data augmentation to reduce overfitting
- Experiment with other architectures like **EfficientNet**, **InceptionV3**, **DenseNet**
- Deploy the model as a **web application**

---

## 📜 License
This project is for **educational purposes** only.
