# 🚗 Distracted Driver Detection using CNN and Transfer Learning (ResNet50)
[![GitHub last commit](https://img.shields.io/github/last-commit/aispai9995/driver-distraction-detection-python-CNN?color=green&logo=github&style=for-the-badge)](https://github.comaispai9995/driver-distraction-detection-python-CNN) 
[![GitHub top language](https://img.shields.io/github/languages/top/aispai9995/driver-distraction-detection-python-CNN?color=F37626&logo=jupyter&style=for-the-badge)](https://github.com/aispai9995/driver-distraction-detection-python-CNN) 
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/aispai9995/driver-distraction-detection-python-CNN?color=blue&logo=python&style=for-the-badge)](https://github.com/aispai9995/driver-distraction-detection-python-CNN) 


## Objective
This is an old competition from [Kaggle](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data).

In this competition, we are given driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc). The goal is to predict whether the driver is distracted or not from the image of the driver using Convolutional Neural Network (CNN) Models. If distracted, the model classifies the image into one of the classes below, which identifies the distracting activity.

The 10 classes to predict from are: <br>
c0: safe driving <br>
c1: texting - right <br>
c2: talking on the phone - right <br>
c3: texting - left <br>
c4: talking on the phone - left <br>
c5: operating the radio <br>
c6: drinking <br>
c7: reaching behind <br>
c8: hair and makeup <br>
c9: talking to passenger <br>

## File descriptions
<li> imgs - Folder of all (train/test) images </li>
<li> sample_submission.csv - a sample submission file in the correct format </li> 
<li> driver_imgs_list.csv - a list of training images, their subject (driver) id, and class id </li> 
<br><br>
In this case, only the images from the train folder present inside the imgs folder were used for the model training and evaluation. As the test images in the original dataset were not labeled, they were only used during the final predictions when interacting with the models through Tkinter UI.
<br><br>
This project implements **distracted driver activity classification** using:
1. A **custom Convolutional Neural Network (CNN)**.
2. A **fine-tuned ResNet50** transfer learning model.

## 📂 Directory Structure

The train images (22424 images) in the original dataset were imported and organized into train (60%), test (20%), and validation (20%) as shown in the directory structure below. The preprocessing steps were different for the two models. Hence, 2 copies of the dataset were created, one for CNN and the second for ResNet50.

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

## ⚙️ Data Preprocessing

### 1. Image Resizing
- **Simple CNN:** 128×128  
- **ResNet50:** 224×224

### 2. Data Augmentation (using `keras.Sequential` - applied only during training phase)
- **Random Flip:** Horizontal flip
- **Random Rotation:** ±10%
- **Zoom:** ±10%
- **Contrast Variation:** ±10%
- **Translation:** ±10% height/width shift

### 3. Rescaling
- **CNN:** Pixel values scaled from `[0, 255]` → `[0.0, 1.0]`
- **ResNet50:** Preprocessed using `keras.applications.resnet50.preprocess_input`

## 🕸️ Model Architectures

### Custom CNN
- A deep layered approach with multiple convolutional and pooling operations, followed by dense layers for classification.
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for spatial reduction
- Dense layers for classification
- Dropout for regularization
- Output layer with `softmax` activation (10 classes)

  <img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/7cad67d2-3aa6-4640-9f2f-046b1cc4731c" />

### ResNet50 (Transfer Learning)
- Pretrained ResNet50 base (ImageNet weights)
- Top layers replaced with:
  - Flatten
  - Fully connected dense layers (256 → 512 neurons)
  - Dropout regularization
  - Final softmax layer for 10 classes
- Feature extraction with 3 trainable layers and Fine-tuning with 35 trainable layers

<img width="700" height="200" alt="image" src="https://github.com/user-attachments/assets/2242a07f-c5fd-412b-8ef8-57ab137bde4c" />

## 📦 Hyperparameters and Tuning:
### Custom CNN
<li> Optimizer: Adam  with a learning_rate of 0.0001 </li>
<li> Loss Function: sparse_categorical_crossentropy </li>
<li> Metrics: Accuracy </li>
<li> Epochs: 20 </li>
<li> Input Size: 128 × 128 × 3 </li>
<li> EarlyStopping was also used during training with a patience of 8. It is a regularization technique to prevent overfitting and reduce training time.

### ResNet50 (Transfer Learning)
The transfer learning process with ResNet 50 had 2 phases – Feature Extraction and Finetuning. The following optimal parameters were identified during the training process.
<li> Base Model: ResNet50 with ImageNet weights, include_top=False  </li>
<li> Trainable Layers: 3 for feature extraction, 35 for fine-tuning </li>
<li> Optimizer: Adam with a learning_rate of 1e-4 for feature extraction and 1e-5 for fine-tuning </li>
<li> Loss Function: sparse_categorical_crossentropy </li>
<li> Metrics: Accuracy </li>
<li> Epochs: 5 for feature extraction and 5 for fine-tuning </li>
<li> Input Size: 224 × 224 × 3 </li>


## 📊 Training Results

| Model        | Training Accuracy | Validation Accuracy | Test Accuracy |
|--------------|-------------------|---------------------|---------------|
| CNN          | 99.30%            | 89.12%              | 93.75%        |
| ResNet50 TL  | 99.49%            | 94.34%              | 93.93%        |


## 🖥 Tkinter UI
A simple **GUI application** is created for end-users:
- **Select model**: Simple CNN or ResNet50
- **Upload an image**
- **Prediction output**: Shows whether the driver is distracted and the type of activity

<img width="300" height="600" alt="image" src="https://github.com/user-attachments/assets/417b1534-7b4c-48ac-a022-4841b4f67088" />

## Inferences
-	ResNet50 achieved higher accuracy than custom CNN on the test set, indicating the effectiveness of transfer learning. It analyzed even the minor features correctly and made accurate predictions between similar activities (e.g., Texting Left vs. Talking on Phone Left).
-	"Safe Driving" class was the most accurately predicted by Custom CNN; misclassifications often occurred for similar distracting activities.

## Learnings
- Transfer learning with pretrained models like ResNet drastically improves performance with limited data and training time.
-	Preprocessing (rescaling, image resizing, and augmentation) played a crucial role in model performance consistency.
-	Model predictions are sensitive to image quality and orientation; real-world application needs stricter input controls or augmentation strategies.

## 📌 Future Improvements
- Add more aggressive data augmentation to reduce overfitting
- Experiment with other architectures like **EfficientNet**, **InceptionV3**, **DenseNet**
- Build the UI using **OpenCV**
- Deploy the model as a **web application**
---
