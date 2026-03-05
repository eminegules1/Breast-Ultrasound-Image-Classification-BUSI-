# Breast Ultrasound Image Classification (BUSI)

This repository implements a deep learning pipeline for the multi-class classification of breast ultrasound images. Using the **BUSI (Breast Ultrasound Images)** dataset, the system categorizes scans into three distinct classes: **Normal**, **Benign**, and **Malignant**.

This project aligns with my ongoing focus on high-precision medical imaging and AI-driven diagnostic tools.

---

## 🔬 Pipeline Overview

The project follows a structured machine learning workflow, from data exploration to deep feature extraction.

### 1. Data Loading & Exploration

**Structured Ingestion:** Images are read from a directory where folder names correspond to class labels.


**Data Mapping:** A centralized DataFrame is built containing all file paths and their respective labels.


**Distribution Analysis:** Class distributions are visualized using `seaborn` to identify potential data imbalances.



### 2. Preprocessing & Augmentation

**Spatial Normalization:** All images are resized to **256×256 RGB**.


**Pixel Scaling:** Values are rescaled to the $[0, 1]$ range.


**Data Splitting:** The dataset is partitioned into **80% Training**, **10% Validation**, and **10% Testing** sets.


**Batch Processing:** Utilizes `ImageDataGenerator` for efficient memory management during training.



### 3. Model Architecture

The system utilizes **Transfer Learning** to leverage pre-trained spatial features.

**Backbone:** **DenseNet-121** (Pre-trained on ImageNet), acting as a frozen feature extractor.


* **Classification Head:**
* Flatten Layer $\rightarrow$ Dense (1024).


* Dropout (0.5) for regularization.


* Dense (1024) $\rightarrow$ Dropout (0.3).


* Final Output: Dense (3 units) with **Softmax** activation.





### 4. Training Configuration

**Optimizer:** Adam (Learning Rate: 0.001).


**Loss Function:** Categorical Crossentropy.


**Callbacks:** `ModelCheckpoint` is implemented to save the best model based on validation accuracy.



---

## ⚠️ Critical Implementation Notes

To ensure the integrity of the diagnostic results, the following optimizations are integrated into the workflow:

* **The Mask Filter:** The BUSI dataset contains `_mask.png` files for segmentation. These are explicitly filtered out of the classification training set to prevent data corruption.



**DenseNet Preprocessing:** Inputs are processed to match the specific distribution requirements of the DenseNet backbone.

 
**Class Imbalance Handling:** Given that the dataset often contains more "Benign" samples than "Malignant," class weights are applied during `model.fit()`.



---

## 📈 Evaluation & Results

The model's performance is monitored through learning curves for both loss and accuracy. Final metrics are reported on the unseen test set to ensure an unbiased evaluation of the model's ability to generalize to new clinical scans.

---

