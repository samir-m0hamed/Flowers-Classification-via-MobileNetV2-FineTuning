# 🌸 Flowers Classification: Transfer Learning & Fine-Tuning

This repository contains a complete Deep Learning pipeline for classifying flower species with high precision. The project demonstrates the power of **Transfer Learning** and **Fine-Tuning** using the **MobileNetV2** architecture to build a robust classifier even with limited data.

## 📂 Dataset Overview
The model is trained on a dataset containing 5 classes:
* **Daisy**
* **Roses**
* **Sunflowers**
* **Tulips**
* **Dandelion**

## 🛠️ Technical Highlights
* **Data Pipeline:** Implementation of `tf.data.Dataset` for efficient image loading, decoding, and prefetching, ensuring zero bottlenecks during training.
* **Exploratory Data Analysis (EDA):** Interactive visualization using `Plotly` to analyze class distribution and data balance.
* **Model Architecture:** * **Feature Extractor:** MobileNetV2 (Pre-trained on ImageNet).
    * **Custom Head:** Global Average Pooling + Dense Output layer.
* **Training Strategy:**
    * **Phase 1:** Training the classification head while freezing the base model.
    * **Phase 2 (Fine-Tuning):** Unfreezing top layers of MobileNetV2 with a very low learning rate to adapt to the specific features of flower images.

## 📊 Key Results & Performance
The project achieved significant performance improvements through the two-stage training process:

| Training Phase | Description | Key Outcome |
| :--- | :--- | :--- |
| **Initial Training** | Training only the top layers | Established a strong baseline accuracy. |
| **Fine-Tuning** | Unfreezing base layers | Refined the model to capture intricate floral patterns. |

### 📈 Metrics Visualization
* **Accuracy/Loss Curves:** The notebook includes clear plots showing the convergence of training and validation metrics, demonstrating a well-regularized model.
* **Visual Validation:** A grid of **32 sample predictions** is displayed at the end of the notebook, comparing `True Labels` vs `Predicted Labels` to demonstrate real-world reliability.

## 🚀 How to Run
1.  **Environment:** Open the `.ipynb` file in Google Colab or any Jupyter environment.
2.  **Dependencies:** Ensure you have `tensorflow`, `plotly`, `pandas`, and `matplotlib` installed.
3.  **Data:** The notebook is designed to work with images stored in Google Drive; update the `drive.mount` path if necessary.

## 🧬 Model Structure
```python
# Architecture Summary
- Input (224, 224, 3)
- MobileNetV2 (Base)
- GlobalAveragePooling2D
- Dense (5 units, Softmax)
