# Fine-Tuning a Model for Multi-Class Image Classification

This project demonstrates the process of fine-tuning a pre-trained deep learning model (ResNet50) for multi-class image classification. The dataset is split into training, validation, and test sets, and the model is evaluated using metrics such as accuracy, precision, and AUC.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Project Overview
The goal of this project is to fine-tune a ResNet50 model pre-trained on ImageNet for a custom image classification task with 5 classes. Key features include:
- Data augmentation for robust training.
- Use of advanced metrics: Accuracy, Precision, Recall, F1-Score, and AUC.
- Evaluation on a well-defined test set.

---

## Dataset
The dataset used for this project is a custom dataset consisting of 5 classes of images. It is structured as follows:


The dataset is preprocessed using TensorFlow's `image_dataset_from_directory` utility to generate training and test batches.

---

## Model Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet.
- **Custom Layers**:
  - Global Average Pooling
  - Dense Layer (512 units, ReLU activation)
  - Dropout Layer (50% dropout rate)
  - Output Layer (Softmax activation for multi-class classification)

---

## Training and Evaluation
1. **Training**:
    - Optimizer: Adam (Learning rate = 1e-5)
    - Loss: Categorical Crossentropy
    - Metrics: Accuracy

2. **Evaluation**:
    - Metrics: Accuracy, Precision, AUC
    - Additional insights: Classification Report

---

## Results
- **Accuracy**: 83.4% (Test Dataset, Batch Size = 16)
- **Precision**: 82.9%
- **AUC**: 0.81 (One-vs-All)

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `dataset/` directory following the structure mentioned above.

---

## Usage
1. Run the script to fine-tune the model:
    ```bash
    python train_model.py
    ```

2. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```

3. View the results and metrics in the terminal output.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments
- The ResNet50 model used in this project is pre-trained on ImageNet.
- TensorFlow and Keras were used for deep learning model development.
- Dataset sourced from [mention your source if applicable].



