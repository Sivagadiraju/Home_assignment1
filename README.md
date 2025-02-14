# TensorFlow Tasks

## Author
**[Siva sathya vara prasad raju Gadiraju]**  
Student ID: [700756448]  
Course: [Neural Network and Deep Learning]  

---

## Project Overview
This repository contains implementations of various TensorFlow tasks covering tensor manipulations, loss functions, model training with different optimizers, and TensorBoard logging. These exercises help in understanding core TensorFlow operations and deep learning concepts.

## Table of Contents
1. [Tensor Manipulations & Reshaping](#tensor-manipulations--reshaping)
2. [Loss Functions & Hyperparameter Tuning](#loss-functions--hyperparameter-tuning)
3. [Train a Model with Different Optimizers](#train-a-model-with-different-optimizers)
4. [Train a Neural Network and Log to TensorBoard](#train-a-neural-network-and-log-to-tensorboard)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results & Analysis](#results--analysis)
8. [References](#references)

---

## 1. Tensor Manipulations & Reshaping
### Task Description
- Create a random tensor of shape (4,6).
- Find its rank and shape using TensorFlow functions.
- Reshape it into (2,3,4) and transpose it to (3,2,4).
- Broadcast a smaller tensor (1,4) to match the larger tensor and perform addition.
- Explain broadcasting in TensorFlow.

### Expected Output
- Print rank and shape of the tensor before and after reshaping/transposing.

### Broadcasting Explanation
Broadcasting allows TensorFlow to automatically expand the dimensions of smaller tensors to match larger tensors in element-wise operations. If the dimensions are compatible, TensorFlow implicitly replicates the smaller tensor along the missing dimensions.

---

## 2. Loss Functions & Hyperparameter Tuning
### Task Description
- Define true values (`y_true`) and model predictions (`y_pred`).
- Compute Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses.
- Modify predictions slightly and check how loss values change.
- Plot loss function values using Matplotlib.

### Expected Output
- Loss values printed for different predictions.
- Bar chart comparing MSE and Cross-Entropy Loss.

---

## 3. Train a Model with Different Optimizers
### Task Description
- Load the MNIST dataset.
- Train two models: one with Adam optimizer and another with SGD optimizer.
- Compare training and validation accuracy trends.

### Expected Output
- Accuracy plots comparing Adam vs. SGD performance.

---

## 4. Train a Neural Network and Log to TensorBoard
### Task Description
- Load the MNIST dataset and preprocess it.
- Train a simple neural network model and enable TensorBoard logging.
- Launch TensorBoard and analyze loss and accuracy trends.

### Expected Output
- Visualize loss and accuracy curves on TensorBoard.

---

## Installation
To run this project, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib
```

---

## Usage
Run the scripts for each task as follows:
```bash
python tensor_manipulations.py
python loss_functions.py
python train_optimizers.py
python tensorboard_logging.py
```
To start TensorBoard:
```bash
tensorboard --logdir=logs/
```
Then, open your browser and navigate to:
```
http://localhost:6006/
```

---

## Results & Analysis
- Tensor manipulations demonstrate TensorFlow’s reshaping and broadcasting features.
- Loss function comparison helps in understanding error metrics.
- Adam optimizer shows faster convergence than SGD in training MNIST models.
- TensorBoard provides real-time visualization of training performance.

---

## References
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Deep Learning with Python by François Chollet

---

