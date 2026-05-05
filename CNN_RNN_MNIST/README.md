# CNN vs RNN — MNIST Digit Recognizer

A deep learning project that trains and compares two architectures — **CNN** and **RNN** — on the MNIST dataset. You pass a grayscale image of a handwritten digit, and the model predicts the number.

---

## Overview

| Model | Approach | Result |
|-------|----------|--------|
| CNN | Treats image as a 2D grid of pixels | ✅ Higher Accuracy |
| RNN | Treats image as a sequence of 28 rows | Decent, but lower |

Both models are trained on the same dataset, evaluated on the same test set, and their loss curves are compared side by side.

---

## Dataset

**MNIST** — 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels.

- Training set: 48,000 samples
- Validation set: 12,000 samples
- Test set: 10,000 samples

Preprocessing:
- Converted to tensor
- Normalized with mean `0.5` and std `0.5`

---

## Project Structure

```
CNN_RNN_MNIST/
│
├── CNN_RNN_MNIST.ipynb   # Main notebook (data prep, models, training, evaluation)
├── data/                 # MNIST dataset (auto-downloaded)
└── README.md
```

---

## Models

### CNN (Convolutional Neural Network)

```
Input (1×28×28)
  → Conv2d(1, 24) + ReLU + MaxPool
  → Conv2d(24, 48) + ReLU + MaxPool
  → Conv2d(48, 96) + ReLU + MaxPool
  → Flatten
  → Linear(864, 234) + ReLU
  → Linear(234, 10)
```

### RNN (Recurrent Neural Network)

```
Input reshaped to (batch, 28, 28)  — each row = one timestep
  → RNN(input=28, hidden=128, layers=2)
  → Last hidden state → Linear(128, 10)
```

---

## Training

- **Optimizer:** Adam (default lr)
- **Loss:** CrossEntropyLoss
- **Epochs:** 10
- **Batch size:** 64

---

## Results

Both models are evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Loss curves** (train vs validation, plotted per epoch)

CNN outperformed RNN clearly on this task — makes sense, since spatial features in images are better captured by convolutional layers.

---

## How to Run

1. Clone the repo and open the notebook:

```bash
git clone <your-repo-url>
cd CNN_RNN_MNIST
jupyter notebook CNN_RNN_MNIST.ipynb
```

2. Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas matplotlib
```

3. Run all cells — MNIST will auto-download on first run.

---

## Requirements

```
torch
torchvision
scikit-learn
pandas
matplotlib
numpy
```

---

## Author

**Deepanshu Tevathiya**
AI / ML Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/deepanshu-tevathiya)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/deepanshu-tevathiya)
