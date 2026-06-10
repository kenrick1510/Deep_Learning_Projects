# Deep Learning Projects: Image Classification and Temperature Forecasting

This repository contains two deep learning projects completed in Jupyter Notebook format:

1. **Project 1:** Image classification using a custom CNN and MobileNetV3Small transfer learning.
2. **Project 2:** Monthly temperature forecasting using GRU models and synthetic temperature generation using a Variational Autoencoder (VAE).

The projects demonstrate model building, training, evaluation, visualization, and performance comparison using TensorFlow/Keras.

---

## Repository Files

```text
.
├── Lim_Kenrick_proj1.ipynb      # Image classification: CNN vs MobileNetV3Small
├── Lim_Kenrick_proj2.ipynb      # Temperature forecasting: GRU models and VAE
├── train-2025.pkl               # Training image dataset, required for Project 1
├── valid-2025.pkl               # Validation image dataset, required for Project 1
├── test-2025.pkl                # Test image dataset, required for Project 1
├── temperatures-2025.pkl        # Temperature sequence dataset, required for Project 2
└── README.md
```

> Note: The `.pkl` dataset files are required to run the notebooks but may not be included in the repository if they are too large or restricted by coursework rules.

---

## Project 1: Image Classification

### Overview

Project 1 compares two image classification approaches:

- A **custom Convolutional Neural Network (CNN)** built from scratch.
- A **MobileNetV3Small transfer learning model** using pretrained ImageNet features.

The goal is to classify images into multiple object or animal categories and compare the performance of a lightweight custom model against a pretrained transfer learning model.

### Main Steps

1. Load training, validation, and test data from pickle files.
2. Visualize random images and class labels.
3. Build a custom CNN model with convolutional blocks, batch normalization, max pooling, dropout, and global average pooling.
4. Tune the CNN using **Keras Tuner Bayesian Optimization**.
5. Build a MobileNetV3Small transfer learning model.
6. Train both models using early stopping and learning-rate reduction.
7. Evaluate both models using accuracy, classification reports, and confusion matrices.
8. Compare prediction behaviour using four visual groups:
   - Both models correct
   - Both models incorrect
   - CNN correct only
   - MobileNet correct only

### Models Used

#### Custom CNN

The CNN uses four convolutional blocks with increasing filters, batch normalization, ReLU activation, max pooling, dropout, global average pooling, and a dense classification head.

Key design choices:

- Lightweight architecture
- Less than 30,000 trainable parameters
- Global Average Pooling instead of Flatten to reduce parameter count
- Bayesian hyperparameter tuning with a maximum of 10 trials

#### MobileNetV3Small

MobileNetV3Small is used as a frozen feature extractor, followed by a custom classification head.

Key design choices:

- Transfer learning from ImageNet
- Input resizing to match MobileNet requirements
- Faster convergence than the custom CNN
- Stronger generalization due to pretrained visual features

### Project 1 Results

| Model | Test Accuracy | Notes |
|---|---:|---|
| Custom CNN | 58% | Lightweight model, meets baseline requirement, but struggles with visually similar classes |
| MobileNetV3Small | 89% | Much stronger performance due to pretrained feature extraction |

The CNN performed well on visually distinctive classes such as salamandra and jellyfish but struggled with similar-looking categories such as cougar, lion, Egyptian cat, and dromedary. MobileNetV3Small achieved much higher overall accuracy and stronger class-level F1-scores.

---

## Project 2: Temperature Forecasting and Generation

### Overview

Project 2 focuses on monthly temperature time-series modelling. It has two main parts:

1. **GRU forecasting models** for predicting future minimum and maximum temperatures.
2. **Variational Autoencoder (VAE)** for generating synthetic temperature sequences.

The dataset is loaded from `temperatures-2025.pkl` and split into training, validation, and test sequences.

---

## Part I: GRU Temperature Forecasting

### Forecasting Task

Each input sequence contains **72 months** of historical temperature data, and the model predicts the next **12 months**.

Each timestep contains two features:

- Mean minimum monthly temperature
- Mean maximum monthly temperature

### GRU Models Compared

#### Model 1: Direct Multi-Step GRU

This model predicts all 12 future months at once.

- Two stacked GRU layers
- 16 hidden units
- Dense output reshaped into a 12-month forecast
- Avoids recursive error accumulation because all future steps are predicted directly

**Test MAE:** approximately **1.2125 °C**

#### Model 2: One-Step Autoregressive GRU

This model predicts one month ahead, then feeds its prediction back into the model to forecast the next month.

- Two stacked GRU layers
- 16 hidden units
- Recursive rollout for 12-month forecasting
- Strong short-term predictions but may accumulate errors over longer horizons

**Test MAE:** approximately **1.1861 °C**

#### Model 3: Encoder-Decoder GRU

This model uses an encoder-decoder structure inspired by sequence-to-sequence learning.

- Encoder compresses the 72-month input into a context representation
- Decoder generates the 12-month forecast
- Teacher forcing is used during training
- Forecasts are smoother but less accurate than Models 1 and 2

**Test MAE:** approximately **1.3936 °C**

### GRU Result Summary

| Model | Approach | Test MAE |
|---|---|---:|
| Model 1 | Direct multi-step GRU | 1.2125 °C |
| Model 2 | One-step autoregressive GRU | 1.1861 °C |
| Model 3 | Encoder-decoder GRU | 1.3936 °C |

Model 2 achieved the lowest test MAE, while Model 1 provided stable direct multi-step forecasting. Model 3 captured seasonal structure but produced smoother predictions and had the highest error.

---

## Part II: Variational Autoencoder

### Overview

The VAE learns a latent representation of historical maximum temperature sequences and generates synthetic temperature patterns.

Only the maximum temperature feature is used for this part.

### VAE Components

- Custom `Sampling` layer using the reparameterization trick
- Encoder network that maps input sequences into latent mean and log variance
- KL divergence regularization to structure the latent space
- Decoder network that reconstructs or generates temperature sequences
- Normalization to improve training stability

### VAE Output

The generated synthetic sequences show smooth seasonal temperature cycles. Some generated sequences follow regular sinusoidal patterns, while others include small irregular variations, reflecting the VAE's learned latent structure and stochastic sampling behaviour.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle
- Tabulate
- Jupyter Notebook

---

## Installation

Install the required Python packages:

```bash
pip install tensorflow keras-tuner numpy pandas matplotlib seaborn scikit-learn tabulate
```

If running inside Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

---

## How to Run

### Project 1

1. Place the following files in the same folder as `Lim_Kenrick_proj1.ipynb`:

```text
train-2025.pkl
valid-2025.pkl
test-2025.pkl
```

2. Open the notebook:

```bash
jupyter notebook Lim_Kenrick_proj1.ipynb
```

3. Run the cells from top to bottom.

### Project 2

1. Place the following file in the same folder as `Lim_Kenrick_proj2.ipynb`:

```text
temperatures-2025.pkl
```

2. Open the notebook:

```bash
jupyter notebook Lim_Kenrick_proj2.ipynb
```

3. Run the cells from top to bottom.

---

## Key Learnings

- Transfer learning can significantly improve image classification performance compared with a small CNN trained from scratch.
- Global Average Pooling helps reduce the number of trainable parameters in CNN architectures.
- GRU models are effective for seasonal time-series forecasting.
- Autoregressive forecasting can perform well in the short term but may accumulate errors over longer horizons.
- Encoder-decoder models can capture sequence structure but may smooth predictions when the latent representation is too compressed.
- VAEs can generate realistic synthetic seasonal temperature sequences by learning a structured latent space.

---

## Author

**Kenrick Lim**  
Master of Data Science, The University of Western Australia
