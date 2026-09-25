# ANN Classification for Customer Churn

An end-to-end **Machine Learning classification project** that predicts customer churn using an **Artificial Neural Network (ANN)**. The project covers data preprocessing, categorical encoding, feature scaling, neural network training, evaluation, and inference using a reproducible Python pipeline.

## Overview

Customer churn prediction is a binary classification problem aimed at identifying customers who are likely to discontinue a service.

This project implements a supervised learning pipeline that transforms customer data into model-ready features and uses a trained ANN to estimate churn probability.

### ML Pipeline

```text
Raw Customer Data
       ↓
Data Preprocessing
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
ANN Training
       ↓
Model Evaluation
       ↓
Churn Prediction
```

## Key Capabilities

* Binary customer churn classification using an **Artificial Neural Network**
* Automated preprocessing with reusable encoders and scalers
* Numerical feature normalization for neural network training
* Model persistence for inference without retraining
* Training and prediction logging
* Evaluation using standard classification metrics
* Reproducible Python-based training and inference workflow

## Technical Stack

| Category          | Technologies          |
| ----------------- | --------------------- |
| Language          | Python                |
| Deep Learning     | TensorFlow · Keras    |
| Data Processing   | Pandas · NumPy        |
| Machine Learning  | Scikit-learn          |
| Visualization     | Matplotlib · Seaborn  |
| Model Persistence | H5 · Pickle           |
| Task              | Binary Classification |

## Model Architecture

The prediction model is an **Artificial Neural Network (ANN)** designed for binary classification.

The pipeline consists of:

* Input layer receiving the preprocessed customer features
* Fully connected neural network layers
* Non-linear activation functions for feature learning
* Output layer producing the churn prediction
* Binary classification objective for churn detection

The trained model is stored in **H5 format** and can be reused for inference without repeating the training process.

## Data Preprocessing

Before training, the customer data is transformed into a numerical representation suitable for the neural network.

The preprocessing pipeline includes:

* Handling categorical variables using encoders
* Feature scaling using reusable scalers
* Numerical feature transformation
* Consistent preprocessing between training and inference

The fitted preprocessing objects are persisted so that new customer records are transformed using the **same feature mapping and scaling parameters** used during training.

## Evaluation

The model can be evaluated using standard classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

These metrics provide a more complete view of model performance than accuracy alone, particularly when analyzing churn classification errors.

## Project Structure

```text
ANN-Churn-Classification/
│
├── data/                 # Dataset and processed data
├── models/               # Trained ANN model
├── encoders/             # Saved preprocessing objects
├── notebooks/            # Exploratory analysis and experiments
├── src/                  # Training and inference scripts
├── logs/                 # Training and prediction logs
├── requirements.txt      # Python dependencies
├── README.md
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/erij2000/ANN-Churn-Classification.git
cd ANN-Churn-Classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

Run the training pipeline to preprocess the data, train the ANN, evaluate the model, and save the resulting artifacts.

```bash
python train.py
```

### Run inference

Use the trained model and saved preprocessing artifacts to generate churn predictions for new customer records.

```bash
python predict.py
```

> Update the commands above if the actual training and prediction scripts use different filenames or directories.

## Reproducibility

The project separates **model artifacts** from preprocessing components so that the same transformation pipeline can be applied consistently during inference.

Saved artifacts include:

* Trained ANN model
* Feature encoders
* Feature scaler
* Training/prediction logs

## License

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

See [`LICENSE`](LICENSE) for the complete license terms.
