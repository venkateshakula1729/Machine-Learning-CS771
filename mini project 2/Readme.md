# Continual Learning with Prototypes on CIFAR-10

This project demonstrates a continual learning framework using a prototype-based model applied to the CIFAR-10 dataset. It involves feature extraction, continual learning on datasets with similar input distributions, and adaptation to datasets with diverse distributions.

## Problem Statement

Traditional machine learning models suffer from **catastrophic forgetting** when learning new tasks sequentially. This project tackles the challenge of training a single, fixed-capacity model on a stream of 20 CIFAR-10 subsets without losing performance on previously learned datasets.

### Dataset Structure
- **D1-D10**: Same input distribution p(x)
- **D11-D20**: Different input distributions with domain shifts
- **Constraint**: Only D1 is labeled; remaining datasets are unlabeled
- **Evaluation**: 20 held-out labeled datasets for accuracy assessment
  
## Folder Structure

The folder contains the following files:

- [**`FeatureExtractor.ipynb`**](./FeatureExtractor.ipynb): Extracts features from the CIFAR-10 dataset.
- [**`task1.ipynb`**](./task1.ipynb): Implements continual learning on datasets D1 to D10 with homogeneous input distributions.
- [**`task2.ipynb`**](./task2.ipynb): Uses the saved model from `task1` to continue training on datasets D11 to D20 with heterogeneous input distributions.

---

## How to Run

### Step 1: Extract Features

1. Run [`FeatureExtractor.ipynb`](./FeatureExtractor.ipynb) to extract features from the CIFAR-10 dataset.

**Note**: Feature extraction may take significant time. If you prefer, you can download pre-extracted features from [**Drive**](https://drive.google.com/drive/folders/1Ya0vAoNCkkTJkZj1g6aGCP6T2VnHo48Z?usp=sharing)

### Step 2: Train on Homogeneous Datasets

1. Run [`task1.ipynb`](./task1.ipynb) to train the prototype-based model on datasets D1 to D10.
2. After completion, the model trained on the 10th dataset will be saved for use in the next step.

### Step 3: Adapt to Heterogeneous Datasets

1. Run [`task2.ipynb`](./task2.ipynb) to load the saved model from Step 2.
2. Continue training on datasets D11 to D20, adapting the model to diverse input distributions.

---

## Requirements

- Python 3.x
- Jupyter Notebook
- Required libraries are specified in the notebooks.

---
