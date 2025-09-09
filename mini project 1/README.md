# CS771: Mini-Project 1 - Binary Classification

## Group Information
**Group Number:** 72  
**Members:**
- Venkatesh (220109)
- Manikanta (220409)
- Prashant (220803)
- Sai Nikhil (221095)
- Pankaj Nath (221188)

## Project Overview
This mini-project focuses on developing and evaluating binary classification models for three distinct datasets derived from an original raw dataset. Each dataset represents the same underlying data but with different feature representations. The goal is to identify the best-performing model based on validation accuracy while considering the amount of training data used.

## Project Objectives
1. **Task 1:** Develop individual binary classification models for each dataset, exploring different model architectures and training strategies.
2. **Task 2:** Create a unified model by combining the datasets to leverage complementary information from different feature representations.

## Datasets
The three datasets represent the same underlying data in different formats, each with unique characteristics:

### 1. Emoticons as Features Dataset
- **Description:** Contains 13 categorical features where each input is represented by emoticons.
- **Preprocessing:** 
  - One-hot encoding applied to a sparse matrix.
  - Tokenization using TensorFlow tokenizer.

### 2. Deep Features Dataset
- **Description:** Each input is represented as a 13x786 matrix of embeddings.
- **Preprocessing:** Flattening of the matrix to maintain compatibility with models.

### 3. Text Sequence Dataset
- **Description:** Features are represented as strings of 50 digits, allowing for sequential representation.
- **Preprocessing:** Numerical conversion and dynamic padding implemented for variable-length inputs.

## Model Development
### Emoticons Dataset
- **Data Processing Strategies:**
  1. **Strategy I:** One-hot encoding of emoticon sequences.
  2. **Strategy II:** Tokenization with unique numerical values.

- **Models Trained:**
  - Custom Neural Network with early stopping and hyperparameter tuning.
  - Traditional Models: Logistic Regression, Support Vector Machines (SVM), Random Forest, XGBoost, K-Nearest Neighbors (KNN).

### Deep Features Dataset
- **Models Used:**
  - XGBoost with extensive hyperparameter tuning.
  - Random Forest with 200 estimators.
  - Logistic Regression as a baseline.
  - SVM with RBF kernel for high-dimensional data.

### Text Sequence Dataset
- **Model Architecture:**
  - Hybrid model with a CNN feature extractor and traditional classifiers (XGBoost and Logistic Regression).
  - Focus on parameter efficiency with a total trainable parameter count under 10,000.

## Results and Evaluation
- Each model's performance was assessed based on validation accuracy across varying training sizes (20%, 40%, 60%, 80%, 100%).
- **Key Findings:**
  - Neural networks outperformed traditional models in the Emoticons Dataset.
  - XGBoost demonstrated superior performance on the Deep Features Dataset.
  - Logistic Regression and XGBoost performed well on the Text Sequence Dataset.

### Performance Summary
| Model                     | 20%   | 40%   | 60%   | 80%   | 100%  |
|---------------------------|-------|-------|-------|-------|-------|
| Logistic Regression        | 0.7198 | 0.8200 | 0.8650 | 0.9162 | 0.9243 |
| Random Forest              | 0.6033 | 0.7301 | 0.7464 | 0.8384 | 0.8589 |
| Custom Neural Network      | 0.9141 | 0.9284 | 0.9427 | 0.9714 | 0.9734 |
| XGBoost                   | 0.9489 | 0.9734 | 0.9816 | 0.9836 | 0.9877 |

## Conclusion
This project successfully implemented and evaluated multiple binary classification models across various datasets. The findings highlight the importance of model architecture and preprocessing strategies in achieving high validation accuracy.

## Installation and Usage
To replicate the results or explore further:
1. Ensure you have the required libraries installed (e.g., TensorFlow, Scikit-learn, etc.).
2. Follow the instructions in the accompanying scripts to preprocess the data and train the models.

