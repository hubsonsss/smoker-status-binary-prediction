# Binary Prediction of Smoker Status Using Bio-Signals
# Authors:
- Hubert Sobociński
- Michał Matuszyk
- Martyna Leśniak
## Overview

This repository contains the implementation and evaluation of a machine learning model for predicting smoker status using bio-signals. 🚬 Smoking is a major public health concern, contributing to numerous diseases and significantly reducing life expectancy. Traditional identification methods often rely on self-reported data, which may be unreliable.

This study leverages physiological and biochemical indicators, including vital signs and blood test results, to train machine learning models. 🏥 The best-performing model, a Random Forest classifier, demonstrated high accuracy and robustness, providing a reliable, non-invasive method for detecting smoking status. This approach has potential applications in public health campaigns and early disease detection.

## Key Components

### 1. Data Preparation

📊 Exploratory Data Analysis (EDA):

🔍 Analysis of dataset structure and distributions.

🚨 Identification of missing values and outliers.

📈 Visualization of key features via histograms and boxplots.

🔗 Correlation analysis between features and smoking status.

#### 🛠 Data Preprocessing:

🗑️ Removal of duplicates.

🚀 Handling of outliers through statistical capping.

🔢 Log transformations for normalizing skewed features.

🏋️‍♂️ Feature engineering: BMI calculation, composite hearing and eyesight features, AST/ALT ratio.

✂️ Feature selection to remove redundant variables.

📏 Normalization using Min-Max scaling.

### 2. Machine Learning Models

#### 🤖 Preliminary Modeling:

🧩 The dataset was split into training (80%) and testing (20%) sets.

🏆 Initial evaluation of multiple models:

🌳 Decision Tree

📏 Support Vector Machine (SVM)

📊 Logistic Regression

🤝 K-Nearest Neighbors (KNN)

🎲 Naïve Bayes

🧠 Neural Networks

🌲 Random Forest

📊 Performance metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

🔝 Random Forest and Neural Network models demonstrated the highest predictive capabilities.

### 3. Model Optimization & Enhancement

#### ⚙️ Neural Network Optimization:

🏗️ Experimentation with different architectures, including varying the number of hidden layers and neurons.

🎛️ Testing activation functions (ReLU, Sigmoid) and different epoch settings.

🚨 Results showed high ROC-AUC but a tendency to under-classify smokers.

#### 🌲 Random Forest Optimization:

🎯 Hyperparameter tuning using Grid Search.

🎚️ Best parameters found:

bootstrap=False

max_depth=None

max_features='sqrt'

min_samples_leaf=2

min_samples_split=5

n_estimators=300

✅ Achieved high recall for smoker classification (~80%).

📌 Feature importance analysis identified hemoglobin, height, and GTP as key predictive factors.

### 4. Competition Performance

🏅 The model was evaluated in a Kaggle-style competition setting.

📊 Performance scores ranged between 0.8063 and 0.8615, with top models achieving results within 1-2% of the competition’s leading scores.

🖥️ Despite hardware limitations, the model demonstrated high competitiveness and potential for further fine-tuning.

Results and Insights

✅ Bio-signals offer an effective and non-invasive way to predict smoking status.

🌲 Random Forest outperformed other models in recall and overall robustness.

📢 The model has practical applications in public health, aiding smoking cessation efforts and disease prevention.

🩸 Key physiological markers such as hemoglobin levels and triglycerides were strong indicators of smoking status.

