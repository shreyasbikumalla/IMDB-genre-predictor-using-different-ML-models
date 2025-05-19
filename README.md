# 🎬 IMDb Movie Genre Classification

This project implements a comprehensive genre classification system using IMDb movie titles and descriptions. It explores both traditional machine learning and deep learning approaches to tackle a 27-class classification problem with significant class imbalance and noisy text data.

## 🧠 Overview

The classification pipeline includes:
- Text preprocessing: Cleaning, lowercasing, stopword removal, lemmatization
- TF-IDF vectorization (5000 and 2000 feature variants)
- Dimensionality reduction: PCA and LDA
- Model evaluation via accuracy, cross-validation, and confusion matrices

## 🧪 Models Used

### Traditional ML Models:
- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost
- Custom Bayesian Classifier

### Deep Learning Models:
- Convolutional Neural Network (CNN)
- Long Short-Term Memory Network (LSTM)

## 📁 Files in the Repository

```
.
├── final code imdb.ipynb         # Complete training and evaluation notebook
├── 802 project _final report_shreyas.pdf  # Final project report
├── README.md                     # This file
```

## 📊 Dataset

- Source: [Kaggle IMDb Genre Classification](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- 35,000 training samples; 7,000 test samples
- 27 genres (Drama, Comedy, Action, Documentary, etc.)
- Single-label classification setup

## 📈 Evaluation Metrics

- Accuracy on test set
- Cross-validation error and variance
- Confusion matrices for class-wise performance

## 🔧 Requirements

Install the following packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow keras nltk
```

Ensure NLTK resources are downloaded:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## ▶️ Running the Code

1. Open `final code imdb.ipynb` in Jupyter or Google Colab.
2. Run all cells sequentially to train models, apply PCA/LDA, and visualize results.
3. Refer to the final report for detailed performance insights and analysis.

## 🔮 Future Work

- Extend to multi-label classification
- Integrate transformer-based models (e.g., BERT)
- Apply data augmentation and class balancing
- Explore ensemble techniques and co-occurrence modeling
