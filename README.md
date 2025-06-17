# TwitNalyze

This project is focused on analyzing the **sentiment of tweets** using **Natural Language Processing (NLP)** techniques. It applies machine learning algorithms to classify tweets into different sentiment categories based on their content.

---

##  Project Overview

-  **Goal**: Predict the sentiment of a tweet (`Positive`, `Negative`, `Neutral`, etc.).
-  **Feature Engineering**: `TF-IDF` vectorization with n-grams
-  **Text Preprocessing**: Cleaning, stopword removal, optional stemming/lemmatization
-  **Evaluation**: Classification report including precision, recall, and F1-score
-  **Confusion Matrix**: Evaluating Confusion Matrix

---

##  Dataset

The dataset contains tweet data with the following key columns:

- `id`: Unique identifier  
- `entity`: Topic or target of the tweet  
- `tweet`: The actual tweet text  
- `sentiment`: Sentiment label (0, 1, 2, 3)

## Sentiment Label Mapping:

| Label | Meaning    |
|-------|------------|
| 0     | Irrelevant |
| 1     | Negative   |
| 2     | Neutral    |
| 3     | Positive   |

---

##  Text Preprocessing

Performed:

- Lowercasing  
- Removing punctuation and special characters  
- Removing stopwords  
- Optional: **Stemming** using `PorterStemmer` (tunable)  
- Generating a new column: `stemmed_content` for processed text

---

##  Feature Extraction

- Using `TfidfVectorizer` with custom parameters:
python
TfidfVectorizer(
    ngram_range=(1, 4),
    max_df=0.8,
    min_df=2,
    max_features=15000,
    sublinear_tf=True,
    stop_words='english',
    strip_accents='unicode',
    norm='l2'
)

---

##  Model Training
| Model Used               | Score (Accuracy) |
|--------------------------|------------------|
| Logistic Regression      | 0.8397           |
| Multinomial Naive Bayes  | 0.7618           |
| Random Forest Classifier | 0.9029           |
| SVC Classifier           | 0.835            |
| XG Boost Classifier      | 0.6556           |

---

## Requirements
Python 3.7+
Libraries:
- scikit-learn
- nltk
- pandas
- numpy
- xgboost

---
