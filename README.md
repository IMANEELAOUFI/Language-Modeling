# Language-Modeling
# Overview
This project focuses on utilizing Natural Language Processing (NLP) techniques and models provided by the Sklearn library. The objective is to gain familiarity with language models for regression and classification tasks.

# Part 1: Language Modeling / Regression
## Dataset
The dataset used for this part can be found here.

## Preprocessing
- Tokenization, stemming, lemmatization, and removal of stop words are performed to preprocess the collected dataset.
- Discretization is applied to transform continuous data into categorical data.
  
## Encoding
Word2Vec (CBOW, Skip Gram), Bag of Words, and TF-IDF are used to encode the data vectors.

## Models
The following models are trained using the Word2Vec embeddings:

- Support Vector Regression (SVR)
- Linear Regression
- Decision Tree
  
## Evaluation
The models are evaluated using standard metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The best model is selected based on these metrics, and the choice is justified.

## Results Interpretation
The obtained results are interpreted to understand the effectiveness of the selected model and the overall performance of the language models for regression.



![image](https://github.com/IMANEELAOUFI/Language-Modeling/assets/118814232/8341f7d5-dc6c-4e74-b3c7-6e2f240217c0)


![Uploading image.png…]()




# Part 2: Language Modeling / Classification
## Dataset
The dataset used for this part can be found here.

## Preprocessing
Similar to Part 1, a preprocessing NLP pipeline is established for tokenization, stemming, lemmatization, stop words removal, and discretization.

## Encoding
Data vectors are encoded using Word2Vec (CBOW, Skip Gram), Bag of Words, and TF-IDF.

## Models
The following models are trained using the Word2Vec embeddings:

- Support Vector Machine (SVM)
- Naive Bayes
- Logistic Regression
- AdaBoost
  
## Evaluation
Models are evaluated using standard metrics such as Accuracy, Loss, and F1 Score, along with other metrics like BLEU Score. The best model is selected based on these metrics, and the choice is justified.

## Results Interpretation
The obtained results are interpreted to understand the effectiveness of the selected model and the overall performance of the language models for classification.

# Conclusion
In this project, we explored NLP techniques and models for language modeling using Sklearn. We performed preprocessing, encoding, training, and evaluation for both regression and classification tasks. Through this work, we gained valuable insights into the effectiveness of different models and techniques in NLP applications.
