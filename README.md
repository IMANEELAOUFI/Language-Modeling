# Language-Modeling
# Overview
This project focuses on utilizing Natural Language Processing (NLP) techniques and models provided by the Sklearn library. The objective is to gain familiarity with language models for regression and classification tasks.

# Part 1: Language Modeling / Regression
## Dataset
The dataset used for this part can be found here:https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv.

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



![image](https://github.com/IMANEELAOUFI/Language-Modeling/assets/118814232/ab325146-1855-4be3-8164-27a6d8adc1b4)
![image](https://github.com/IMANEELAOUFI/Language-Modeling/assets/118814232/8a18d70a-49bf-49d0-bc37-d11e562bbebf)
![image](https://github.com/IMANEELAOUFI/Language-Modeling/assets/118814232/84811944-5842-4a39-a31f-bfeaf68d29bb)









# Part 2: Language Modeling / Classification
## Dataset
The dataset used for this part can be found here: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

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

![image](https://github.com/IMANEELAOUFI/Language-Modeling/assets/118814232/27ac96ae-1035-41a2-9365-1a7a911a7dcd)
![image](https://github.com/IMANEELAOUFI/Language-Modeling/assets/118814232/a5cb849e-49d9-4ec2-9367-99f63ca51d97)




# Conclusion
In this project, we explored NLP techniques and models for language modeling using Sklearn. We performed preprocessing, encoding, training, and evaluation for both regression and classification tasks. Through this work, we gained valuable insights into the effectiveness of different models and techniques in NLP applications.
