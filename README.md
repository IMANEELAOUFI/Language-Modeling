# Language-Modeling
# Overview
This project is divided into two main parts: Language Modeling / Regression and Language Modeling / Classification. Each part uses a specific dataset and involves establishing an NLP preprocessing pipeline, encoding data vectors using various techniques, training models, and evaluating their performance using standard metrics. The goal is to identify the best-performing model for each task.

# Part 1: Language Modeling / Regression
## Dataset
The dataset used for this task can be found here. It contains short answers along with their corresponding scores.

## Steps
## 1) NLP Preprocessing Pipeline

- Tokenization: Splitting the text into individual tokens or words.
- Stemming and Lemmatization: Reducing words to their root forms.
- Removal of Stop Words: Eliminating common words that do not contribute much to the meaning.
- Discretization: Converting continuous features into discrete bins (if applicable).
  
## 2) Data Encoding

Word2Vec: Using both Continuous Bag of Words (CBOW) and Skip Gram models to convert text into numerical vectors.
Bag of Words (BoW): Creating a matrix of token counts.
TF-IDF (Term Frequency-Inverse Document Frequency): Reflecting the importance of words in the documents.
## 3) Model Training

Support Vector Regression (SVR)
Naive Bayes
Linear Regression
Decision Tree Regression
## 4) Model Evaluation

Mean Squared Error (MSE): Measures the average of the squares of the errors.
Root Mean Squared Error (RMSE): The square root of MSE, giving the error in the same units as the response variable.
R-squared (R²): Represents the proportion of the variance for the dependent variable that's explained by the independent variables.
## 5) Result Interpretation

Select the best model based on the evaluation metrics.
Provide a detailed explanation of why the chosen model performs the best.

# Part 2: Language Modeling / Classification
## Dataset
The dataset used for this task is available on Kaggle: Twitter Entity Sentiment Analysis. It contains tweets labeled with sentiments.

## Steps
## 1) NLP Preprocessing Pipeline

Tokenization: Splitting the text into individual tokens or words.
Stemming and Lemmatization: Reducing words to their root forms.
Removal of Stop Words: Eliminating common words that do not contribute much to the meaning.
Discretization: Converting continuous features into discrete bins (if applicable).
## 2) Data Encoding

Word2Vec: Using both Continuous Bag of Words (CBOW) and Skip Gram models to convert text into numerical vectors.
Bag of Words (BoW): Creating a matrix of token counts.
TF-IDF (Term Frequency-Inverse Document Frequency): Reflecting the importance of words in the documents.
## 3) Model Training

Support Vector Machine (SVM)
Naive Bayes
Logistic Regression
AdaBoost
## 4) Model Evaluation

Accuracy: The ratio of correctly predicted instances to the total instances.
Loss: Measures the error of the model.
F1 Score: The harmonic mean of precision and recall.
Other Metrics (e.g., BLEU Score): Evaluate the quality of text-based predictions (if applicable).
## 5) Result Interpretation

Select the best model based on the evaluation metrics.
Provide a detailed explanation of why the chosen model performs the best.
## Conclusion
In this project, we implemented comprehensive NLP pipelines for both regression and classification tasks using various text encoding techniques and machine learning algorithms. Through rigorous evaluation using standard metrics, we identified the best-performing models and provided insights into their performance. This project demonstrates the application of NLP and machine learning techniques to real-world text data, offering valuable lessons in model selection and evaluation.
