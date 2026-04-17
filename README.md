# ai-ml-project-spam-not-spam-classifier


Project Overview

This project is a simple implementation of spam detection using basic machine learning techniques. The main goal is to classify messages into two categories: Spam and Not Spam (Ham).

This project is designed for beginners to understand how machine learning can be applied to real-world problems like filtering unwanted messages.

Objective

To understand the basics of text classification
To learn how to convert text data into numerical form
To build a simple machine learning model for spam detection
Technologies Used

Python
Scikit-learn (sklearn)
Basic text processing techniques
Dataset

The dataset used in this project contains:

A list of messages (text)
Labels for each message (Spam or Ham)
Example:

"Win a free ticket now" → Spam
"Let's meet tomorrow" → Ham
Project Structure

Spam-Detection-Project/ │ ├── README.md ├── Spam_Detection_Project_Report.pdf ├── spam_detection.py └── dataset.csv

Working Process

1. Data Preprocessing
* Convert all text to lowercase
* Remove unnecessary symbols or characters
  
2. Feature Extraction
* Used Bag of Words technique
* Converted text into numerical format using CountVectorizer
  
3. Model Used
* Multinomial Naive Bayes classifier
  
4. Training the Model
* The dataset is split into training and testing data
* The model learns patterns from training data
  
5. Prediction
* The model predicts whether a message is spam or not
  
Basic Code Structure

from sklearn.feature_extraction.text import CountVectorizer from sklearn.naive_bayes import MultinomialNB

Sample data
messages = ["Free offer now", "Call me later", "Win money"] labels = ["spam", "ham", "spam"]

Convert text to numbers
cv = CountVectorizer() X = cv.fit_transform(messages)

Train model
model = MultinomialNB() model.fit(X, labels)

Test
test = cv.transform(["Free money"]) print(model.predict(test))

Result

The model is able to classify messages with reasonable accuracy. Since this is a basic implementation with a small dataset, the results are simple but effective for understanding the concept.
