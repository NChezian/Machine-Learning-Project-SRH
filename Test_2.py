#               SMS Spam Classification using Naive Bayes and TF-IDF Vectorization

# Section 1 - Using the Pandas & Zipfile Libraries, the input dataset from  https://archive.ics.uci.edu/ml/datasets/sms+spam+collection 
#             is taken in and placed into a pandas dataframe and the labels are converted into binary values.

import pandas as pd
import zipfile

# Extracts the contents of the "smsspamcollection.zip" file to the current directory.

with zipfile.ZipFile("C:/Users/nchez/OneDrive/Desktop/ML-2/smsspamcollection.zip", 'r') as zip_file:
    zip_file.extractall()

# The SMS spam collection Dataset is loaded into a Pandas dataframe.

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Label to binary value conversion (Ham = 0, Spam = 1)

df["label"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

# Section 2 - Using the Sklearn library, various functions are imported, firsty the data is split into training data and tetsing dataset,
#             by using the train_test_split() function. Then the data is further vectorized using the TfidfVectorizer() function.
#             Further on, to perform grid search by tunig 'Alpha' hyperparameter.

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Splitting of dataset into training data and test data.
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# TfidfVectorizer is used to vectorize data, Order is set to the default value 1 considering unigrams.
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Defining Hyperparamters for the grid search and evaluatng performance.
params = {"alpha": [0.1, 0.5, 1, 2]}

# Section 3 -  Multinomial Naive Bayes classifier is trained using the GridSearchCV approach to find the optimal hyperparameters for the classifier. 

from sklearn.naive_bayes import MultinomialNB

# Training the Multinomial Naive Bayes classifier using GridSearchCV to determine optimum hyperparameter
clf = GridSearchCV(MultinomialNB(), params, cv=5)
clf.fit(X_train_tfidf, y_train)

# Section 4 - The predicted labels are evaluated using various metrics, including accuracy, confusion matrix, and classification report. 
#             The results provide insights into the performance of the classifier on the test data.

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predicting the labels of the test data using the best classifier
y_pred = clf.predict(X_test_tfidf)

# Computing the accuracy of the said classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing the confusion matrix, recall matrix, and F1 score
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test,y_pred))