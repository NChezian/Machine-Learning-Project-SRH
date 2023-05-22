# Machine-Learning-Project-SRH

Section 1
In this section, the code utilizes the pandas and zipfile libraries to handle the input dataset. 
The dataset, obtained from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection, is a ZIP file containing the SMS spam collection. 
The code extracts the contents of the ZIP file using the ZipFile class and places it into a pandas dataframe. 
The labels in the dataset are also converted into binary values, where "Ham" is represented as 0 and "Spam" as 1.

Section 2
In this section, the code uses the sklearn library to perform various tasks. 
Firstly, the dataset is split into training and testing datasets using the train_test_split() function. 
The messages in the dataset are further vectorized using the TfidfVectorizer() function, 
which converts text into numerical feature vectors based on the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm. 
The training and testing data are transformed into TF-IDF feature vectors using this vectorizer.

Section 3
In this section, the code trains a Multinomial Naive Bayes classifier using the GridSearchCV approach. 
The GridSearchCV class from sklearn is utilized to search for the optimal hyperparameters for the classifier. 
The hyperparameters are defined as a dictionary (params) containing different values for the alpha parameter. 
Grid search is performed with 5-fold cross-validation to evaluate the performance of the classifier with different hyperparameter values.

Section 4
In this section, the code evaluates the predicted labels using various metrics. 
The predicted labels are generated using the trained classifier on the test data. 
The accuracy of the classifier is computed using the accuracy_score() function. 
Additionally, the code calculates and displays the confusion matrix, recall matrix, 
and F1 score using the confusion_matrix() and classification_report() functions from sklearn.metrics. 
These metrics provide insights into the performance of the classifier on the test data.

The code presented above demonstrates a pipeline for SMS spam classification using the Naive Bayes algorithm and TF-IDF vectorization.
It involves data preprocessing, feature extraction, model training, and evaluation.
