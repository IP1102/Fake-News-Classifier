#Necessary Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Getting the data
data = pd.read_csv('news.csv')

#Splitting the dataset
X_train,X_test,Y_train,Y_test=train_test_split(data.text, data.label, test_size=0.25)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.8)

tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

#SVM
classifier = SVC(kernel = 'linear')


#Fitting the model
classifier.fit(tfidf_train, Y_train)

#Predicting
Y_pred = classifier.predict(tfidf_test)
acc = accuracy_score(Y_test, Y_pred)
print(f'SVM Accuracy: {acc}')




