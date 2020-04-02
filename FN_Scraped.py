#Necessary Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import bs4 as bs
import pickle
import requests


#Getting the data
data = pd.read_csv('news.csv')
# data_real = pd.read_csv('realnews.csv')
#Splitting the dataset
X_train,X_test,Y_train,Y_test=train_test_split(data.text, data.label, test_size=0.25)
# X_test_real = data_real['News']

resp = requests.get('https://www.nytimes.com/2020/04/02/world/coronavirus-news.html')
soup = bs.BeautifulSoup(resp.text)
news_text = soup.find('header', {'class': 'css-13s9jzp edomiq21' }).text
# news_text.find('p', {'class':'css-15hwz5e evys1bk0'}).text

data_real = pd.DataFrame({'news':[news_text]})
data_real.to_csv()

X_test_real = data_real['news']


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.8)

tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test_real)

#SVM
classifier = SVC(kernel = 'linear')


#Fitting the model
classifier.fit(tfidf_train, Y_train)

#Predicting
Y_pred = classifier.predict(tfidf_test)
acc = accuracy_score(Y_test, Y_pred)
print(f'SVM Accuracy: {acc}')




