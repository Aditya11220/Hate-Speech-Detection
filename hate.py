import pandas as pd
import numpy as np
dataset = pd.read_csv("tweets.csv")
dataset
dataset.isnull()
dataset.info()
dataset.describe()
dataset["labels"] = dataset["class"].map({0 : "Hatespeech",
                                         1 : "Offensive language",
                                         2 : "no hate or offensive language"})
dataset
data = dataset[["tweet", "labels"]]
data
import re
import nltk
import string
#import stop words 
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
#import stemming
stemmer = nltk.SnowballStemmer("english")
# Data cleaning
def clean_data(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"\[*?\]", '', text)
    text = re.sub(r"<*?>+", '', text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub(r"\n", '', text)
    text = re.sub(r"\w*\d\w*", '', text)
    
    #stopword removal
    text = [word for word in text.split(" ") if word not in stopwords]
    text = " ".join(text)
    #SDtemming the text
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)
    return text
    data.loc[:, "tweet"] = data["tweet"].apply(clean_data)
data
X = np.array(data["tweet"])
Y = np.array(data["labels"])
data
X
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
X = cv.fit_transform(X)
X
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# Building our ml model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

Y_pred = dt.predict(X_test)
#conclusion matx=rix and accuracy
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test, Y_pred)
cn
import seaborn as sns
import matplotlib.pyplot as ply
%matplotlib inline
sns.heatmap(cn, annot=True, fmt="f", cmap="YlGnBu")
from sklearn.metrics
 import accuracy_score
accuracy_score(Y_test, Y_pred)
sample = " kiss you"
sample = clean_data(sample)
sample
data1 = cv.transform([sample]).toarray()
data1
dt.predict(data1)
print("Accuracy Score:", accuracy_score(Y_test, Y_pred))
