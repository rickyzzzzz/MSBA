
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


# 1)Read in the data into a DataFrame. 
def data2df(path, label):
    file, text = [], []
    for f in os.listdir(path):
        file.append(f)
        fhr = open(path+f, 'r', encoding='utf-8', errors='ignore') #encoding='utf-8', errors='ignore'.
        t = fhr.read()
        text.append(t)
        fhr.close()
    return(pd.DataFrame({'file': file, 'text':text, 'class':label}))

dfneg = data2df('HealthProNonPro/NonPro/', 0)
dfpos = data2df('HealthProNonPro/Pro/', 1)

df = pd.concat([dfpos, dfneg], axis =0)
df.sample(frac=0.005)


# In[3]:


#2)Setup the data for Training/Testing. Use 30% for testing.
X, y = df['text'], df['class']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=214)


# In[4]:


#3)Create a custom preprocessing function. 
# setup a custom preprocessor

import re
import string
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import WordNetLemmatizer


# In[5]:


stemmer = SnowballStemmer("english")
def preprocess(text):
    #replace one or more white-space characters with a space
    regex = re.compile(r"\s+")                               
    text = regex.sub(' ', text)    
    #lower-casing
    text = text.lower()          
    #removing digits/punctuations 
    regex = re.compile(r"[%s%s]" % (string.punctuation, string.digits))
    text = regex.sub(' ', text)           
    #removing stop words
    sw = stopwords.words('english')
    text = text.split()                                              
    text = ' '.join([w for w in text if w not in sw]) 
    #removing short words
    text = ' '.join([w for w in text.split() if len(w) >= 2])
    #stemming/lemmatization
    text = ' '.join([stemmer.stem(w) for w in text.split()]) 
    #text = ' '.join([(WordNetLemmatizer()).lemmatize(w) for w in text.split()]) 
    return text


# In[6]:


# setup the preprocessing->model pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

clf = Pipeline(steps=[
    ('pp', TfidfVectorizer(
        preprocessor=preprocess,
        lowercase=True, stop_words='english', 
        use_idf=True, smooth_idf=True, norm='l2',
        min_df=1, max_df=1.0, max_features=None, 
        ngram_range=(1, 1))),
    ('mdl', MultinomialNB())
    ])


# In[7]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'pp__norm':['l1', 'l2', None],
    'mdl__alpha':[0.01, 0.1, 0.2, 0.5, 1]
}
gscv = GridSearchCV(clf, param_grid, iid=False, cv=4, return_train_score=False)


# In[8]:


gscv.fit(Xtrain, ytrain)


# In[13]:


#print("<<Best estimator:>>", "\n", gscv.best_estimator_, "\n")
print("<<Best score:>>", "\n", gscv.best_score_, "\n")
print("<<Best Parameters:>>", "\n", gscv.best_params_, "\n")
#print("<<Best Results:>>", "\n", gscv.cv_results_, "\n")


# In[14]:


ypred = gscv.best_estimator_.predict(Xtest)
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))

