#LOADING DATA
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.svm import SVC
import re
from nltk.stem import PorterStemmer
file_contents = open("emailSample1.txt","r").read()
vocabList = open("vocab.txt","r").read()
vocabList=vocabList.split("\n")[:-1]

vocabList_d={}
for ea in vocabList:
    value,key = ea.split("\t")[:]
    vocabList_d[key] = value

#PREPROCESSING EMAILS
def ProcessEmail(email_contents,vocabList_d):
    email_contents=email_contents.lower()
    email_contents = re.sub("[0-9]+","number",email_contents)
    email_contents = re.sub("[http|https]://[^\s]*","httpaddr",email_contents)
    email_contents = re.sub("[^\s]+@[^\s]+","emailaddr",email_contents)
    email_contents = re.sub("[$]+","dollar",email_contents)
    specialChar = ["<","[","^",">","+","?","!","'",".",",",":"]
    for char in specialChar:
        email_contents = email_contents.replace(str(char),"")
    email_contents = email_contents.replace("\n"," ")
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents= " ".join(email_contents)
    word_indices=[]
    for char in email_contents.split():
        if len(char) >1 and char in vocabList_d:
            word_indices.append(int(vocabList_d[char]))
    return word_indices
word_indices= ProcessEmail(file_contents,vocabList_d)

#DEFINING SPAM FEATURES
def emailFeatures(word_indices,vocabList_d):
    n = len(vocabList_d)
    features = np.zeros((n,1))
    for i in word_indices:
        features[i]=1
    return features

features = emailFeatures(word_indices,vocabList_d)
print("Length of feature vector: ",len(features))
print("Number of non-zero entries: ",np.sum(features))

#TRAINING THE SPAM CLASSIFIER
spam_mat = loadmat("spamTrain.mat")
X_train = spam_mat["X"]
y_train = spam_mat["y"]

C=0.1
spam_SVC = SVC(C=0.1,kernel = "linear")
spam_SVC.fit(X_train,y_train.ravel())
print("Training Accuracy:",(spam_SVC.score(X_train,y_train.ravel()))*100,"%")

#TESTING THE SPAM CLASSIFIER
spamtest_mat = loadmat("spamTest.mat")
X_test = spamtest_mat["Xtest"]
y_test = spamtest_mat["ytest"]
spam_SVC.predict(X_test)
print("Test Accuracy:",(spam_SVC.score(X_test,y_test.ravel()))*100,"%")

#WORDS THAT ARE MARKED AS SPAM
weights = spam_SVC.coef_[0]
weights_col = np.hstack((np.arange(1,1900).reshape(1899,1),weights.reshape(1899,1)))
df = pd.DataFrame(weights_col)
df.sort_values(by=[1],ascending=False,inplace=True)
predictors = []
idx = []
for i in df[0][:15]:
    for keys,values in vocabList_d.items():
        if str(int(i)) == values:
            predictors.append(keys)
            idx.append(int(values))
print("Top predictors of spam:")
for _ in range(15):
    print(predictors[_],"\t\t",round(df[1][idx[_]-1],6))
