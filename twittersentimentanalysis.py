# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

totalpos = 0
totalneg = 0

#import csv file
import csv
with open('twitterdataset.csv',encoding='latin-1') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    tweets = []
    for row in readCSV:
        
        tweet = row[5]
        tweets.append(tweet)
        
        
#unpickling the classifier and vectorizer
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    
    
with open('tfidfmodel.pickle','rb') as f:
    tfdif = pickle.load(f)
        
#cleanse data
predata = []
for i in range(0,len(tweets)):
    review = re.sub(r"^http://twitpic.com/[a-zA-Z0-9]*\s"," ",str(tweets[i])) #remove all non characters
    review = re.sub(r"\s+http://twitpic.com/[a-zA-Z0-9]*\s"," ",review)
    review = re.sub(r"\s+http://twitpic.com/[a-zA-Z0-9]*$"," ",review)
    review = review.lower() #convert to lower
    review = re.sub(r"that's","that is",review)
    review = re.sub(r"there's","there is",review)
    review = re.sub(r"what's","what is",review)
    review = re.sub(r"where's","where is",review)
    review = re.sub(r"it's","it is",review)
    review = re.sub(r"who's","who is",review)
    review = re.sub(r"she's","she is",review)
    review = re.sub(r"he's","he is",review)
    review = re.sub(r"i'm","i am",review)
    review = re.sub(r"they're","they are",review)
    review = re.sub(r"who're","who are",review)
    review = re.sub(r"ain't","am not",review)
    review = re.sub(r"wouldn't","would not",review)
    review = re.sub(r"shouldn't","should not",review)
    review = re.sub(r"can't","can not",review)
    review = re.sub(r"couldn't","could not",review)
    review = re.sub(r"won't","will not",review)
    review = re.sub(r"\W"," ",review)
    review = re.sub(r"\d"," ",review)
    review = re.sub(r"\s+[a-z]\s+"," ",review)
    review = re.sub(r"\s+[a-z]$"," ",review)
    review = re.sub(r"^[a-z]\s+"," ",review)
    review = re.sub(r"\s+"," ",review)
    sent = clf.predict(tfdif.transform([review]).toarray())
    #print(review,":",sent)
    #predata.append(review)
    if sent[0] == 1:
        totalpos +=1
    else:
        totalneg +=1
        
#plotting the bar chart
import matplotlib.pyplot as plt
import numpy as np

objects = ['Positive','Negavtive']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[totalpos,totalneg],alpha = 0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Total Number')
plt.title('Number of positive and negative tweets')

plt.show()
       

    

    


    
