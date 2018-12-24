# ******************** Importing Libraries ************************************

from textblob import TextBlob               # For processing textual data
import tweepy                               # It is a Tweeter API.For Fetching Tweets
import pandas as pd                         # For DataFrames
import numpy as np                          # For Arrays
import matplotlib.pyplot as plt             # For Visaulisation
import re                                   # For Regular Expression
import nltk                                 # For Natural Language Processing
nltk.download("stopwords")                  # Downloading Stopwords
from nltk.corpus import stopwords           # Importing Stopwords
from nltk.stem.porter import PorterStemmer  # Importing Stemmer for Exchanging words with root words
from sklearn.feature_extraction.text import CountVectorizer # For Creating Bag of Words Model
from sklearn.cross_validation import train_test_split # For Splitting the dataset into Training and Test Set



# ******************** Importing Datasets for Training ************************
cols = ['sentiment','id','date','query_string','user','text']
dataset = pd.read_csv("train.csv",header=None, names=cols,encoding= 'latin-1' )

# ******************** function to clean tweet text ***************************

def clean_tweet(t,corpus): 
    twe=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", t)
    twe=twe.lower()
    twe=twe.split()
    ps = PorterStemmer()
    twe = [ps.stem(word) for word in twe if not word in set(stopwords.words("english"))]
    twe = ' '.join(twe)
    corpus.append(twe)

# ******************** Cleaning Tweets of Training Dataset ********************
#dataset.drop(['id','date','query_string','user'],axis=1,inplace=True)
l=dataset.iloc[790000:810000,:]
corpus=[]

for i in range(790000,810000):
    clean_tweet(l["text"][i],corpus)
    if(i%1000 == 0):
        print(i)
    
    
# ******************** Creating Bag Of Words Model ****************************    

cv = CountVectorizer(max_features =17000 )
X = cv.fit_transform(corpus).toarray()  
y = l.loc[790000:810000,"sentiment"].values

# ******************** Splitting the dataset into Training and Test Set *******

X_train,X_test,y_train,ytest = train_test_split(X,y,test_size= 0.30,random_state =0)


# ******************** Feature Scaling ****************************************
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''
# ******************** Fitting Machine Learning Model to Training Test *******************

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) 
classifier.fit(X_train,y_train)

# ******************** Predict The Test Result *****************************

y_pred = classifier.predict(X_test)

# ******************** Lets Making A Confusion Matrix *************************
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest,y_pred)
ac= accuracy_score(ytest,y_pred)


#                     --------------------------------------------- 
#*********************|Now,We will do Twitter Sentimental Analysis|************
#                     ---------------------------------------------



# ******************** Assigning Keys for API Authentication ******************

Cus_API_key = "eBfktu63CRWxykCBesSdg9GRl"
Cus_API_Secretkey = "m8mpWEGWVxoetY58zvGQrTTfv18JluQrkTc10D7SfQTY16wK9a"
Access_Token="821343349215031296-G0DgUhMV2rlRiD31oMHIL6Xi6WQZ8Zp"
Access_Token_key="J9Wokn9ZlpwTThPznt52pVjqC37nr0Wiznd1vdBS5XQo3"


# ******************** Twitter API Authentication *****************************

auth = tweepy.OAuthHandler(Cus_API_key,Cus_API_Secretkey)
auth.set_access_token(Access_Token,Access_Token_key)
api = tweepy.API(auth)

# ******************** Fetching Tweets of any Twitter User ********************

fetched_tweets = api.search("Donald trump",count = 200)

#******************* function to classify sentiment ***************************
        
    

              
# ******************** Parsing Tweets *************************

tweets= []
for tweet in fetched_tweets:
                parsed_tweet = {} 
                parsed_tweet['text'] = tweet.text 
                if tweet.retweet_count > 0: 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet)

                
# ******************* Getting Live Sentiments **********************
length = len(tweets)
corpuss = []
for i in range(0,length):
    clean_tweet(tweets[i]["text"],corpuss)
Xt = cv.fit_transform(corpuss).toarray()
#X=X.append(Xt)
yt_pred = classifier.predict(Xt)
    
d = {"Tweets": tweets , "Sentiments": yt_pred}
Df= pd.DataFrame(d)

