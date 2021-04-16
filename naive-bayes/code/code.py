#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
df = pd.read_csv("head.csv")
clf_label = "sentiment_category" # can be changed with topic_category
X = df.the_document_tokens
if(clf_label == "sentiment_category"):
    y = df.sentiment_category
elif(clf_label == "topic_category"):
    y = df.topic_category

print(type(X))
print(type(y))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
print(type(X_train))
type(X_train)

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names()
len(feature_names)
X_train = pd.DataFrame(X_train.toarray(), columns=feature_names) # converting Series to Dataframe
X_test = pd.DataFrame(X_test.toarray(), columns=feature_names) # converting Series to Dataframe
X_train[clf_label+"_label"] = np.array(y_train, dtype='object')
print(X_train.info())
categories = [category for category in X_train.sentiment_category_label.unique()]
X_trains = {}
for sent_category in categories:
    X_trains[sent_category] = X_train[X_train['sentiment_category_label']==sent_category]
    X_trains[sent_category] = X_trains[sent_category].iloc[: , :-1]
len(X_trains["neg"]), len(X_trains["pos"]) # number of documents in negative and positive cases in training data
max_freq_in_negative_case = {} # created to store most freqeunt words in negative labeled documents
for word in X_trains["neg"]:
    freq = X_trains["neg"][word].sum() # sum of the specific word in negative case 
    max_freq_in_negative_case[word] = freq
max_freq_in_negative_case = dict(sorted(max_freq_in_negative_case.items(), key=lambda item: item[1], reverse=True))

max_freq_in_positive_case = {} # created to store most freqeunt words in positive labeled documents
for word in X_trains["pos"]:
    freq = X_trains["pos"][word].sum() # sum of the specific word in positive case 
    max_freq_in_positive_case[word] = freq


max_freq_in_positive_case = dict(sorted(max_freq_in_positive_case.items(), key=lambda item: item[1], reverse=True))

difference = {} # to store freqeuency difference of common words in positive and negative cases
for key in max_freq_in_negative_case.keys(): 
    difference[key] = abs(max_freq_in_positive_case[key] - max_freq_in_negative_case[key])

difference = dict(sorted(difference.items(), key=lambda item: item[1], reverse=True))

y_train = np.array(y_train, dtype='object')
y_test = np.array(y_test, dtype='object')

print(y_train)

# Naive Bayes Classifier (Works for both sentiment category and topic category)
class SixCategoryNaiveBayesClassifier: 
    # outcomesPD is X, y is y_train -> dtype = 'object'
    def fit(self, X, y_train, clf_label):
        (unique, counts) = np.unique(y_train, return_counts=True) # count the unique number of classes in the category label
        frequencies = np.asarray((unique, counts)).T 
        self.freq = {} # to store the frequency of the categories seperately
        for category, frequency in frequencies:
            self.freq[category] = frequency  # store the freqeuncy for each category
        total_examples = sum(self.freq.values()) # total number of documents in the dataset
        prob_category = {} # to store the category probabilities
        for category in self.freq.keys():
            prob_category[category] = self.freq[category] / total_examples # to calculate the probability of the category class
        print(prob_category) # print the probabilities of each category seperately
        number_of_word_in_category_case = {} # to store the number of word in the specific category
        outcomesCategoryPD = {} # will be used in filtering the categories from the training data
        prob_word_given_category = {} # category, word
        for category in self.freq.keys(): 
            outcomesCategoryPD[category] = X[X[clf_label]==category] # filtering operation
            number_of_word_in_category_case[category] = outcomesCategoryPD[category].iloc[: , :-1].values.sum() # total number of word is specific category case
            prob_word_given_category[category] = {} # to store the conditional probability of each word for each class
        vocabularies = [word for word in X.iloc[: , :-1]] # vocabularies in training data
        vocab_size = len(vocabularies) # vocabulary size
        prior_category = {} # to store the prior probability of any category
        posterior_category = {} # to store the posterior information for each category
        for category in self.freq.keys():
            for word in vocabularies:
                count_specific_word_in_category_case = outcomesCategoryPD[category][word].sum() + 1 # count the specific word in the specific category
                prob_word_given_category[category][word] = np.log(count_specific_word_in_category_case / (number_of_word_in_category_case[category]+vocab_size)) # calculate the conditional probability using Laplace Smooting
                prior_category[category] = np.log(prob_category[category]) # store the prior probability for each category
                posterior_category[category] = prob_word_given_category[category][word] + prior_category[category]
        return prior_category, prob_word_given_category,vocabularies,outcomesCategoryPD # return the prior category, conditional probabilities, vocabularies
    
    def predict(self, X): # X is X_test -> outcomesTestPD
        test_views = []
        for test_index in X.index:
            test_features = X.loc[test_index]
            test_features = test_features[test_features > 0]
            test_view = [feature for feature in test_features.to_frame().T.columns]
            test_views.append(test_view)
        print(len(test_views[0]))
        y_pred = np.array([]).astype(np.object)
        for view in test_views:
            posterior_for_category = {}
            for category in self.freq.keys():
                posterior_for_category[category] = prior_category[category]
                for token in view:
                    if(token in vocabularies):
                        posterior_for_category[category] += prob_word_given_category[category][token] # sum all the log likelihood with the prior probability 
            max_key = max(posterior_for_category, key=posterior_for_category.get) # return argmax of the predicted class
            y_pred = np.append(y_pred, max_key) # append the predicted class to y_pred 
        return y_pred # return y_pred

def calculate_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

nb_clf_for_categories = SixCategoryNaiveBayesClassifier()
prior_category, prob_word_given_category,vocabularies,outcomesCategoryPD = nb_clf_for_categories.fit(X_train,y_train,clf_label+"_label")

y_pred = nb_clf_for_categories.predict(X_test)

accuracy = calculate_accuracy(y_test, y_pred)

#prob_word_given_category["neg"]["like"], prob_word_given_category["pos"]["like"]
#prob_word_given_category["neg"]["book"], prob_word_given_category["pos"]["book"]
dict(sorted(prob_word_given_category["neg"].items(), key=lambda item: item[1], reverse=True))

dict(sorted(prob_word_given_category["pos"].items(), key=lambda item: item[1], reverse=True))

def printConfusionMatrix(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
printConfusionMatrix(y_pred, y_test)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)

X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names()
print(len(feature_names))

X_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)

X_train[clf_label+"_label"] = np.array(y_train, dtype='object')


y_train = np.array(y_train, dtype='object')
y_test = np.array(y_test, dtype='object')

nb = SixCategoryNaiveBayesClassifier()

prior_category, prob_word_given_category,vocabularies,outcomesCategoryPD = nb.fit(X_train,y_train,clf_label+"_label")

y_pred = nb.predict(X_test)

accuracy = calculate_accuracy(y_test, y_pred)

print(accuracy)

most_strong_words_in_positive = dict(sorted(prob_word_given_category["pos"].items(), key=lambda item: item[1], reverse=True))
most_strong_words_in_negative = dict(sorted(prob_word_given_category["neg"].items(), key=lambda item: item[1], reverse=True))

def printConfusionMatrix(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
printConfusionMatrix(y_pred, y_test)


# # TF-IDF (using CountVectorizer)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text  import TfidfTransformer

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)  #instantiate CountVectorizer()

word_count_vector = vectorizer.fit_transform(X_train) # this steps generates word counts for the words in the documets 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)


# print idf values 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=vectorizer.get_feature_names(),columns=["idf_weights"])

# sort ascending 
df_idf.sort_values(by=['idf_weights']) #Notice that the words ‘just’ and ‘like’ have the lowest IDF values. This is expected as these words appear in each and every document in our collection. The lower the IDF value of a word, the less unique it is to any particular document.

#  Compute the TFIDF score for the documents
X_train=vectorizer.transform(X_train) 
 
# tf-idf scores 
tf_idf_vector=tfidf_transformer.transform(X_train)

feature_names = vectorizer.get_feature_names() 
#get tfidf vector for first document 
first_document_vector=tf_idf_vector[0]
shape = first_document_vector.shape


#print the scores 
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
df.sort_values(by=["tfidf"],ascending=False)

sorted_vals = df.sort_values(by=["tfidf"],ascending=False)
most_weighted_words = {}
for index in sorted_vals.index:
    features = sorted_vals.loc[index]
    if(features["tfidf"] > 0):
        most_weighted_words[features.name] = features["tfidf"]
print(most_weighted_words.keys())

X_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
X_train[clf_label+"_label"] = np.array(y_train, dtype='object')

y_train = np.array(y_train, dtype='object')
y_test = np.array(y_test, dtype='object')

nb = SixCategoryNaiveBayesClassifier()

prior_category, prob_word_given_category,vocabularies,outcomesCategoryPD = nb.fit(X_train,y_train,clf_label+"_label")

X_test = vectorizer.transform(X_test)
print(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
y_pred = nb.predict(X_test)
accuracy = calculate_accuracy(y_test, y_pred)
print(accuracy)
# # Bigram
vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(2, 2))
X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names()
len(feature_names)


X_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)

X_train = X_train.iloc[:,:42031]
X_train[clf_label+"_label"] = np.array(y_train, dtype='object')

X_test = X_test.iloc[:,:42031]

y_train = np.array(y_train, dtype='object')
y_test = np.array(y_test, dtype='object')

nb_clf_for_categories = SixCategoryNaiveBayesClassifier()

prior_category, prob_word_given_category,vocabularies,outcomesCategoryPD = nb_clf_for_categories.fit(X_train,y_train,clf_label+"_label")

y_pred = nb_clf_for_categories.predict(X_test)

accuracy = calculate_accuracy(y_test, y_pred)

def printConfusionMatrix(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
printConfusionMatrix(y_pred, y_test)

dict(sorted(prob_word_given_category["pos"].items(), key=lambda item: item[1], reverse=True))

dict(sorted(prob_word_given_category["neg"].items(), key=lambda item: item[1], reverse=True))

