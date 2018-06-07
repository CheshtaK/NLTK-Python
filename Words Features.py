import nltk
import random
from nltk.corpus import movie_reviews

#Pickle module
import pickle

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:100]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#print((find_features(movie_reviews.words('pos/cv000_29590.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

#Naive Bayes Classifier

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent: ",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#Pickle module for saving classifiers

#Writing 
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#Opening and using classifier
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
