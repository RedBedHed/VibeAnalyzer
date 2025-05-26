import numpy as np
import re
import math
import argparse
import pickle

# From https://www.geeksforgeeks.org/removing-stop-words-nltk-python/#, adapted for this program.
stopwords = set(
    ['this', 'dont', 'yours', 'his', 'can', 'weren', 'themselves', 'hasnt', 'do', 'at', 'during', 
     'their', 'them', 'mightnt', 'were', 'wouldn', 'haven', 'has', 'couldn', 'myself', 'that', 
     'wasn', 'neednt', "youre", 'while', 'it', 'nor', 'm', 'doesnt', 'just', 'himself', 'with', 
     'youd', 'until', 'a', 'does', 'where', 'shes', 've', 'arent', 'ours', 'under', 'ourselves', 
     'are', 'you', 'be', 'once', 'aren', 'having', 'on', 'to', 'below', 'not', 'such', 'itself', 
     'but', 're', 'of', 'yourselves', 'then', 'ain', 'thatll', 'isnt', 'being', 'same', 'through', 
     'further', 'up', 'how', 'doesn', 'her', 'very', 'couldnt', 'werent', 'which', 'he', 'me', 'in', 
     'each', 'we', 'havent', 'isn', 'doing', 'because', 's', 'hasn', 'shant', 'down', 'as', 'didnt', 
     'only', 'herself', 'before', 'don', 'and', 'against', 'what', 'by', 'wont', 'for', 'so', 'above', 
     'been', 'or', 'again', 'shouldnt', 'whom', 'why', 'here', 'shouldve', 'hadn', 'shan', 'those', 'o', 
     'is', 'about', 'over', 'there', 't', 'after', 'youll', 'the', 'who', 'did', "you've", 'mustn', 'too', 
     'mightn', 'll', 'hadnt', 'if', 'was', 'both', 'am', 'these', 'most', 'they', 'few', 'off', 'now', 'had', 
     'out', 'our', 'into', 'needn', 'wasnt', 'between', 'yourself', 'she', 'other', 'an', 'shouldn', 'y', 'some', 
     'hers', 'i', 'ma', 'him', 'when', 'will', 'all', 'own', 'any', 'than', 'have', 'wouldnt', 'didn', 'mustnt', 
     'my', 'no', 'theirs', 'd', 'your', 'from', 'its', 'won', 'should', 'more']
)

# Class labels 1-6
CLASS_LABELS = ['sadness','joy','love','anger','fear','surprise']

# Constants.
CLASS_NUMBER = 6
K            = 10
SPLIT        = 5

"""
Class: Classifier

A Binary Naive Bayes "Emotional Analysis" classifier.

Implementation:
    Author: Ellie Moore

Training Data Used:
    https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset
    Author: Bhavik Jikadara
    License: https://creativecommons.org/licenses/by/4.0/
    Changes Made: None

References:
    https://web.stanford.edu/~jurafsky/slp3/4.pdf
    Authors: Daniel Jurafsky & James H. Martin

"""
class Classifier:

    def __init__(self, string=None, test=False):
        self.train('emotions.csv', string, test)

    def train(self, csv, string, test):
        samples = None

        # If testing, cross validate.
        if test:
            try:
                with open(csv, "rb") as f:
                    # Read the file.
                    samples = str(f.read())

                    # Split into samples.
                    samples = samples.split('\\r\\n')
                    samples = np.asarray(samples[1:-1])

                    # Shuffle samples.
                    np.random.shuffle(samples)
                    samples = samples.tolist()
            except:
                print("Error: missing dataset file.")
                return

            # K-fold cross validation.
            self.acc = 0
            for k in range(K):
                self.fold(samples, idx=k, test=True)
                print(f"Fold {k + 1}/{K} complete.")

            print(f"Model Accuracy: {(self.acc / K)*100}%")
            return

        # If not testing, classify. 
        try:    
            # Load the serialized model.     
            with open("log_prior.pickle", "rb") as f:
                self.log_prior = pickle.load(f)
            with open("setV.pickle", "rb") as f:
                self.setV = pickle.load(f)
            with open("log_likelihood.pickle", "rb") as f:
                self.log_likelihood = pickle.load(f)
        except:    
            # Generate a new model. 
            try:
                with open(csv, "rb") as f:
                    # Read the file.
                    samples = str(f.read())

                    # Split into samples.
                    samples = samples.split('\\r\\n')
                    samples = samples[1:-1]
            except:
                print("Error: missing dataset file.")
                return

            self.fold(samples)

            with open("log_prior.pickle", "wb") as f:
                pickle.dump(self.log_prior, f)
            with open("setV.pickle", "wb") as f:
                pickle.dump(self.setV, f)
            with open("log_likelihood.pickle", "wb") as f:
                pickle.dump(self.log_likelihood, f)

        if string is None:
            print("Error: missing string.")

        print(f"Post: \"{string}\"")
        print(f"Vibe: {self.classify(string)}")

    def fold(self, samples, idx=0, test=False):
        # If testing, use k folds.
        # Separate into training and validation sets.
        if test:
            fold_size = len(samples) // K + 1
            start = fold_size * idx
            end = start + fold_size
            if end > len(samples):
                end = len(samples)
            test_data = samples[start:end]
            samp_data = samples[:start]
            samp_data.extend(samples[end:])
            samples = samp_data

        # Bags for each class.
        bag = [{} for _ in range(CLASS_NUMBER)]

        # Number of samples per class.
        class_count = [0 for _ in range(CLASS_NUMBER)]

        # Log likelihoods for each class.
        self.log_likelihood = [{} for _ in range(CLASS_NUMBER)]

        # Model vocabulary from training data.
        vocabulary = []

        # Number of samples.
        sample_count = len(samples)

        # Iterate through samples.
        i = 0
        for sample in samples:
            # Split sample into features and class.
            u = sample.split(',')
            features = u[0]
            class_num = int(u[1])

            # Increment the sample count for 
            # this class.
            class_count[class_num] += 1

            # split feature string into individual features (words).
            words = features.split(" ")

            # Set of words we've seen for this feature string.
            seen = set()

            # Iterate through the words in the feature string.
            for w in words:

                # Filter stop words.
                if w in stopwords:
                    continue

                # Skip words we've seen (binary naive bayes).
                if w in seen:
                    continue

                # We've now seen this word.
                seen.add(w)

                # If the word is already in the bag for this class,
                # increment the frequency.
                if w in bag[class_num]:
                    bag[class_num][w] += 1

                # If the word isn't already in the bag for this class,
                # Add it to the vocabulary and bag with frequency=1.
                else:
                    vocabulary.append(w)
                    bag[class_num][w] = 1

            i += 1

        # Calculate the priors for each class.
        self.log_prior = np.asarray(class_count, dtype=float)
        self.log_prior /= sample_count
        self.log_prior = np.log(self.log_prior)

        # Calculate the likelihood for each word in each class.
        sig = [0 for _ in range(CLASS_NUMBER)]
        for c in range(CLASS_NUMBER):
            for w in vocabulary:
                cnt = bag[c][w] if w in bag[c] else 0
                sig[c] += cnt + 1

        for c in range(CLASS_NUMBER):
            for w in vocabulary:
                cnt = bag[c][w] if w in bag[c] else 0
                self.log_likelihood[c][w] = np.log((cnt + 1)/sig[c])

        # Convert the vocabulary list to a set.
        self.setV = set(vocabulary)

        # If testing, calculate the model accuracy.
        if test:
            # How many do we get right?
            right = 0
            for sample in test_data:
                # Split sample into features and class.
                u = sample.split(',')
                features = u[0]
                class_num = int(u[1])
                
                # Classify.
                e = self.classify(features, test=test)
                if class_num == e:
                    right += 1

            # Add the accuracy.
            self.acc += right / len(test_data)

    def classify(self, s, test=False):
        # Remove punctuation.
        str = re.sub(r'[^\w\s]', '', s)

        # Convert to lowercase.
        str = str.lower()

        # Split into tokens.
        str = str.split(' ')

        # Classify.
        mx = -math.inf
        class_num = -1
        for c in range(CLASS_NUMBER):
            lp = self.log_prior[c]

            seen = set()
            for w in str:                

                if w in stopwords:
                    continue

                if w in seen:
                    continue
                
                seen.add(w)

                if w in self.setV:
                    lp += self.log_likelihood[c][w]

            if lp > mx:
                mx = lp
                class_num = c

        if test:
            return class_num
        else:
            return CLASS_LABELS[ class_num ]

"""
Main
"""
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-t", help="Test Mode: Performs k-fold cross validation and assesses model accuracy.\nInput any value.")
    p.add_argument("-s", help="String: A string to analyze for emotionality.\nInput a string.")
    o = p.parse_args()
    if o.t and o.s:
        p.error("Argument must be either -t or -s.")
    elif o.t:
        c = Classifier(test=True)
    elif o.s:
        c = Classifier(string=o.s)
