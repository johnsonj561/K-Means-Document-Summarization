# Information Retrieval
# Florida Atlantic University, Fall 2017
# Justin Johnson jjohn273
#
# Assignment 3: Document Summarization
# 1. Cluster docs4sum.txt into 10 clusters
# 2. Calculate the centroid of each cluster
# 3. Use centroid of each cluster to create a 10 sentence summary
# 4. Explain choice of clustering and similarity

# Notes
# The data set contains several sentences that contain just 1 punctuation character, '?'
# We will ignore these sentences to improve clustering algorithm

import re
import numpy
from os import listdir
from os.path import join, abspath
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


print """
Assignment 3: Document Summarization
Information Retrieval: Florida Atlantic University, Fall 2017
Justin J, jjohn273
"""

# define data set
data_path = 'hw4data-docs4sum.txt'
raw_data = open(data_path, 'r').read()

# define stemmer and stop words
ps = PorterStemmer()
nltk_stop_words = set(stopwords.words('english'))

# split document into sentences and strip whitespace (delimeted by line)
sentences = raw_data.split('\n')
sentences = map(lambda sentence: sentence.strip(), sentences)

# tokenize, stem, and remove stop words from each sentence
tokenizedSentences = []
for idx, sentence in enumerate(sentences):
	tokens = word_tokenize(sentence)
	tokens = filter(lambda token: token not in nltk_stop_words, tokens)
	tokens = map(lambda token: ps.stem(token), tokens)
	tokenizedSentences.append(' '.join(tokens))

# remove sentences that are equal to '?'	
#for key, value in tokenizedSentences.items():
#	if len(value) == 1 and value[0] == "?":
#		print key
#		del tokenizedSentences[key]

		
# create tfidf matrix from the processed sentences
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tokenizedSentences)


# cluster our tokenized sentences into 10 groups
cluster_count = 10
kMeansCluster = KMeans(n_clusters=cluster_count)
kMeansCluster.fit(tfidf_matrix)
clusters = kMeansCluster.labels_.tolist()

# create a sentence/cluster dictionary
# also tally cluster totals
# sentenceClusterDict { idx: { text: String, cluster: Number } }
sentenceClusterDict = {}
clusterTotals = numpy.zeros(10)
for idx, sentence in enumerate(sentences):
	sentenceClusterDict[idx] = {}
	sentenceClusterDict[idx]['text'] = sentence
	sentenceClusterDict[idx]['cluster'] = clusters[idx]
	clusterTotals[clusters[idx]] += 1


# create dictionary that contains 10 entries, 1 entry for each cluster
# each key in dictionary will point to list of sentences belonging to that cluster
clusterDict = {}
for key, value in sentenceClusterDict.items():
	print value
	if value['cluster'] not in clusterDict:
		clusterDict[value['cluster']] = []
	clusterDict[value['cluster']].append(value['text'])

# we now have a dictionary with 10 entries, each entry contains sentences of the same cluster

# for each cluster

# calculate the centroid, or calculate cosine similarity matrix
# sum each row of cosine similarity matrix, the row with highest sum is the doc that has the most in common with the othe documents!
# return this sentence as the summary of the cluster