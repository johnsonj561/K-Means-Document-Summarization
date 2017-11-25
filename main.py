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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from modules.TextPreProcessor import removeShortDocs
from modules.TextPreProcessor import removeStopWords
from modules.TextPreProcessor import stemSentences


print """
Assignment 3: Document Summarization
Information Retrieval: Florida Atlantic University, Fall 2017
Justin J, jjohn273
"""

# define data set and parameters
data_path = 'hw4data-docs4sum.txt'
raw_data = open(data_path, 'r').read()
ps = PorterStemmer()
nltk_stop_words = set(stopwords.words('english'))
cluster_count = 10
min_sentence_length = 35


####################################
# PRE-PROCESSING
####################################

# split document into sentences and strip whitespace (delimeted by line)
sentences = raw_data.split('\n')
sentences = map(lambda sentence: sentence.strip(), sentences)

# remove sentences that do not contribute meaning by assuming short sentences have less meaning
sentences = removeShortDocs(sentences, min_sentence_length)

# remove stop words from all sentences
processedSentences = removeStopWords(sentences, nltk_stop_words)

# stem all tokens of all sentences
processedSentences = stemSentences(sentences, ps)


####################################
# Apply K Means Clustering
####################################
	
# create tfidf matrix from the processed sentences
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processedSentences)

# cluster our tokenized sentences into 10 groups
kMeansCluster = KMeans(n_clusters=cluster_count)
kMeansCluster.fit(tfidf_matrix)
clusters = kMeansCluster.labels_.tolist()


####################################
# Organize Cluster Results
####################################

# Create new dictionary that tracks which cluster each sentence belongs to
# keeps copy of original sentences and stemmed sentences
# sentenceDictionary { idx: { text: String, stemmed: String, cluster: Number } }
sentenceDictionary = {}
for idx, sentence in enumerate(sentences):
	sentenceDictionary[idx] = {}
	sentenceDictionary[idx]['text'] = sentence
	sentenceDictionary[idx]['cluster'] = clusters[idx]
	sentenceDictionary[idx]['stemmed'] = processedSentences[idx]

# Create new dictionary that contains 1 entry for each cluster
# each key in dictionary will point to array of sentences, all of which belong to that cluster
# we attach the index to the sentenceDictionary object so we can recall the original sentence
clusterDictionary = {}
for key, sentence in sentenceDictionary.items():
	if sentence['cluster'] not in clusterDictionary:
		clusterDictionary[sentence['cluster']] = []
	clusterDictionary[sentence['cluster']].append(sentence['stemmed'])
	sentence['idx'] = len(clusterDictionary[sentence['cluster']]) - 1
		
		
####################################
# Calculate Cosine Similarity Scores
####################################		

# For each cluster of sentences,
# Find the sentence with highet cosine similarity over all sentences in cluster
maxCosineScores = {}
for key, clusterSentences in clusterDictionary.items():
	maxCosineScores[key] = {}
	maxCosineScores[key]['score'] = 0
	tfidf_matrix = vectorizer.fit_transform(clusterSentences)
	cos_sim_matrix = cosine_similarity(tfidf_matrix)
	for idx, row in enumerate(cos_sim_matrix):
		sum = 0
		for col in row:
			sum += col
		if sum > maxCosineScores[key]['score']:
			maxCosineScores[key]['score'] = sum
			maxCosineScores[key]['idx'] = idx



####################################
# Construct Document Summary
####################################	

# for every cluster's max cosine score,
# find the corresponding original sentence
resultIndices = []
i = 0
for key, value in maxCosineScores.items():
	cluster = key
	idx = value['idx']
	stemmedSentence = clusterDictionary[cluster][idx]
	# key corresponds to the sentences index of the original document
	# we will use this key to sort our results in order of original document
	for key, value in sentenceDictionary.items():
		if value['cluster'] == cluster and value['idx'] == idx:
			resultIndices.append(key)

resultIndices.sort()

# Iterate over sentences and construct summary output
result = ''
for idx in resultIndices:
	result += sentences[idx] + ' '
		

print result












