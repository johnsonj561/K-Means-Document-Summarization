# A set of functions that operate on an array of document strings
from nltk.tokenize import word_tokenize

# Remove Stop Words
# Removes all stop words as they are defined in stopWords
# Returns array of documents s.t. there are no stop words
def removeStopWords(sentences, stopWords):
	resultSentences = []
	for idx, sentence in enumerate(sentences):
		tokens = word_tokenize(sentence)
		tokens = filter(lambda token: token not in stopWords, tokens)
		resultSentences.append(' '.join(tokens))
	return resultSentences


# Stem Sentences
# Applies the stemmer's stem method to all tokens
# Returns array of documents s.t. all terms are stems
def stemSentences(sentences, stemmer):
	stemmedSentences = []
	for idx, sentence in enumerate(sentences):
		tokens = word_tokenize(sentence)
		tokens = map(lambda token: stemmer.stem(token), tokens)
		stemmedSentences.append(' '.join(tokens))
	return stemmedSentences
	
	
# Remove Short Documents
# Returns array of documents greater than minLength
def removeShortDocs(sentences, minLength):
	return filter(lambda sentence: len(sentence) > minLength, sentences)