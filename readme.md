# Document Summarization Through K Means Clustering

Given [data set] of 850 sentences, we are going to create a 10 sentence summary.

For the 1st iteration, we will accomplish this by:

1. Split document into array of sentences
2. Tokenize, stem, stop word each sentence
3. Cluster sentences into 10 groups using K Means clustering by NLTK
4. Calculate the cosine similarity matrix for all sentence pairs in a given cluster, yielding 10 matrices, 1 per cluster
5. For each Cosine Similarity matrix, Sum scores of each row to determine which row (sentence) has highest similarity with the most documents
6. Return the sentence from each cluster that has highest similarity with other sentences of the cluster (returned by step 5)
7. Combine all 10 resulting high similarity sentences into 1 paragraph, which is to become the summary.
8. Sort the summary's sentences in the order that they appear in the original document, to maintain positional relationships of original document.
9. Return summary


### Tools Used

Python's [Natural Language Toolkit]
- tokenizer
- stop words
- Porter stemmer
- tfidf vectorizer
- k means clustering algorithm





[data set]: hw4data-docs4sum.txt
[Natural Language Toolkit]: http://www.nltk.org/index.html