# ECE467 Project 1
# @author Di Mei
# A text categorization system by using Rocchio/TF*IDF

import math, string, numpy
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')

def tfdf(doc_file, train_doc_tcount, train_token_dcount):
    doc_file_content = open(doc_file).read().lower()
    tokens = tokenizer.tokenize(doc_file_content)
    tokens = [stemmer.stem(token) for token in tokens]
    token_freq_dict = dict(Counter(tokens))
    train_doc_tcount[doc_file] = token_freq_dict
    for token in token_freq_dict:
        if token in train_token_dcount:
            temp = train_token_dcount[token] + 1
            train_token_dcount[token] = temp + 1
        else:
            train_token_dcount[token] = 1
    return train_doc_tcount, train_token_dcount

def doc_tfidf(per_doc_tcount, token_dcount, num_docs, train=True, vocab=[]):
    tfidf, tokens = [], []
    if train:
        tokens = token_dcount.keys()
    else:
        tokens = vocab
    for token in tokens:
        if token in per_doc_tcount:
            tf = math.log10(per_doc_tcount[token]+1)
            idf = math.log10(num_docs/token_dcount[token])
            tfidf.append(tf * idf)
        else:
            tfidf.append(0)
    return tfidf

def cat_centroid(cat, doc_tfidf, cat_doc, vocab_len):
    doc_list = cat_doc[cat]
    num_doc = len(doc_list)
    sum_vec = [0]*vocab_len
    for doc in doc_list:
        sum_vec = [a+b for a,b in zip(sum_vec, doc_tfidf[doc])]
    return [a/num_doc for a in sum_vec]

def train(train_file):
    train_file_content = open(train_file, "r")
    N = 0 # number of training doc
    train_doc_cat = {} # {doc: category}
    train_cat_doc = {} # {cat: [docs]}
    train_doc_tcount = {} # {doc: {token: count}}
    train_doc_tfidf = {} # {doc: [tf*idf]}
    train_token_dcount = {} # {token: count}
    train_cat_tfidf = {} # {category: tfidf}

    for doc_line in train_file_content:
        doc_cat = doc_line.split()
        train_doc_cat[doc_cat[0]] = doc_cat[1]
        if doc_cat[1] not in train_cat_doc:
            train_cat_doc[doc_cat[1]] = [doc_cat[0]]
        else:
            train_cat_doc[doc_cat[1]].append(doc_cat[0])
        tfdf(doc_cat[0], train_doc_tcount, train_token_dcount)
        N += 1
    for doc in train_doc_tcount:
        train_doc_tfidf[doc] = doc_tfidf(train_doc_tcount[doc], train_token_dcount, N)
    for cat in train_cat_doc:
        train_cat_tfidf[cat] = cat_centroid(cat, train_doc_tfidf, train_cat_doc, len(train_token_dcount))
    
    return train_cat_tfidf, train_token_dcount.keys()

def test(test_file, cat_centroid, vocab):
    test_file_content = open(test_file, "r")
    N = 0 # number of training doc
    test_doc_tcount = {} # {doc: {token: count}}
    test_token_dcount = {} # {token: count}
    test_doc_tfidf = {} # {doc: [tf*idf]}
    test_pred = {} # {doc: category}

    for doc_line in test_file_content:
        doc = doc_line.split()[0]
        tfdf(doc, test_doc_tcount, test_token_dcount)
        N += 1
    for doc in test_doc_tcount:
        test_doc_tfidf[doc] = doc_tfidf(test_doc_tcount[doc], test_token_dcount, N, train=False, vocab=vocab)
    for doc in test_doc_tfidf:
        doc_sim = []
        temp1 = numpy.array(test_doc_tfidf[doc])
        for cat in cat_centroid:
            temp2 = numpy.array(cat_centroid[cat])
            doc_sim.append(dot(temp1, temp2)/(norm(temp1)*norm(temp2)))
        test_pred[doc] = list(cat_centroid)[doc_sim.index(max(doc_sim))]
    
    return test_pred

if __name__ == '__main__':
    # input file showing a list of training documents' paths
    train_file = input("Enter the file listing training documents: ")
    # input file showing a list of testing documents' paths
    test_file = input("Enter the file listing test documents: ")
    print("training ...")
    cat_centroid, vocab = train(train_file)
    print("complete training!")
    print("testing ...")
    test_pred = test(test_file, cat_centroid, vocab)
    print("complete testing!")
    # store predictions into a given output file
    predict_file = input("input the name of output file: ")
    predict_file_content = open(predict_file, "w")
    for doc in test_pred:
        predict_file_content.write(doc + " " + test_pred[doc] + '\n')