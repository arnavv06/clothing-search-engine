# Parts A-C: one structure serves as both the inverted index and the positional index

import math
from src.preprocess import analyze


# term -> {docid: [positions]}; df is len(postings) and tf is len(positions)
def build_index(documents):
    index = {}
    for doc in documents:
        for position, term in enumerate(analyze(doc["content"])):
            index.setdefault(term, {}).setdefault(doc["docid"], []).append(position)

    return index


# number of documents containing the term
def document_frequency(index, term):
    return len(index.get(term, {}))


# lnc: cosine length of each document vector, weight 1 + log10(tf) and no idf
def doc_norms(index):
    squares = {}
    for postings in index.values():
        for docid, positions in postings.items():
            weight = 1 + math.log10(len(positions))
            squares[docid] = squares.get(docid, 0.0) + weight ** 2

    return {docid: math.sqrt(total) for docid, total in squares.items()}
