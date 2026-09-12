# Part B: ranked retrieval with the lnc.ltc weighting scheme

import math
from src.preprocess import analyze, STOPWORD_STEMS
from src.indexer import document_frequency


# ltc: (1 + log10(tf)) * log10(N/df) for each query term, then cosine normalised
def query_weights(index, query, n_docs):
    counts = {}
    for term in analyze(query):
        if term not in STOPWORD_STEMS:
            counts[term] = counts.get(term, 0) + 1

    weights = {}
    for term, count in counts.items():
        df = document_frequency(index, term)
        if df:
            weights[term] = (1 + math.log10(count)) * math.log10(n_docs / df)

    norm = math.sqrt(sum(weight ** 2 for weight in weights.values()))
    if norm == 0:  # every query term has df = N, so idf = 0 and there is nothing to rank on
        return {}

    return {term: weight / norm for term, weight in weights.items()}


# accumulate cosine scores over the postings of each query term
def rank(index, norms, query, top_k=10):
    weights = query_weights(index, query, len(norms))

    scores = {}
    for term, query_weight in weights.items():
        for docid, positions in index[term].items():
            doc_weight = 1 + math.log10(len(positions))
            scores[docid] = scores.get(docid, 0.0) + query_weight * doc_weight / norms[docid]

    matched = [(docid, score) for docid, score in scores.items() if score > 0]  # drop idf = 0 matches
    ranked = sorted(matched, key=lambda item: (-item[1], item[0]))  # score desc, then docID asc

    return ranked[:top_k]
