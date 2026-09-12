# Part C: exact phrase and ordered proximity search over the positional index

from src.preprocess import analyze


# a phrase matches only if term i sits at start + i, so adjacency is checked, not co-occurrence
def phrase_search(index, query):
    terms = analyze(query)
    if not terms or any(term not in index for term in terms):
        return []

    docids = set(index[terms[0]])
    for term in terms[1:]:
        docids &= set(index[term])

    results = []
    for docid in sorted(docids):
        starts = set(index[terms[0]][docid])
        for offset, term in enumerate(terms[1:], start=1):
            starts &= {position - offset for position in index[term][docid]}  # shift back to the start
        if starts:
            results.append((docid, sorted(starts)))

    return results


# ordered proximity: the second term must appear within k positions after the first
def proximity_search(index, first, second, k):
    terms = analyze(first) + analyze(second)
    if len(terms) != 2 or any(term not in index for term in terms):
        return []

    left, right = terms
    results = []
    for docid in sorted(set(index[left]) & set(index[right])):
        pairs = [
            (p1, p2)
            for p1 in index[left][docid]
            for p2 in index[right][docid]
            if 0 < p2 - p1 <= k
        ]
        if pairs:
            results.append((docid, pairs))

    return results
