# query routing, plus the two ranking extras: phrase boost and spelling suggestions

import re
from src.preprocess import load_corpus, analyze, tokenize, STOPWORD_STEMS
from src.indexer import build_index, doc_norms
from src.vsm import rank
from src.positional import phrase_search, proximity_search

PHRASE_BOOST = 0.3  # weight of the proximity bonus added to the cosine score
WITHIN_RE = re.compile(r"^(\w+)\s+WITHIN/(\d+)\s+(\w+)$", re.IGNORECASE)  # cotton WITHIN/3 shirt


# build the index once and keep the documents for displaying results
def load_engine():
    documents = load_corpus()
    index = build_index(documents)

    return {
        "documents": {doc["docid"]: doc for doc in documents},
        "index": index,
        "norms": doc_norms(index),
    }


# Levenshtein distance, used only for the did-you-mean suggestions
def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            cost = 0 if a == b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current

    return previous[-1]


# for query terms missing from the index, offer the closest term in the vocabulary
def suggestions(index, query):
    found = {}
    for term in analyze(query):
        if term in index or term in STOPWORD_STEMS:
            continue
        closest = min(index, key=lambda known: edit_distance(term, known))
        limit = 1 if len(term) <= 4 else 2  # short words need a tighter limit, else silk -> size
        if edit_distance(term, closest) <= limit:
            found[term] = closest

    return found


# smallest ordered gap between consecutive query terms inside one document
def min_gap(index, terms, docid):
    gaps = []
    for left, right in zip(terms, terms[1:]):
        if docid in index.get(left, {}) and docid in index.get(right, {}):
            ordered = [p2 - p1 for p1 in index[left][docid] for p2 in index[right][docid] if p2 > p1]
            if ordered:
                gaps.append(min(ordered))

    return min(gaps) if gaps else None


# rerank the VSM results by how close the query terms are; the result set is unchanged
def rank_with_boost(index, norms, query, top_k=10):
    terms = analyze(query)
    boosted = []
    for docid, score in rank(index, norms, query, top_k=len(norms)):
        gap = min_gap(index, terms, docid)
        if gap is not None:
            score += PHRASE_BOOST / (1 + gap)
        boosted.append((docid, score))

    return sorted(boosted, key=lambda item: (-item[1], item[0]))[:top_k]


# tokenize() and analyze() emit one token each, so index positions line up with these unstemmed tokens
def snippet(document, position, width=4):
    tokens = tokenize(document["content"])
    start = max(0, position - width)

    return " ".join(tokens[start:position + width + 1])


# pick the retrieval mode from the query syntax and return a uniform response
def search(engine, query, boost=False, top_k=10):
    index, norms, documents = engine["index"], engine["norms"], engine["documents"]
    query = query.strip()

    within = WITHIN_RE.match(query)
    if within:
        first, k, second = within.group(1), int(within.group(2)), within.group(3)
        hits = proximity_search(index, first, second, k)
        mode = f"proximity (k={k})"
        terms_searched = f"{first} {second}"
    elif len(query) > 1 and query.startswith('"') and query.endswith('"'):
        hits = phrase_search(index, query[1:-1])
        mode = "phrase"
        terms_searched = query[1:-1]
    else:
        scored = rank_with_boost(index, norms, query, top_k) if boost else rank(index, norms, query, top_k)
        results = [
            {
                "docid": docid,
                "title": documents[docid]["title"],
                "category": documents[docid]["category"],
                "score": score,
                "positions": [],
                "snippet": "",
            }
            for docid, score in scored
        ]

        return {
            "mode": "hybrid" if boost else "vsm",
            "query": query,
            "results": results,
            "suggestions": suggestions(index, query),
        }

    hits.sort(key=lambda hit: (-len(hit[1]), hit[0]))  # most occurrences first, then docID
    results = []
    for docid, positions in hits[:top_k]:
        first_position = positions[0][0] if isinstance(positions[0], tuple) else positions[0]
        results.append(
            {
                "docid": docid,
                "title": documents[docid]["title"],
                "category": documents[docid]["category"],
                "score": len(positions),
                "positions": positions,
                "snippet": snippet(documents[docid], first_position),
            }
        )

    return {
        "mode": mode,
        "query": query,
        "results": results,
        "suggestions": suggestions(index, terms_searched),
    }
