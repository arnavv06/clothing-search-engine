# Part E: runs every mandatory test query and writes outputs/test_report.txt

import os
import sys
from src.preprocess import analyze, STOPWORD_STEMS
from src.indexer import document_frequency
from src.positional import phrase_search, proximity_search
from src.search import load_engine, search


OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "test_report.txt")

# the last query is the mandatory one whose terms do not occur in the corpus
FREE_TEXT_QUERIES = [
    "cotton shirt",
    "black t-shirt",
    "stretch denim jeans",
    "festive kurta",
    "winter jacket",
    "high waist leggings",
    "printed saree",
    "fleece hoodie",
    "machine washable",
    "navy blue sweatshirt",
    "women casual dress",
    "festive wear",
    "waterproof silk sherwani",
]

PHRASE_QUERIES = [
    "cotton shirt",
    "stretch denim",
    "festive wear",
    "high waist",
    "breathable fabric",
    "regular fit",
    "zip closure",
]

PROXIMITY_QUERIES = [
    ("cotton", "shirt", 3),
    ("stretch", "denim", 4),
    ("winter", "wear", 3),
    ("festive", "kurta", 4),
    ("quilted", "jacket", 2),
    ("women", "leggings", 4),
]

report = []
checks = []


# collect the report lines in memory, written to the file at the end
def write(line=""):
    report.append(line)


# record a structural assertion; no expected docID is ever hard-coded
def check(description, passed):
    checks.append((description, passed))
    write(f"  [{'PASS' if passed else 'FAIL'}] {description}")


# documents containing every term, ignoring where the terms appear
def cooccurrence(index, terms):
    if not terms or any(term not in index for term in terms):
        return set()

    return set.intersection(*[set(index[term]) for term in terms])


# one line per result, plus positions for phrase and proximity hits
def write_results(response):
    if not response["results"]:
        write("    (no results)")
    for position, result in enumerate(response["results"], start=1):
        score = result["score"]
        score = f"{score:.4f}" if isinstance(score, float) else f"{score} match(es)"
        write(f"    {position:>2}. {result['docid']}  {score:>10}  [{result['category']}] {result['title']}")
        if result["positions"]:
            write(f"        positions: {result['positions']}")
    if response["suggestions"]:
        write(f"    did you mean: {response['suggestions']}")


def section(title):
    write()
    write("=" * 78)
    write(title)
    write("=" * 78)


# E.1: free-text queries ranked by lnc.ltc cosine
def test_free_text(engine):
    section(f"PART E.1 - FREE-TEXT QUERIES (lnc.ltc cosine), {len(FREE_TEXT_QUERIES)} queries")
    for query in FREE_TEXT_QUERIES:
        response = search(engine, query)
        write()
        write(f"query: {query!r}  ->  {len(response['results'])} result(s) shown")
        write_results(response)

        scores = [result["score"] for result in response["results"]]
        ids = [result["docid"] for result in response["results"]]
        ordered = sorted(zip(scores, ids), key=lambda pair: (-pair[0], pair[1]))
        check(f"{query!r}: at most 10 results", len(ids) <= 10)
        check(f"{query!r}: sorted by score desc then docID asc", list(zip(scores, ids)) == ordered)


# E.2: exact phrase queries, compared against plain co-occurrence
def test_phrases(engine):
    section(f"PART E.2 - EXACT PHRASE QUERIES (positional index), {len(PHRASE_QUERIES)} queries")
    index = engine["index"]
    for phrase in PHRASE_QUERIES:
        response = search(engine, f'"{phrase}"')
        matched = phrase_search(index, phrase)
        both = cooccurrence(index, analyze(phrase))
        write()
        write(f"phrase: {phrase!r}  ->  {len(matched)} document(s) contain the phrase, "
              f"{len(both)} contain all terms anywhere")
        write_results(response)

        check(f"{phrase!r}: phrase matches are a subset of the co-occurrence set",
              {docid for docid, _ in matched} <= both)
        terms = analyze(phrase)
        check(f"{phrase!r}: every reported position really holds the phrase",
              all(analyze(engine["documents"][docid]["content"])[start:start + len(terms)] == terms
                  for docid, starts in matched for start in starts))


# E.3: ordered proximity queries, including a sweep over k
def test_proximity(engine):
    section(f"PART E.3 - ORDERED PROXIMITY QUERIES, {len(PROXIMITY_QUERIES)} queries with different k")
    index = engine["index"]
    order_matters = False
    for first, second, k in PROXIMITY_QUERIES:
        response = search(engine, f"{first} WITHIN/{k} {second}")
        write()
        write(f"query: {first} WITHIN/{k} {second}  ->  {len(response['results'])} result(s)")
        write_results(response)

        forward_hits = proximity_search(index, first, second, k)
        reversed_hits = proximity_search(index, second, first, k)
        write(f"    reversed ({second} WITHIN/{k} {first}): {len(reversed_hits)} result(s)")
        order_matters = order_matters or len(forward_hits) != len(reversed_hits)
        check(f"{first} WITHIN/{k} {second}: every gap satisfies 0 < p2-p1 <= k",
              all(0 < p2 - p1 <= k for _, pairs in forward_hits for p1, p2 in pairs))

    check("proximity search is ordered: at least one query gives a different result reversed",
          order_matters)

    write()
    write("k-sweep (result sets must grow monotonically with k):")
    for first, second in [("quilted", "jacket"), ("women", "leggings"), ("stretch", "denim")]:
        sets = [{docid for docid, _ in proximity_search(index, first, second, k)} for k in range(1, 9)]
        write(f"  {first} WITHIN/k {second}: " + "  ".join(f"k={k}:{len(s)}" for k, s in enumerate(sets, start=1)))
        check(f"{first}/{second}: result set grows monotonically with k",
              all(sets[i] <= sets[i + 1] for i in range(len(sets) - 1)))

        phrase_docs = {docid for docid, _ in phrase_search(index, f"{first} {second}")}
        check(f"{first}/{second}: proximity at k=1 equals exact phrase search", sets[0] == phrase_docs)


# E.4: a query with no corpus terms, and a misspelled query
def test_unknown_terms(engine):
    section("PART E.4 - QUERY WITH A TERM THAT DOES NOT OCCUR IN THE CORPUS")
    index = engine["index"]
    query = "waterproof silk sherwani"
    terms = analyze(query)
    write()
    write(f"query: {query!r}")
    write(f"  term -> df: {[(term, document_frequency(index, term)) for term in terms]}")
    response = search(engine, query)
    write_results(response)

    check(f"{query!r}: none of its terms occur in the corpus",
          all(document_frequency(index, term) == 0 for term in terms))
    check(f"{query!r}: returns no results instead of failing", response["results"] == [])

    write()
    typo = "cotten shrit"
    response = search(engine, typo)
    write(f"query: {typo!r} (misspelled)")
    write_results(response)
    check(f"{typo!r}: spelling correction suggests in-vocabulary terms",
          all(right in index for right in response["suggestions"].values()))


# E.5: the three cases where positional information changes the result
def test_positional_changes_results(engine):
    section("PART E.5 - CASES WHERE POSITIONAL INFORMATION CHANGES THE RESULT")
    index = engine["index"]

    write()
    write("CASE 1 - 'regular fit': positional evidence changes the ORDER")
    both = cooccurrence(index, analyze("regular fit"))
    phrase_docs = {docid for docid, _ in phrase_search(index, "regular fit")}
    plain = search(engine, "regular fit")["results"]
    boosted = search(engine, "regular fit", boost=True)["results"]
    write(f"  {len(both)} documents contain both terms, only {len(phrase_docs)} contain the phrase.")
    write(f"  plain lnc.ltc top-10        : {[result['docid'] for result in plain]}")
    write(f"  with phrase-proximity boost : {[result['docid'] for result in boosted]}")
    in_plain = sum(1 for result in plain if result["docid"] in phrase_docs)
    in_boosted = sum(1 for result in boosted if result["docid"] in phrase_docs)
    write(f"  phrase matches inside top-10: {in_plain}/10 plain -> {in_boosted}/10 boosted")
    write("  Both terms appear in every 'Casual Fit' description too, so plain VSM ranks those")
    write("  first. The boost uses the gap between the terms, which promotes real 'Regular Fit'")
    write("  products without dropping the other documents from the result set.")
    check("'regular fit': the boost puts more phrase matches in the top 10", in_boosted > in_plain)
    check("'regular fit': the boost reorders but does not shrink the result set",
          {result["docid"] for result in search(engine, "regular fit", top_k=100)["results"]}
          == {result["docid"] for result in search(engine, "regular fit", boost=True, top_k=100)["results"]})

    write()
    write("CASE 2 - 'festive wear': positional retrieval answers a query VSM cannot")
    vsm_hits = search(engine, "festive wear")["results"]
    phrase_hits = phrase_search(index, "festive wear")
    dfs = [(term, document_frequency(index, term)) for term in analyze("festive wear")]
    write(f"  term -> df: {dfs}")
    write(f"  Every term occurs in all {len(engine['documents'])} documents, so idf = 0 for each,")
    write("  the ltc query vector is all zeros, and no document gets a non-zero cosine score.")
    write(f"  free-text VSM : {len(vsm_hits)} results")
    write(f"  phrase search : {len(phrase_hits)} results {[docid for docid, _ in phrase_hits]}")
    check("'festive wear': VSM returns nothing because every term has idf = 0", vsm_hits == [])
    check("'festive wear': phrase search still finds the documents", len(phrase_hits) > 0)

    write()
    write("CASE 3 - 'cotton kurta': co-occurrence without adjacency")
    both = cooccurrence(index, analyze("cotton kurta"))
    phrase_docs = {docid for docid, _ in phrase_search(index, "cotton kurta")}
    write(f"  {len(both)} documents contain both terms: {sorted(both)}")
    write(f"  {len(phrase_docs)} documents contain the phrase.")
    write("  A Boolean AND or a co-occurrence check would return all of them as matches; the")
    write("  positional index shows the two terms are never adjacent, so the phrase does not occur.")
    check("'cotton kurta': terms co-occur but the phrase never does", both and not phrase_docs)


# index invariants that must hold whatever the corpus contains
def test_index_invariants(engine):
    section("INDEX SANITY CHECKS")
    index = engine["index"]
    documents = engine["documents"]
    write()

    check("df equals the number of postings for every term",
          all(document_frequency(index, term) == len(index[term]) for term in index))
    total_tokens = sum(len(analyze(doc["content"])) for doc in documents.values())
    total_tf = sum(len(positions) for postings in index.values() for positions in postings.values())
    check(f"sum of all tf equals the {total_tokens} tokens in the corpus", total_tokens == total_tf)
    check("positions are sorted, unique and inside the document",
          all(positions == sorted(set(positions))
              and max(positions) < len(analyze(documents[docid]["content"]))
              for postings in index.values() for docid, positions in postings.items()))
    check("every document has a cosine norm", len(engine["norms"]) == len(documents))
    check("stop words are kept in the index so phrase search can use them",
          any(term in STOPWORD_STEMS for term in index))


if __name__ == "__main__":
    engine = load_engine()
    write("Clothing search engine - Part E test report")
    write(f"{len(engine['documents'])} documents, {len(engine['index'])} index terms")

    test_free_text(engine)
    test_phrases(engine)
    test_proximity(engine)
    test_unknown_terms(engine)
    test_positional_changes_results(engine)
    test_index_invariants(engine)

    passed = sum(1 for _, ok in checks if ok)
    section(f"SUMMARY: {passed}/{len(checks)} checks passed")
    for description, ok in checks:
        if not ok:
            write(f"  FAILED: {description}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.write("\n".join(report) + "\n")

    print(f"{len(FREE_TEXT_QUERIES)} free-text, {len(PHRASE_QUERIES)} phrase, "
          f"{len(PROXIMITY_QUERIES)} proximity queries run")
    print(f"{passed}/{len(checks)} checks passed")
    print(f"full report written to {OUTPUT_PATH}")
    sys.exit(0 if passed == len(checks) else 1)  # non-zero exit if any check failed
