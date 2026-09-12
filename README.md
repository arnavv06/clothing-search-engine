# Clothing Search Engine

Information Retrieval Assignment-1 (CSD358). A small search engine over a corpus of 100 clothing
product descriptions, with an inverted index, lnc.ltc ranked retrieval, and a positional index
supporting phrase and proximity search.

**Group members:** Daksh Jain (2410110113) · Arnav Jyoti (2410110071)

**GitHub:** [github.com/arnavv06/clothing-search-engine.git](https://github.com/arnavv06/clothing-search-engine.git)

## Requirements

```
Python 3.9+
pip install nltk
```

`nltk` is used only for its Porter stemmer. No corpus download is needed.

## Running

All commands are run from the project root.

```bash
python -m src.build_index     # build the index and write the dumps in outputs/
python -m src.app             # interactive search (Part D)
python run_tests.py           # the full Part E test suite -> outputs/test_report.txt
```

## Search modes

The interface picks the mode from the query syntax:

| Input                     | Mode                                                           |
| ------------------------- | -------------------------------------------------------------- |
| `cotton shirt`          | free-text ranked retrieval, lnc.ltc cosine                     |
| `"cotton shirt"`        | exact phrase search, using the positional index                |
| `cotton WITHIN/3 shirt` | ordered proximity, second term within k positions of the first |

Commands inside the app: `:boost` toggles the phrase-proximity boost on free-text queries,
`:help` lists the modes, `:quit` exits.

Example:

```
query> "festive wear"

phrase: 10 result(s)
   1. D005    1 match  [Saree] Women's Printed Daily Wear Saree - Blue
      positions: [25]
      ...indian wear it features festive wear printed border and...
```

## Files

```
data/corpus_100.txt      the supplied corpus
src/preprocess.py        Part A - corpus loading, tokenization, stemming, stop-word list
src/indexer.py           Parts A-C - the index and the lnc document norms
src/vsm.py               Part B - lnc.ltc cosine ranking
src/positional.py        Part C - phrase and ordered proximity search
src/search.py            query routing, phrase boost, spelling suggestions
src/app.py               Part D - command line interface
src/build_index.py       writes the index dumps in outputs/
run_tests.py             Part E - all mandatory test queries
outputs/                 generated deliverables
PLAN.md                  design notes and corpus analysis
```

## How it works

**Pre-processing.** Lowercase, strip possessives, drop punctuation, split on `[a-z0-9]+`, then
Porter stem. `t-shirt`, `t shirt` and `tshirt` all become one term. The same function processes
documents and queries.

**Index.** One structure, `term -> {docid: [positions]}`, serves as both the inverted index and the
positional index: df is the number of postings and tf is the number of positions, so nothing is
stored twice. `outputs/inverted_index.txt` and `outputs/positional_index.txt` print it in the two
forms the assignment asks for.

**Ranking.** lnc.ltc. Document weights are `1 + log10(tf)` with no idf, normalised by a cosine
length computed once per document. Query weights are `(1 + log10(tf)) * log10(N/df)`, also
normalised. Results are sorted by score descending, then by docID ascending.

**Stop words.** Stop words are kept in the index and removed only from the query vector. Removing
them from the index would shift every later position and break phrase search. This corpus makes
the point clearly: 41 of the 141 terms occur in all 100 documents, so `wear` and `festiv` have
idf = 0 and are stop words by any df-based rule, yet `"festive wear"` is a phrase query the
assignment requires.

## Novelty

Three additions beyond the required parts:

1. **Stop-word policy derived from the data.** `outputs/dictionary.txt` records df and idf for
   every term, and its header counts the 41 terms with idf = 0 that motivate the policy above.
2. **Phrase-aware ranking.** With `:boost` on, a free-text query adds `0.3 / (1 + min_gap)` to each
   cosine score, where `min_gap` is the smallest distance between query terms in that document.
   For `regular fit` this moves phrase matches in the top 10 from 5/10 to 10/10 while keeping the
   same 45 retrieved documents.
3. **"Did you mean".** Query terms missing from the index are matched against the 141-term
   vocabulary by edit distance, so `cotten shrit` suggests `cotton` and `shirt`.

## Testing

`python run_tests.py` runs 13 free-text queries, 7 phrase queries, 6 proximity queries with
different values of k, a query whose terms do not occur in the corpus, and a misspelled query. It
writes the top-10 results for each to `outputs/test_report.txt` and verifies 66 structural
assertions, for example that phrase matches are always a subset of the co-occurrence set and that
proximity results grow monotonically with k. No expected document IDs are hard-coded; the script
exits non-zero if any check fails.

The report ends with the three cases where positional information changes the result:

- **`regular fit`** - 45 documents contain both terms, 15 contain the phrase; the boost reorders
  the ranking without shrinking the result set.
- **`festive wear`** - both terms have df = 100, so the ltc query vector is all zeros and VSM
  returns nothing, while phrase search returns 10 documents.
- **`cotton kurta`** - 5 documents contain both terms but none contains the phrase, so a Boolean
  AND would return 5 false positives.
