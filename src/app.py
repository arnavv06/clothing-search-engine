# Part D: command line interface for all three search modes

from src.indexer import document_frequency
from src.preprocess import analyze, STOPWORD_STEMS
from src.search import load_engine, search

HELP = """modes:
  cotton shirt            free-text ranked retrieval (lnc.ltc cosine)
  "cotton shirt"          exact phrase search (positional index)
  cotton WITHIN/3 shirt   ordered proximity search, within k positions
commands:
  :boost                  toggle the phrase-proximity boost on free-text queries
  :help                   show this message
  :quit                   exit"""


# say why a query returned nothing instead of just printing an empty list
def explain_empty(engine, query):
    terms = [term for term in analyze(query.strip('"')) if term not in STOPWORD_STEMS]
    known = [term for term in terms if term in engine["index"]]
    if not known:
        return "no query term occurs in the corpus"
    if all(document_frequency(engine["index"], term) == len(engine["norms"]) for term in known):
        return "every query term occurs in all 100 documents (idf = 0), so all cosine scores are 0 - try phrase mode"

    return "no document matched"


# docid, score, category and title; positional queries also show positions and a snippet
def print_results(response, engine):
    results = response["results"]
    print(f"\n{response['mode']}: {len(results)} result(s)")

    for rank_position, result in enumerate(results, start=1):
        if response["mode"] in ("vsm", "hybrid"):
            score = f"{result['score']:.4f}"
        else:
            score = f"{result['score']} match" + ("" if result["score"] == 1 else "es")
        print(f"  {rank_position:>2}. {result['docid']}  {score:>9}  [{result['category']}] {result['title']}")
        if result["positions"]:
            print(f"      positions: {result['positions']}")
            print(f"      ...{result['snippet']}...")

    if not results:
        print(f"  {explain_empty(engine, response['query'])}")
    if response["suggestions"]:
        hints = ", ".join(f"{wrong} -> {right}" for wrong, right in response["suggestions"].items())
        print(f"  did you mean: {hints}")


# read queries until the user quits
def main():
    engine = load_engine()
    boost = False
    print(f"clothing search engine - {len(engine['documents'])} products indexed")
    print(HELP)

    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == ":quit":
            break
        if query == ":help":
            print(HELP)
            continue
        if query == ":boost":
            boost = not boost
            print(f"phrase-proximity boost {'on' if boost else 'off'}")
            continue

        print_results(search(engine, query, boost=boost), engine)


if __name__ == "__main__":
    main()
