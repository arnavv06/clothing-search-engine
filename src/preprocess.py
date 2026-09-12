# Part A: corpus loading, tokenization, stemming and the stop-word list

import os
import re
from nltk.stem import PorterStemmer

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus_100.txt")

# standard English stop words; applied by the ranker, never removed from the index,
# because dropping them would shift positions and break phrase search
STOPWORDS = frozenset("""
a an and are as at be been being but by can cannot could did do does doing
down for from had has have having he her here hers him his how i if in into is
it its me more most my no nor not of off on once only or other our ours out
over own same she should so some such than that the their theirs them then
there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours
""".split())

stemmer = PorterStemmer()


# parse the <DOC> blocks into dicts of docid, category, title, text and content
def load_corpus(path=CORPUS_PATH):
    raw = open(path, encoding="utf-8").read()
    documents = []
    for block in re.findall(r"<DOC>(.*?)</DOC>", raw, re.DOTALL):
        doc = {}
        for field in ("docid", "category", "title", "text"):
            value = re.search(rf"<{field.upper()}>(.*?)</{field.upper()}>", block, re.DOTALL)
            doc[field] = " ".join(value.group(1).split())
        # title + text is what gets indexed; category is only shown with results
        doc["content"] = doc["title"] + " " + doc["text"]
        documents.append(doc)

    return documents


# lowercase, drop punctuation and split into tokens
def tokenize(text):
    text = text.lower()
    text = re.sub(r"(\w)['’]s\b", r"\1", text)  # men's -> men, so it cannot collide with size "s"
    text = re.sub(r"\bt[\s-]?shirt", "tshirt", text)  # t-shirt / t shirt / tshirt -> one term

    return re.findall(r"[a-z0-9]+", text)


# full pipeline; a term's index in this list is also its position in the document
def analyze(text):
    return [stemmer.stem(token) for token in tokenize(text)]


# the stop list in stemmed form, which is what query terms are compared against
STOPWORD_STEMS = frozenset(stemmer.stem(word) for word in STOPWORDS)
