# writes the dictionary and index dumps that are required as deliverables

import math
import os
from src.preprocess import load_corpus
from src.indexer import build_index

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# term, df and idf; the header line records how many terms have idf = 0
def write_dictionary(index, n_docs, path):
    zero_idf = sum(1 for term in index if len(index[term]) == n_docs)
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# N = {n_docs} documents, {len(index)} terms, {zero_idf} terms with idf = 0\n")
        out.write(f"# {'term':<16}{'df':>5}{'idf':>9}\n")
        for term in sorted(index):
            df = len(index[term])
            out.write(f"{term:<18}{df:>5}{math.log10(n_docs / df):>9.4f}\n")


# term -> df -> [(docID, tf), ...]
def write_inverted_index(index, path):
    with open(path, "w", encoding="utf-8") as out:
        for term in sorted(index):
            postings = ", ".join(
                f"({docid}, {len(index[term][docid])})" for docid in sorted(index[term])
            )
            out.write(f"{term} -> df={len(index[term])} -> [{postings}]\n")


# term -> df -> [(docID, tf, [positions]), ...]
def write_positional_index(index, path):
    with open(path, "w", encoding="utf-8") as out:
        for term in sorted(index):
            postings = ", ".join(
                f"({docid}, {len(index[term][docid])}, {index[term][docid]})"
                for docid in sorted(index[term])
            )
            out.write(f"{term} -> df={len(index[term])} -> [{postings}]\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    documents = load_corpus()
    index = build_index(documents)

    write_dictionary(index, len(documents), os.path.join(OUTPUT_DIR, "dictionary.txt"))
    write_inverted_index(index, os.path.join(OUTPUT_DIR, "inverted_index.txt"))
    write_positional_index(index, os.path.join(OUTPUT_DIR, "positional_index.txt"))

    zero_idf = sorted(term for term in index if len(index[term]) == len(documents))
    print(f"{len(documents)} documents, {len(index)} terms")
    print(f"{len(zero_idf)} terms occur in every document (idf = 0): {', '.join(zero_idf[:8])} ...")
    print(f"wrote dictionary.txt, inverted_index.txt, positional_index.txt to {OUTPUT_DIR}")
