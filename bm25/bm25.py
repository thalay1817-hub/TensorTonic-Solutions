import math
import numpy as np
from collections import Counter

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):

    N = len(docs)

    # Document lengths
    doc_lens = np.array([len(doc) for doc in docs])
    avgdl = np.mean(doc_lens)

    # Term frequency per document
    tf_docs = [Counter(doc) for doc in docs]

    # Document frequency
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    # Remove duplicate query terms while keeping order
    query_terms = list(dict.fromkeys(query_tokens))

    # Initialize scores as numpy array
    scores = np.zeros(N)

    for term in query_terms:
        df_t = df.get(term, 0)

        # IDF calculation
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

        for i, tf_doc in enumerate(tf_docs):
            tf = tf_doc.get(term, 0)

            denom = tf + k1 * (1 - b + b * (doc_lens[i] / avgdl))

            if denom != 0:
                scores[i] += idf * (tf * (k1 + 1)) / denom

    return scores