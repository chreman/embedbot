import pandas as pd
import numpy as np
import logging

from sklearn.metrics.pairwise import cosine_similarity


class w2vEngine(object):
    """docstring for w2vEngine"""
    def __init__(self, modelpath):
        super(w2vEngine, self).__init__()
        logname = ("EmbedBot.log")
        self.log = logging.getLogger("w2vEngine")
        self.log.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)-15s %(name)s '
                                      '[%(levelname)s] %(message)s')
        fh = logging.FileHandler(logname)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
        self.modelpath = modelpath
        store = pd.HDFStore(modelpath)
        df = store.get('df')
        self.vocabulary = df['word'].tolist()
        self.log.info("Size of vocabulary: %d" % len(self.vocabulary))
        self.mat = np.row_stack(df['vector'].tolist())

    def get_synonyms(self, query_term, n):
        i = self.vocabulary.index(query_term)
        y = self.mat[i].reshape(1, -1)
        sims = cosine_similarity(self.mat, y).flatten()
        best_sims = sims.argsort()
        best_sims = list(best_sims[-10:-1])
        best_sims.reverse()
        results = []
        for i in best_sims:
            candidate = self.vocabulary[i]
            if not candidate.startswith(query_term):
                if not query_term.startswith(candidate):
                    results.append(self.vocabulary[i])
        return results[:n]

    def get_abstractions(self, query_terms, signs, n):
        query_terms = [q.replace("-", "") for q in query_terms]
        indices = [self.vocabulary.index(q) for q in query_terms]
        vectors = self.mat[indices]
        query_vectors = [signs[i]*vectors[i] for i in range(len(query_terms))]
        query_vector = sum(query_vectors).reshape(1, -1)
        sims = cosine_similarity(self.mat, query_vector).flatten()
        best_sims = sims.argsort()
        best_sims = list(best_sims[-10:-1])
        best_sims.reverse()
        candidates = [self.vocabulary[i] for i in best_sims]
        results = []
        for c in candidates:
            if not any([c.startswith(q) for q in query_terms]):
                results.append(c)
        return results[:n]
