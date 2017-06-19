from pyspark.ml.feature import Word2VecModel
import logging


class w2vEngine(object):
    """docstring for w2vEngine"""
    def __init__(self, sparksession, modelpath):
        super(w2vEngine, self).__init__()
        logname = ("EmbedBot.log")
        self.log = logging.getLogger("w2vEngine")
        self.log.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)-15s %(name)s '
                                      '[%(levelname)s] %(message)s')
        fh = logging.FileHandler(logname)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
        self.spark = sparksession
        self.modelpath = modelpath
        self.model = Word2VecModel.load(modelpath)
        self.vocabulary = self.get_vocabulary()

    def get_vocabulary(self):
        vocabulary = set()
        rows = self.model.getVectors().select('word').collect()
        for row in rows:
            vocabulary.add(row['word'])
        self.log.info("Size of vocabulary: %d" % len(vocabulary))
        return vocabulary

    def get_synonyms(self, query, n):
        syns = self.model.findSynonyms(query, n)
        return [s['word'] for s in syns.collect()]

    def get_abstractions(self, query_terms, signs):
        query_df = self.spark.createDataFrame([([q.replace("-", "")],)
                                              for q in query_terms],
                                              ["words"])
        vectors = self.model.transform(query_df)
        vectors = [r['w2v'] for r in vectors.collect()]
        query_vectors = [signs[i]*vectors[i] for i in range(len(query_terms))]
        query_vector = sum(query_vectors)
        synonyms = self.model.findSynonyms((-1) * query_vector, 5)
        results = [s['word'] for s in synonyms.collect()]
        results = [r for r in results if r not in query_terms]
        return results[0]
