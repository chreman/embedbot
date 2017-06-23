from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import udf, explode, concat
from pyspark.sql.types import ArrayType, StringType


class NounExtractor(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, inputCol=None, outputCol=None):
        super(NounExtractor, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCols=None, outputCol=None):
        pass

    def _transform(self, df):
        def extract_nouns(fulltext):
            import nltk
            nltk.data.path.append("~/nltk_data")
            nltk.data.path.append("/home/hadoop/nltk_data")
            sentences = nltk.sent_tokenize(fulltext)
            nouns = []
            for sent in sentences:
                for word, pos in nltk.pos_tag(nltk.tokenize
                                                  .word_tokenize(sent)):
                    if pos == ('NN' or 'NNP'):
                        nouns.append(word)
            return nouns
        udf_extract_nouns = udf(extract_nouns, ArrayType(StringType()))
        return df.withColumn(self.outputCol,
                             udf_extract_nouns(df[self.inputCol]))


class StringConcatenator(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, inputCols=None, outputCol=None):
        super(StringConcatenator, self).__init__()
        # kwargs = self.__init__._input_kwargs
        # self.setParams(**kwargs)
        self.inputCols = inputCols
        self.outputCol = outputCol

    def setParams(self, inputCols=None, outputCol=None):
        # kwargs = self.setParams._input_kwargs
        # return self._set(**kwargs)
        pass

    def _transform(self, df):
        col1, col2 = self.inputCols
        return df.withColumn(self.outputCol, concat(df[col1], df[col2]))


class StringListAssembler(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, inputCols=None, outputCol=None):
        super(StringListAssembler, self).__init__()
        # kwargs = self.__init__._input_kwargs
        # self.setParams(**kwargs)
        self.inputCols = inputCols
        self.outputCol = outputCol

    def setParams(self, inputCols=None, outputCol=None):
        # kwargs = self.setParams._input_kwargs
        # return self._set(**kwargs)
        pass

    def _transform(self, df):

        def concatenate_lists(*cols):
            cl = []
            for col in cols:
                cl.extend(col)
            return cl
        schema = ArrayType(StringType())
        udf_concatenate_lists = udf(concatenate_lists, schema)
        if len(self.inputCols) == 2:
            col1, col2 = self.inputCols
            return df.withColumn(self.outputCol,
                                 udf_concatenate_lists(df[col1],
                                                       df[col2]))
        if len(self.inputCols) == 3:
            col1, col2, col3 = self.inputCols
            return df.withColumn(self.outputCol,
                                 udf_concatenate_lists(df[col1],
                                                       df[col2],
                                                       df[col3]))


class SentTokenizer(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, inputCol=None, outputCol=None):
        super(SentTokenizer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCols=None, outputCol=None):
        pass

    def _transform(self, df):
        def sent_tokenize(x):
            import nltk
            nltk.data.path.append("~/nltk_data")
            nltk.data.path.append("/home/hadoop/nltk_data")
            return nltk.sent_tokenize(x)

        udf_sent_tokenize = udf(sent_tokenize, ArrayType(StringType()))
        return df.withColumn(self.outputCol,
                             udf_sent_tokenize(df[self.inputCol]))


class ColumnExploder(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, inputCol=None, outputCol=None):
        super(ColumnExploder, self).__init__()
        # kwargs = self.__init__._input_kwargs
        # self.setParams(**kwargs)
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCols=None, outputCol=None):
        # kwargs = self.setParams._input_kwargs
        # return self._set(**kwargs)
        pass

    def _transform(self, df):
        df = df.select('*', explode(df[self.inputCol]).alias(self.outputCol))
        return df


class ColumnSelector(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, outputCols=None):
        super(ColumnSelector, self).__init__()
        # kwargs = self.__init__._input_kwargs
        # self.setParams(**kwargs)
        self.outputCols = outputCols

    def setParams(self, inputCols=None, outputCol=None):
        # kwargs = self.setParams._input_kwargs
        # return self._set(**kwargs)
        pass

    def _transform(self, df):
        df = df.select(self.outputCols)
        return df
