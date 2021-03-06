from os import path
import time
import logging
import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, Word2Vec
from transformers import StringConcatenator

def main(args):
    logging.basicConfig(level=logging.INFO, datefmt="%Y/%m/%d %H:%M:%S")
    formatter = logging.Formatter('%(asctime)-15s %(name)s [%(levelname)s] '
                                  '%(message)s')
    logger = logging.getLogger('WorkflowLogger')
    if args.logpath:
        logname = time.ctime()+".log"
        logpath = path.join(args.logpath, logname)
        fh = logging.FileHandler(logpath)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info('Beginning workflow')

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    ########################
    # READING RAW FILES

    logger.info('Creating base df.')
    df = spark.read.json(args.input)

    df = df.dropDuplicates(['doi'])
    if args.sample:
        df = df.sample(False, args.sample, 42)
    if args.debug:
        df.printSchema()
        df.explain(True)
    # logger.info('Base df created, papers in sample: %d' % df.count())

    #########################
    # SPLITTING DF INTO METADATA AND FULLTEXT

    fulltexts = df.select('doi', 'abstract', 'fulltext')
    #########################
    # DEFINING TRANSFORMERS and ESTIMATORS

    logger.info('Initializing feature pipeline.')

    stringconcat = StringConcatenator(inputCols=["abstract", "fulltext"],
                                      outputCol="texts")
    tokenizer = RegexTokenizer(inputCol="texts",
                               outputCol="words", pattern="\\W")
    word2Vec = Word2Vec(vectorSize=500, minCount=20,
                        maxIter=args.maxIter, numPartitions=args.numPartitions,
                        windowSize=args.windowSize,
                        stepSize=args.stepSize,
                        inputCol="words", outputCol="w2v")

    w2vpipeline = Pipeline(stages=[stringconcat,
                                   tokenizer,
                                   word2Vec])
    logger.info('Fitting feature pipeline.')
    w2vpipeline_model = w2vpipeline.fit(fulltexts)
    w2vmodel = w2vpipeline_model.stages[-1]
    vectors = w2vmodel.getVectors()
    vectors.write.save(args.output+"df.parquet")
    logger.info('Saving model.')
    logger.info('Ending workflow, shutting down.')
    sc.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do stuff')
    parser.add_argument('--input', dest='input', help='relative or absolute '
                        'path of the input folder')
    parser.add_argument('--output', dest='output', help='relative or absolute '
                        'path of the output folder')
    parser.add_argument('--sample', dest='sample', help='fraction of dataset '
                        'to sample, e.g. 0.01', type=float)
    parser.add_argument('--maxIter', dest='maxIter', help='number of '
                        'maximum iterations, default 1', type=int, default=1)
    parser.add_argument('--numPartitions', dest='numPartitions',
                        help='number of partitions in word2vec step, '
                        'default 1', type=int, default=1)
    parser.add_argument('--windowSize', dest='windowSize',
                        help='size of window for surrounding words, '
                        'default 5', type=int, default=5)
    parser.add_argument('--stepSize', dest='stepSize',
                        help='Step size to be used for each iteration of '
                        'optimization (>= 0), default 0.025',
                        type=float, default=0.025)
    parser.add_argument('--logpath', dest='logpath', help='relative or '
                        'absolute path of the logfile')
    args = parser.parse_args()
    main(args)
