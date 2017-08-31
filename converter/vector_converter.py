import argparse
import logging
import time
from os import path

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import pandas as pd
import numpy as np
import pickle


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

    raw_df = spark.read.parquet(args.input)
    words = [r['word'] for r in raw_df.select('word').collect()]
    mat = np.row_stack([r
                        for r in raw_df.select('vector').rdd.toLocalIterator()])
    with open(args.output+"/words.pkl", "wb") as outfile:
        pickle.dump(words, outfile)
    with open(args.output+"/mat.pkl", "wb") as outfile:
        pickle.dump(mat, outfile)

    logger.info('Ending workflow, shutting down.')
    sc.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do stuff')
    parser.add_argument('--input', dest='input', help='relative or absolute '
                        'path of the input folder')
    parser.add_argument('--output', dest='output', help='relative or absolute '
                        'path of the output folder')
    parser.add_argument('--logpath', dest='logpath', help='relative or '
                        'absolute path of the logfile')
    args = parser.parse_args()
    main(args)
