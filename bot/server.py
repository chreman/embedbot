import tweepy

import json
import logging
import time

import os
from os import path
import re
import pickle
import argparse

from http.client import IncompleteRead

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from w2vEngine import w2vEngine


def init_spark_context():
    # load spark context
    sconf = SparkConf().setAppName("embedbot")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=sconf)
    spark = SparkSession(sc)
    return spark


class EmbedBot(object):
    """docstring for EmbedBot"""
    def __init__(self, spark, configpath, modelpath):
        self.spark = spark
        logname = ("EmbedBot.log")
        self.log = logging.getLogger("eb")
        self.log.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)-15s %(name)s '
                                      '[%(levelname)s] %(message)s')
        fh = logging.FileHandler(logname)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
        self.configpath = configpath
        self.modelpath = modelpath
        self.log.info("Configpath: %s" % self.configpath)
        self.log.info("Modelpath: %s" % self.modelpath)

    def initialize(self):
        self.config = {}
        self.config['refresh_interval'] = 120
        self.config['cachedir'] = "tmp"
        if not os.path.isdir(self.config.get('cachedir')):
            os.makedirs(self.config.get('cachedir'))

        self.log.info("Initializing.")
        self.load_state()
        self.load_config()
        self.start_engine()
        self.connect()
        self.log.info("Initialized succesfully.")

    def load_config(self):
        with open(self.configpath, "r") as infile:
            config = json.load(infile)
        for k, v in config.items():
            self.config[k] = v
        self.log.info("Config loaded.")

    def load_state(self):
        try:
            with open(path.join(self.config.get('cachedir'), "state.pkl"),
                      "rb") as statefile:
                self.state = pickle.load(statefile)
            self.log.info("State loaded.")
        except Exception as e:
            self.log.error(e)
            self.state = {}
            self.state['last_mention_id'] = 1
            self.state['mention_queue'] = []

    def save_state(self):
        with open(path.join(self.config.get('cachedir'), "state.pkl"),
                  "wb") as statefile:
            pickle.dump(self.state, statefile)
        self.log.info("State saved.")

    def start_engine(self):
        global w2vengine
        w2vengine = w2vEngine(self.spark, self.modelpath)

    def connect(self):
        auth = tweepy.OAuthHandler(self.config['api_key'],
                                   self.config['api_secret'])
        auth.set_access_token(self.config['access_key'],
                              self.config['access_secret'])
        self.api = tweepy.API(auth)
        self.id = self.api.me().id
        self.screen_name = self.api.me().screen_name

    def check_mentions(self):
        try:
            current_mentions = self.api.mentions_timeline(since_id=self.state['last_mention_id'],
                                                          count=100)
            # reduce to direct mentions only
            current_mentions = [t for t in current_mentions
                                if re.split('[^@\w]', t.text)[0]
                                == '@'+self.screen_name]
            if len(current_mentions) != 0:
                self.state['last_mention_id'] = current_mentions[0].id
            self.state['last_mention_time'] = time.time()
            self.state['mention_queue'] += reversed(current_mentions)
            self.log.info('Mentions updated ({} retrieved, {} total in queue)'
                          .format(len(current_mentions),
                                  len(self.state['mention_queue'])))
        except tweepy.TweepError as e:
            self.log.error('Can\'t retrieve mentions.')
            self.log.error(e)
        except IncompleteRead as e:
            self.log.error('Incomplete read error -- skipping mentions update')

    def handle_mentions(self):
        for mention in self.state['mention_queue']:
            prefix = self.get_mention_prefix(mention)
            text = " ".join([w for w in mention.text.split() if '@' not in w])
            query = (text.lower()
                         .replace("-", "+-")
                         .replace(" ", "")
                         .split("+"))
            signs = [(-1) if "-" in q
                     else 1
                     for q in query]
            not_in_vocab = self.check_vocabulary(query)
            if len(not_in_vocab) != 0:
                self.log.info("Not in vocabulary: "+", ".join(not_in_vocab))
                reply = prefix+" Sorry, "+", ".join(not_in_vocab)+" not in my vocabulary."
                self.post_tweet(reply, reply_to=mention)
            reply = self.formulate_reply(prefix, text, query, signs)
            self.log.info("About to post "+reply)
            self.post_tweet(reply, reply_to=mention)
            self.state['mention_queue'].remove(mention)

    def get_mention_prefix(self, tweet):
        """
        Returns a string of users to @-mention when responding to a tweet.
        """
        mention_back = ['@' + tweet.author.screen_name]
        mention_back += [s for s in re.split('[^@\w]', tweet.text)
                         if len(s) > 2
                         and s[0] == '@'
                         and s[1:] != self.screen_name]
        return ' '.join(mention_back)

    def formulate_reply(self, prefix, text, query, signs):
        if len(query) == 1:
            result = w2vengine.get_synonyms(query[0], 5)
            result = ", ".join(result)
            reply = "Related terms to "+query[0]+": "+result
        else:
            result = w2vengine.get_abstractions(query, signs)
            reply = " = ".join(["TEST", text, result])
        return " ".join([prefix, reply])

    def check_vocabulary(self, query):
        query = [q.replace("-", "") for q in query]
        q = set(query)
        diff = q.difference(w2vengine.vocabulary)
        return list(diff)

    def _tweet_url(self, tweet):
        return "http://twitter.com/" + tweet.author.screen_name + "/status/" + str(tweet.id)

    def post_tweet(self, text, reply_to=None, media=None):
        kwargs = {}
        args = [text]
        if media is not None:
            cmd = self.api.update_with_media
            args.insert(0, media)
        else:
            cmd = self.api.update_status

        try:
            self.log.info('Tweeting "{}"'.format(text))
            if reply_to:
                self.log.info("-- Responding to status {}".format(self._tweet_url(reply_to)))
                kwargs['in_reply_to_status_id'] = reply_to.id
            else:
                self.log.info("-- Posting to own timeline")

            tweet = cmd(*args, **kwargs)
            self.log.info('Status posted at {}'.format(self._tweet_url(tweet)))
            return True

        except tweepy.TweepError as e:
            self.log.error('Can\'t post status')
            self.log.error(e)
            return False

    def run(self):
        while True:
            # check mentions every minute-ish
            # if self.reply_to_mentions and (time.time() - self.last_mention_time) > 60:
            if (time.time() - self.state.get('last_mention_time', 0)) > 60:
                self.check_mentions()
                self.handle_mentions()

            # save current state
            self.save_state()
            self.log.info("Sleeping for a bit...")
            time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do stuff')
    parser.add_argument('--config', dest='configpath', help='relative or '
                        'absolute path of the config file')
    parser.add_argument('--model', dest='modelpath', help='relative or '
                        'absolute path of the model')
    args = parser.parse_args()
    spark = init_spark_context()
    bot = EmbedBot(spark, args.configpath, args.modelpath)
    bot.initialize()
    bot.run()
