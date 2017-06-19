#!/bin/bash -xe
sudo pip install -U \
spark \
pandas \
py4j \
tweepy
python -m nltk.downloader punkt
