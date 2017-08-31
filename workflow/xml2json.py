from os import listdir
from os import path
import gzip
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import argparse
import logging
import tarfile

logging.basicConfig(filename='conversion.log',level=logging.DEBUG)

class Converter(object):
    """docstring for Converter"""
    def __init__(self, inputfolder, outputfolder, style, source, compressed=None):
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        self.style = self.load_style(style)
        self.source = source
        if compressed:
            self.compressed = compressed

    def load_style(self, style):
        with open(".".join([style, "json"]), "r") as infile:
            style = json.load(infile)
        return style

    def get_metadata(self, article):
        metadata = {}

        for md, xp in self.style.items():
            try:
                match = [e.text for e in article.findall(xp)]
                if len(match) == 1:
                    try:
                        metadata[md] = int(match[0])
                    except:
                        metadata[md] = match[0]
                else:
                    try:
                        metadata[md] = [int(m) for m in match]
                    except:
                        metadata[md] = match
            except:
                metadata[md] = None
        return metadata

    def get_texts(self, article):
        ps = article.findall(".//body//p")
        ps_abs = article.findall(".//abstract//p")
        text = " ".join("".join(p.itertext()) for p in ps)
        text_abs = " ".join("".join(p.itertext()) for p in ps_abs)
        return {"fulltext": text, "abstract": text_abs}

    def create_json(self, article):
        metadata = self.get_metadata(article)
        texts = self.get_texts(article)
        article_json = {}
        article_json.update(metadata)
        article_json.update(texts)
        return article_json

    def corexml2json(self, xmlfilepath):
        head, tail = path.split(xmlfilepath)
        filename = path.splitext(path.splitext(tail)[0])[0]
        if path.isfile(path.join(self.outputfolder, filename+".json")):
            return
        try:
            with gzip.open(xmlfilepath, "r") as infile:
                tree = ET.parse(infile)
                root = tree.getroot()
        except Exception as e:
            logging.exception(" ".join([str(e), xmlfilepath]))

        try:
            for article in root.getchildren():
                with open(path.join(self.outputfolder,
                                    filename+".json"),
                          "a") as outfile:
                    jsonfile = self.create_json(article)
                    outfile.write(json.dumps(jsonfile)+"\n")
        except Exception as e:
            logging.exception(" ".join([str(e), xmlfilepath]))

    def plosxml2json(self, xmlfilepath):
        head, tail = path.split(xmlfilepath)
        filename = path.splitext(path.splitext(tail)[0])[0]
        if path.isfile(path.join(self.outputfolder, filename+".json")):
            return
        if self.compressed == "gz":
            with tarfile.open(xmlfilepath, "r:gz") as infile:
                infile = infile
        else:
            with open(xmlfilepath, "r") as infile:
                infile = infile
        i = 0
        for line in infile.readlines()[:10]:
            # skip badly formatted XML headers, start parsing at <article tag
            if line.startswith("<article"):
                break
            i += 1
        article = infile.readlines()[i:]
        article = "".join(article)
        try:
            article = ET.fromstring(article)

            with open(path.join(self.outputfolder, filename+".json"), "a") as outfile:
                jsonfile = self.create_json(article)
                outfile.write(json.dumps(jsonfile)+"\n")
        except Exception as e:
            logging.exception(" ".join([str(e), xmlfilepath]))

    def core2json(self):
        files = [f for f in listdir(self.inputfolder)
                 if path.isfile(path.join(self.inputfolder, f))]
        for file in files:
            self.corexml2json(path.join(self.inputfolder, file))

    def plos2json(self):
        files = [f for f in listdir(self.inputfolder)
                 if path.isfile(path.join(self.inputfolder, f))]
        for file in files:
            self.plosxml2json(path.join(self.inputfolder, file))

def main(args):
    converter = Converter(args.inputfolder, args.outputfolder, args.style, args.source, args.compressed)
    if args.source in ['CORE', 'EUPMC']:
        converter.core2json()
    if args.source in ['PLOS']:
        converter.plos2json()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert publisher collections'
                                        ' to JSON dumps ready for spark-input')
    parser.add_argument('--style', dest='style',
                        help='style of the XML source, one of [JATS]')
    parser.add_argument('--source', dest='source',
                        help='from which publisher the collection comes from,'
                             ' one of [CORE, PLOS, EUPMC]')
    parser.add_argument('--input', dest='inputfolder',
                        help='relative or absolute path of the input folder')
    parser.add_argument('--output', dest='outputfolder',
                        help='relative or absolute path of the output folder')
    parser.add_argument('--compressed', dest='compressed',
                        help='flag if inputfiles are compressed')
    args = parser.parse_args()
    main(args)
