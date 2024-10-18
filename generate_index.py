import os
import re
import sys, getopt
import json
import pandas as pd
from datasets import Dataset

import pyterrier as pt
pt.init()

from Batcher import Batcher
import pyt_splade

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# import boto3
# from botocore.exceptions import NoCredentialsError

import rds_config
import pymysql
import logging

from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#rds settings
rds_host  = rds_config.db_host
name = rds_config.db_user
password = rds_config.db_password
db_name = rds_config.db_name

try:
    conn = pymysql.connect(host=rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit()

def get_all_canvasSiteId():
    query_str = "SELECT DISTINCT(canvasSiteId) FROM videos;"
    with conn.cursor() as cur:
        cur.execute(query_str)
        canvasId_lyst = cur.fetchall()
    canvasId_lyst = [id[0] for id in canvasId_lyst]
    return canvasId_lyst


def access_database(canvas_id):
    # Query database and retrieve course documents corresponding to canvas_id
    query_str = "SELECT video_id, transcript FROM videos WHERE canvasSiteId=" + str(canvas_id) + ";"
    with conn.cursor() as cur:
        cur.execute(query_str)
        res = cur.fetchall()

    transcripts = dict(res)

    # Return transcripts or print error if database is empty for canvas_id
    if not len(transcripts):
        print("Course ID Does Not Have Any Corresponding Content Stored In Database")
        return None
    else:
        return transcripts


def read_youtube(video_id, document, sentences, lectures):
    current_word = 0
    words = document['results']['items']
    total_words = len(words)
    sentences[video_id] = []

    # Break course document down into individual sentences
    for sent in sent_tokenize(document['results']['transcripts'][0]['transcript']):
        times = []
        # Find timestamps corresponding to tokens in sentence
        for i in range(current_word, total_words):
            if words[i]['type'] == 'pronunciation':
                if words[i]['alternatives'][0]['content'] in sent:
                    times.append(words[i]['start_time'])
                    current_word += 1
                else:
                    break
            else:
                current_word += 1

        # If sentence is not empty, return sentence and corresponding sentence start time
        if len(times):
            sentences[video_id].append((sent, times[0]))

    # Retrieve full length transcript
    lectures[video_id] = document['results']['transcripts'][0]['transcript']

def read_leccap_1(video_id, document, sentences, lectures):
    current_word = 0
    words = document['transcript']['timedtext']
    total_words = len(words)
    sentences[video_id] = []

    # Break course document down into individual sentences
    for sent in sent_tokenize(document['transcript']['full_transcript']):
        times = []
        # Find timestamps corresponding to tokens in sentence
        for i in range(current_word, total_words):
            if words[i]['words'] in sent:
                times.append(str(words[i]['start_time']))
                current_word += 1
            else:
                break

        # If sentence is not empty, return sentence and corresponding sentence start time
        if len(times):
            sentences[video_id].append((sent, times[0]))

    # Retrieve full length transcript
    lectures[video_id] = document['transcript']['full_transcript']

def read_leccap_2(video_id, document, sentences, lectures):
    current_word = 0
    words = document['timedtext']
    total_words = len(words)
    sentences[video_id] = []

    # Break course document down into individual sentences
    for sent in sent_tokenize(document['full_transcript']):
        times = []
        # Find timestamps corresponding to tokens in sentence
        for i in range(current_word, total_words):
            if words[i]['word'] in sent:
                times.append(str(words[i]['start_time']))
                current_word += 1
            else:
                break

        # If sentence is not empty, return sentence and corresponding sentence start time
        if len(times):
            sentences[video_id].append((sent, times[0]))

    # Retrieve full length transcript
    lectures[video_id] = document['full_transcript']

def read_transcripts(transcripts):
    sentences, lectures = {}, {}

    # Load transcripts from database according to json format...
    for video_id, transcript in transcripts.items():
        document = json.loads(transcript, strict=False)
        if 'results' in document:
            read_youtube(video_id, document, sentences, lectures)
        elif "transcript" in document:
            read_leccap_1(video_id, document, sentences, lectures)
        elif "timedtext" in document:
            read_leccap_2(video_id, document, sentences, lectures)

    # Return full transcripts and individual sentences for each course document
    return lectures, sentences


def transform_lectures(transcripts, sentences):
    # Reformat data
    transcripts = [[str(key), transcripts[key]] for key in transcripts.keys()]
    sentences = [["_".join([str(key), str(i)]), sentence, time] for key, value in sentences.items() for i, (sentence, time) in enumerate(value)]

    # Convert data to pandas dataframes
    transcripts = pd.DataFrame(transcripts, columns=["docno", "text"])
    sentences = pd.DataFrame(sentences, columns=["docno", "text", "time"])

    return transcripts, sentences


def generate_indexes(canvas_id, transcripts, sentences):
    transcript_index_path = '/home/ec2-user/terrier-search/terrier_index/index/' + str(canvas_id) + '/Transcripts'

    # Perform PyTerrier magic and generate index for full transcripts
    transcripts_dataset = Dataset.from_pandas(transcripts)
    if not os.path.exists(transcript_index_path + "/data.properties"):
        iter_indexer = pt.IterDictIndexer(transcript_index_path, meta=['docno', 'text'])
        indxr_pipe = iter_indexer
        # iter_indexer.index(transcripts_dataset)
        indxr_pipe.index(transcripts_dataset)

    sentence_index_path = '/home/ec2-user/terrier-search/index/' + str(canvas_id) + '/Sentences'

    # Perform PyTerrier magic and generate index for sentences
    sentences_dataset = Dataset.from_pandas(sentences)
    if not os.path.exists(sentence_index_path + "/data.properties"):
        iter_indexer = pt.IterDictIndexer(sentence_index_path, meta=['docno', 'time', 'text'])
        iter_indexer.index(sentences_dataset)

    return transcript_index_path, sentence_index_path


def generate_passage_index(canvas_id, transcripts, sentences):
    transcript_index_path = '/home/ec2-user/terrier-search/terrier_index/passage_index/' + str(canvas_id) + '/Transcripts'

    # Perform PyTerrier magic and generate index for full transcripts
    transcripts_dataset = Dataset.from_pandas(transcripts)
    if not os.path.exists(transcript_index_path + "/data.properties"):
        iter_indexer = pt.IterDictIndexer(transcript_index_path, meta=['docno', 'text'])
        indxr_pipe = pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None) >> \
                     iter_indexer
        indxr_pipe.index(transcripts_dataset)

    sentence_index_path = '/home/ec2-user/terrier-search/passage_index/' + str(canvas_id) + '/Sentences'

    # Perform PyTerrier magic and generate index for sentences
    sentences_dataset = Dataset.from_pandas(sentences)
    if not os.path.exists(sentence_index_path + "/data.properties"):
        iter_indexer = pt.IterDictIndexer(sentence_index_path, meta=['docno', 'time', 'text'])
        iter_indexer.index(sentences_dataset)

    return transcript_index_path, sentence_index_path


def generate_splade_indexes(canvas_id, transcripts, sentences):
    # Generate temporary index folder for full transcripts
    # /tmp/ required for lambda

    splade = pyt_splade.SpladeFactory()
    transcript_index_path = '/home/ec2-user/terrier-search/terrier_index/splade_index/' + str(canvas_id) + '/Transcripts'
    # transcript_index_path = '/home/ec2-user/terrier-search/splade_index/' + str(0000) + '/Transcripts'
    # Perform PyTerrier magic and generate index for full transcripts
    transcripts_dataset = Dataset.from_pandas(transcripts)
    if not os.path.exists(transcript_index_path + "/data.properties"):
        iter_indexer = pt.IterDictIndexer(transcript_index_path, meta=['docno', 'text'])
        # indxr_pipe = pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None) >> splade.indexing() >> iter_indexer
        # indxr_pipe = splade.indexing() >> iter_indexer
        indxr_pipe = pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None) >> \
                     Batcher(splade.indexing(), batch_size=64) >> iter_indexer
        index_ref = indxr_pipe.index(transcripts_dataset)

    # Generate temporary index folder for sentences
    # /tmp/ required for lambda
    sentence_index_path = '/home/ec2-user/terrier-search/splade_index/' + str(canvas_id) + '/Sentences'
    # sentence_index_path = '/home/ec2-user/terrier-search/splade_index/' + str(0000) + '/Sentences'
    # Perform PyTerrier magic and generate index for sentences
    sentences_dataset = Dataset.from_pandas(sentences)
    if not os.path.exists(sentence_index_path + "/data.properties"):
        iter_indexer = pt.IterDictIndexer(sentence_index_path, meta=['docno', 'time', 'text'], batch_size=128)
        # indxr_pipe = splade.indexing() >> iter_indexer
        indxr_pipe = Batcher(splade.indexing(), batch_size=64) >> iter_indexer
        index_ref = indxr_pipe.index(sentences_dataset)

    return transcript_index_path, sentence_index_path


if __name__ == '__main__':
    canvasId_lyst = get_all_canvasSiteId()
    print("Number of canvasSiteId: {}".format(len(canvasId_lyst)))

    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, "h-m:", ["model="])
    except getopt.GetoptError:
        print('generate_index.py -m <model>')
        sys.exit(2)

    model = ""
    if opts:
        model = opts[0][1]

    # canvasId_lyst = [197]
    # canvasId_lyst = [547504]

    for canvas_id in tqdm(canvasId_lyst):
        print("=============={}==============".format(canvas_id))
        transcripts = access_database(canvas_id)

        if transcripts:
            transcripts, sentences = read_transcripts(transcripts)
            transcripts, sentences = transform_lectures(transcripts, sentences)

            # save the dataframe
            # transcripts.to_csv('./index/' + str(canvas_id) + '/transcripts.csv', index=None)
            if model == "splade" or model == "s":
                transcript_index_path, sentence_index_path = generate_splade_indexes(canvas_id, transcripts, sentences)
            elif model == "passage" or model == "p":
                transcript_index_path, sentence_index_path = generate_passage_index(canvas_id, transcripts, sentences)
            else:
                transcript_index_path, sentence_index_path = generate_indexes(canvas_id, transcripts, sentences)
            # save meta data into csv
            sen_path = '/home/ec2-user/terrier-search/terrier_index/metadata/' + str(canvas_id)
            if not os.path.exists(sen_path):
                os.makedirs(sen_path)

            sentences.to_csv(sen_path + '/sentences.csv', index=None)
