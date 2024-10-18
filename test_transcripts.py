import sys
import datetime
import faiss
assert faiss.get_num_gpus() > 0
import os

import pyterrier as pt
pt.init()

import pandas as pd
from datasets import Dataset

import pyterrier_colbert.ranking
from logging.handlers import WatchedFileHandler

import sys, getopt

import pandas as pd

import pickle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

checkpoint = "/home/ec2-user/terrier-search/model/colbert.dnn"

def generate_index():
    transcripts = pd.read_csv("/home/ec2-user/terrier-search/data/transcripts.csv")
    transcripts_dataset = Dataset.from_pandas(transcripts)
    if os.path.exists("/home/ec2-user/terrier-search/IterDict/data.properties"):
        return "/home/ec2-user/terrier-search/IterDict/data.properties"
    else:
        iter_indexer = pt.IterDictIndexer("./IterDict", meta=['docno', 'text'])
        iter_index = iter_indexer.index(transcripts_dataset)
        return iter_index

def search(index, search_term):
    colbert = pyterrier_colbert.ranking.ColBERTModelOnlyFactory(checkpoint)

    pipe = (pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
            >> pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None)
            >> colbert.text_scorer()
            >> pt.text.max_passage())

    return pipe.search(search_term)


def perform_bm25_index_sliding(query, transcript_index, sentence_index):
    transcript_pipe = pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])

    transcript_results = transcript_pipe.search(query)

    sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
                     )

    sentence_results = sentence_pipe.search(query)

    return transcript_results, sentence_results


def perform_bm25_search_sliding(query, transcript_index, sentence_index):
    transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                       >> pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None)
                       >> pt.text.max_passage())

    # transcript_pipe = pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])

    transcript_results = transcript_pipe.search(query)

    sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
                     )

    sentence_results = sentence_pipe.search(query)

    return transcript_results, sentence_results


def perform_colbert_index_sliding(query, transcript_index, sentence_index):
    # Instantiate PyTerrier search object with TF-IDF weighting
    colbert = pyterrier_colbert.ranking.ColBERTModelOnlyFactory(checkpoint)

    transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
            >> colbert.text_scorer())

    transcript_results = transcript_pipe.search(query)

    sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
            >> colbert.text_scorer())

    sentence_results = sentence_pipe.search(query)
    return transcript_results, sentence_results


def perform_colbert_search_sliding(query, transcript_index, sentence_index):
    # Instantiate PyTerrier search object with TF-IDF weighting
    colbert = pyterrier_colbert.ranking.ColBERTModelOnlyFactory(checkpoint)

    transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
            >> pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None)
            >> colbert.text_scorer()
            >> pt.text.max_passage())

    transcript_results = transcript_pipe.search(query)

    sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
            >> colbert.text_scorer())

    sentence_results = sentence_pipe.search(query)
    return transcript_results, sentence_results


def perform_search_splade(query, transcript_index, sentence_index):
    # Instantiate PyTerrier search object with TF-IDF weighting
    import pyt_splade
    splade = pyt_splade.SpladeFactory()
    transcript_pipe = (splade.query()
            >> pt.BatchRetrieve(transcript_index, wmodel="Tf", metadata=["docno", "text"]))

    transcript_results = transcript_pipe.search(query)

    sentence_pipe = (splade.query()
            >> pt.BatchRetrieve(sentence_index, wmodel="Tf", metadata=["docno", "text"]))

    sentence_results = sentence_pipe.search(query)
    return transcript_results, sentence_results


if __name__ == '__main__':
    # iter_index = generate_index()
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, "h-m:", ["model="])
    except getopt.GetoptError:
        print('test.py -m <model>')
        sys.exit(2)

    model = ""
    if opts:
        model = opts[0][1]


    query = "cold front"
    canvas_id = 197

    transcript_index = "/home/ec2-user/terrier-search/index/" + str(canvas_id) + "/Transcripts"
    sentence_index = "/home/ec2-user/terrier-search/index/" + str(canvas_id) + "/Sentences"

    transcript_index_splade = "/home/ec2-user/terrier-search/splade_index/" + str(canvas_id) + "/Transcripts"
    sentence_index_splade = "/home/ec2-user/terrier-search/splade_index/" + str(canvas_id) + "/Sentences"

    canvas_id = 90000
    transcript_index_sliding = "/home/ec2-user/terrier-search/index/" + str(canvas_id) + "/Transcripts"
    sentence_index_sliding = "/home/ec2-user/terrier-search/index/" + str(canvas_id) + "/Sentences"

    # # bm25 index sliding window
    # print("================bm25 index sliding window================")
    # transcript_results, sentence_results = perform_bm25_index_sliding(query, transcript_index_sliding,
    #                                                                   sentence_index_sliding)
    # # transcript_results.to_excel("test_output/output_bm25_index_sliding_window.xlsx")
    #
    # # bm25 search sliding window
    # print("================bm25 search sliding window================")
    # transcript_results, sentence_results = perform_bm25_search_sliding(query, transcript_index,
    #                                                                   sentence_index)
    # print(transcript_results)
    # # transcript_results.to_excel("test_output/output_bm25_search_sliding_window.xlsx")
    #
    # # bm25 search sliding window with SPLADE index
    # print("================bm25 search splade================")
    # transcript_results, sentence_results = perform_bm25_search_sliding(query, transcript_index_splade,
    #                                                                    sentence_index_splade)
    # print(transcript_results)
    # # transcript_results.to_excel("test_output/output_bm25_splade.xlsx")
    #
    # # colbert index sliding window
    # print("================colbert index sliding window================")
    # transcript_results, sentence_results = perform_colbert_index_sliding(query, transcript_index_sliding,
    #                                                                   sentence_index_sliding)
    # # transcript_results.to_excel("test_output/output_colbert_index_sliding_window.xlsx")
    #
    # # colbert search sliding window
    # print("================colbert search sliding window================")
    # transcript_results, sentence_results = perform_colbert_search_sliding(query, transcript_index,
    #                                                                      sentence_index)
    # print(transcript_results)
    # # transcript_results.to_excel("test_output/output_colbert_search_sliding_window.xlsx")
    #
    # # colbert search sliding window with SPLADE index
    print("================colbert search splade================")
    transcript_results, sentence_results = perform_search_splade(query, transcript_index_splade,
                                                                          sentence_index_splade)
    print(transcript_results)
    print(sentence_results)
    # transcript_results.to_excel("test_output/output_colbert_splade.xlsx")
