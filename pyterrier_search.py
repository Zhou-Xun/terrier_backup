# Thomas Horak (thorames), Chris Lee (chrisree), Xun Zhou (xunzhou)
# updated_pyterrier_search.py

import pyterrier as pt
import pandas as pd
import datetime
import json
import rds_config

import logging
import sys, getopt

if (not pt.started()):
    pt.init()

logging.basicConfig(filename="/home/ec2-user/myapp/pyterrier_error.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Running Pyterrier search")

logger = logging.getLogger()

#rds settings
# rds_host  = rds_config.db_host
# name = rds_config.db_user
# password = rds_config.db_password
# db_name = rds_config.db_name

# import pymysql
# try:
#     conn = pymysql.connect(host=rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
# except pymysql.MySQLError as e:
#     conn.ping(reconnect=True)


def agg_to_pt(agg):
    if agg == "max":
        return pt.text.max_passage()
    elif agg == "first":
        return pt.text.first_passage()
    elif agg == "mean":
        return pt.text.mean_passage()
    elif agg[-3:] == "max":
        return pt.text.kmaxavg_passage(int(agg[:-3]))
    else:
        return "invalid agg"


def perform_search_baseline(query, canvas_id):
    logging.info('perform_search_baseline pipeline A start')
    transcript_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Transcripts"
    sentence_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Sentences"
    # build transcript pipeline
    transcript_pipe = pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
    # search transcript
    transcript_results = transcript_pipe.search(query)
    # build sentence pipeline
    sentence_pipe = pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
    # search sentence
    sentence_results = sentence_pipe.search(query)
    return transcript_results, sentence_results


def perform_search_bm25(query, canvas_id, pipeline, agg):
    logging.info('perform_search_bm25 start')
    start = datetime.datetime.now()

    agg_func = agg_to_pt(agg)
    if agg_func == "invalid agg":
        return agg_func

    if pipeline == "b":
        logging.info('exec pipeline B')
        transcript_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Transcripts"
        sentence_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Sentences"
        # build transcript pipeline
        transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                           >> pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None)
                           >> pt.BatchRetrieve(transcript_index, wmodel="BM25")
                           >> agg_func)
        # search transcript
        transcript_results = transcript_pipe.search(query)
        # build sentence pipeline
        sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"]))
        # search sentence
        sentence_results = sentence_pipe.search(query)
    elif pipeline == "c":
        logging.info('exec pipeline C')
        transcript_index = "/home/ec2-user/terrier-search/terrier_index/passage_index/" + str(canvas_id) + "/Transcripts"
        sentence_index = "/home/ec2-user/terrier-search/terrier_index/passage_index/" + str(canvas_id) + "/Sentences"
        # build transcript pipeline
        transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                           >> agg_func)
        # search transcript
        transcript_results = transcript_pipe.search(query)
        # build sentence pipeline
        sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"]))
        # search sentence
        sentence_results = sentence_pipe.search(query)
    else:
        return "invalid pipeline"

    logging.info('bm25 pipeline search use {}'.format(datetime.datetime.now() - start))
    return transcript_results, sentence_results


def perform_search_colbert(query, canvas_id, pipeline, agg, colbert):
    # Instantiate PyTerrier search object with TF-IDF weighting
    logging.info('perform_search_colbert start')
    start = datetime.datetime.now()

    agg_func = agg_to_pt(agg)
    if agg_func == "invalid agg":
        return agg_func

    logging.info('using aggregation function: {}'.format(agg))

    if pipeline == "b":
        logging.info('exec pipeline B')
        transcript_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Transcripts"
        sentence_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Sentences"
        # generate transcript pipeline
        transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                           >> pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None)
                           >> colbert.text_scorer()
                           >> agg_func)
        # search transcript
        transcript_results = transcript_pipe.search(query)
        # generate sentence pipeline
        sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
                         >> colbert.text_scorer())

        sentence_results = sentence_pipe.search(query)
    elif pipeline == "c":
        logging.info('exec pipeline C')
        transcript_index = "/home/ec2-user/terrier-search/terrier_index/passage_index/" + str(canvas_id) + "/Transcripts"
        sentence_index = "/home/ec2-user/terrier-search/terrier_index/passage_index/" + str(canvas_id) + "/Sentences"
        # generate transcript pipeline
        transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                           >> colbert.text_scorer()
                           >> agg_func)
        # search transcript
        transcript_results = transcript_pipe.search(query)
        # generate sentence pipeline
        sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
                         >> colbert.text_scorer())
        sentence_results = sentence_pipe.search(query)
    else:
        return "invalid pipeline"

    logging.info('Colbert pipeline search use {}'.format(datetime.datetime.now() - start))
    return transcript_results, sentence_results


def perform_search_monot5(query, canvas_id, pipeline, agg, monoT5, duoT5):
    logging.info('perform_search_monot5 start')

    agg_func = agg_to_pt(agg)
    if agg_func == "invalid agg":
        return agg_func

    logging.info('using aggregation function: {}'.format(agg))

    if pipeline == "b":
        logging.info('exec pipeline B')
        transcript_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Transcripts"
        sentence_index = "/home/ec2-user/terrier-search/terrier_index/index/" + str(canvas_id) + "/Sentences"
        # generate transcript pipeline
        transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                           >> pt.text.sliding(text_attr='text', length=128, stride=64, prepend_attr=None)
                           >> monoT5
                           >> agg_func)
        # search transcript
        transcript_results = transcript_pipe.search(query)
        # generate sentence pipeline
        sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
                         >> monoT5)
        # search sentence
        sentence_results = sentence_pipe.search(query)
    elif pipeline == 'c':
        logging.info('exec pipeline C')
        transcript_index = "/home/ec2-user/terrier-search/terrier_index/passage_index/" + str(canvas_id) + "/Transcripts"
        sentence_index = "/home/ec2-user/terrier-search/terrier_index/passage_index/" + str(canvas_id) + "/Sentences"
        # generate transcript pipeline
        transcript_pipe = (pt.BatchRetrieve(transcript_index, wmodel="BM25", metadata=["docno", "text"])
                           >> monoT5
                           >> agg_func)
        # search transcript
        transcript_results = transcript_pipe.search(query)
        # generate sentence pipeline
        sentence_pipe = (pt.BatchRetrieve(sentence_index, wmodel="BM25", metadata=["docno", "text"])
                         >> monoT5)
        # search sentence
        sentence_results = sentence_pipe.search(query)
    else:
        return "invalid pipeline"

    return transcript_results, sentence_results


def perform_search_splade(query, canvas_id, pipeline, splade, agg):
    # Instantiate PyTerrier search object with TF-IDF weighting
    logging.info('perform_search_splade start')
    # logging.info(transcript_index)
    # logging.info(sentence_index)
    start = datetime.datetime.now()

    transcript_index = "/home/ec2-user/terrier-search/terrier_index/splade_index/" + str(canvas_id) + "/Transcripts"
    sentence_index = "/home/ec2-user/terrier-search/terrier_index/splade_index/" + str(canvas_id) + "/Sentences"

    agg_func = agg_to_pt(agg)
    if agg_func == "invalid agg":
        return agg_func

    if pipeline == "b":
        return "invalid model"
    elif pipeline == 'c':
        logging.info('exec pipeline C')
        transcript_pipe = (splade.query()
                >> pt.BatchRetrieve(transcript_index, wmodel="Tf", metadata=["docno", "text"])
                >> agg_func)

        transcript_results = transcript_pipe.search(query)

        sentence_pipe = (splade.query()
                >> pt.BatchRetrieve(sentence_index, wmodel="Tf", metadata=["docno", "text"]))

        sentence_results = sentence_pipe.search(query)
    else:
        return "invalid pipeline"

    logging.info('Splade pipeline search use {}'.format(datetime.datetime.now() - start))
    return transcript_results, sentence_results


def return_results(query, transcript_results, sentence_results, canvas_id, conn):
    search_results = {"recordings": {query: []}}

    # Reformat sentence search results
    sentence_df = pd.read_csv('/home/ec2-user/terrier-search/terrier_index/metadata/'
                              + str(canvas_id) + '/sentences.csv')
    sentence_scores = sentence_results[['docno', 'score']].values.tolist()

    lecture_moments = {}
    query_len = len(query.split())

    # Loop over sentence search results in order of relevance to query

    for docno, score in sentence_scores:

        # Threshold to limit number of sentences returned
        ## tobefixed: remove score threshold
        # if score >= query_len - 0.5:
        lecture_name = "_".join(docno.split("_")[:-1])
        # logging.info("=========iterating sentence df docno=========")
        # logging.info(docno)
        sentence, time = sentence_df[sentence_df['docno']==docno][["text", "time"]].values.tolist()[0]
        if lecture_name in lecture_moments:
            lecture_moments[lecture_name].append({"context": sentence, "timestamp": time, "score": score})
        else:
            lecture_moments[lecture_name] = [{"context": sentence, "timestamp": time, "score": score}]

    # Reformat transcript search results
    transcript_scores = transcript_results[['docno', 'score']].values.tolist()

    # Loop over lectures and assign ordered sentence search results as well as corresponding video info for front end
    docno_lyst = []
    doc_scores = {}
    for docno, score in transcript_scores:
        if docno in lecture_moments:
            docno_lyst.append(docno)
            doc_scores[docno] = score

    vid_info = {}
    if not docno_lyst:
        return "No overlap between transcript results and sentence results"
    # Retrieve videos from database
    query_str = "SELECT * FROM videos WHERE canvasSiteId={} AND video_id IN ({})".format(canvas_id, str(docno_lyst)[1:-1])

    conn.ping(reconnect=True)

    with conn.cursor() as cur:
        cur.execute(query_str)
        videos = cur.fetchall()

    # Load each video transcript, process sentences and full text
    for video in videos:
        # translate to dot dictionary, to avoid refactor
        video_id = video[0]
        upload_src = video[1]
        title = video[2]
        video_link = video[3]
        date = video[5]
        image = video[7]

        vid_info[str(video_id)] = [title, image, video_link, date, upload_src]

    for docno in docno_lyst:
        vid_date = vid_info[docno][3].strftime("%m/%d/%Y, %H:%M:%S")
        result = {"vid_id": docno, "title": vid_info[docno][0], "vid_date": vid_date, "timestamps": lecture_moments[docno], "image": vid_info[docno][1], "video_link": vid_info[docno][2], "score": doc_scores[docno]}
        search_results["recordings"][query].append(result)

    return search_results


def respond(err, res=None):
    return {
        'statusCode': '400' if err else '200',
        'body': err.message if err else json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
        },
    }


def pyterrier_search(query, canvas_id, pipeline, model, agg, colbert, splade=None, monoT5=None, duoT5=None, conn=None):
    # Set java virtual machine path (important for PyTerrier)
    # Initialize PyTerrier

    # Retrieve indexes from S3 and corresponding video info
    # transcript_index, sentence_index = retrieve_indexes(canvas_id)
    #vid_info = pull_vid_info(class_id)

    # Perform search
    if pipeline not in ['a', 'b', 'c', 'd']:
        return "invalid pipeline"

    if pipeline == "a":
        results = perform_search_baseline(query, canvas_id);
    elif model == "bm25":
        results = perform_search_bm25(query, canvas_id, pipeline, agg)
    elif model == "colbert":
        results = perform_search_colbert(query, canvas_id, pipeline, agg, colbert)
    elif model == "monot5":
        results = perform_search_monot5(query, canvas_id, pipeline, agg, monoT5, duoT5)
    elif model == "splade":
        results = perform_search_splade(query, canvas_id, pipeline, splade, agg)
    else:
        results = "invalid model"

    # if the input aggregation function is valid
    if results in ["invalid agg", "invalid model"]:
        return results
    else:
        transcript_results, sentence_results = results
    #logging.info(transcript_results['docno'])
    vid_result = return_results(query, transcript_results, sentence_results, canvas_id, conn)

    response = {
        "canvasSiteId": canvas_id,
        "vid_result": vid_result,
    }
    
    return response


if __name__ == '__main__':

    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, "hi:q:-m:", ["id=", "query=", "model="])
    except getopt.GetoptError:
        print('test.py -i <id> -q <query>')
        sys.exit(2)

    model = "bm25"
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <id> -q <query>')
            sys.exit()
        elif opt in ("-i", "--id"):
            canvasId = arg
        elif opt in ("-q", "--query"):
            query = arg
        elif opt in ("-m", "--model"):
            model = arg

    res = pyterrier_search(query, canvasId, model)

    # with open('/home/ec2-user/terrier-search/result/{}_{}_{}.json'.format(canvasId, query, model), 'w') as f:
    #     json.dump(res, f)
