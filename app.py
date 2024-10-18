from flask import Flask, request, redirect, url_for
from flask import Response, jsonify, make_response

from pyterrier_search import pyterrier_search

import pyt_splade
splade = pyt_splade.SpladeFactory()

import logging
from logging.handlers import WatchedFileHandler
import datetime
from flask_cors import CORS

from flask import json

app = Flask(__name__)
CORS(app)

import pyterrier_colbert.ranking
checkpoint = "/home/ec2-user/terrier-search/model/colbert.dnn"
global colbert
colbert = pyterrier_colbert.ranking.ColBERTModelOnlyFactory(checkpoint)

import pyterrier as pt
if (not pt.started()):
    pt.init()

from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
monoT5 = MonoT5ReRanker()
# duoT5 = DuoT5ReRanker()
duoT5 = None
# connect to database
import pymysql
import rds_config

rds_host  = rds_config.db_host
name = rds_config.db_user
password = rds_config.db_password
db_name = rds_config.db_name

try:
    conn = pymysql.connect(host=rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    conn.ping(reconnect=True)


@app.before_first_request
def setup_logging():
    """
    Setup logging
    """
    handler = WatchedFileHandler("/home/ec2-user/myapp/error.log")
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

@app.route('/')
def hello_world():
    return "Test terrier model! 2022/12/01 Test Splade"

@app.route('/search', methods=['GET'])
def search():
    app.logger.info('flask search start')
    start = datetime.datetime.now()

    id = request.args.get('id')
    query = request.args.get('query')
    pipeline = request.args.get('pipeline')
    model = request.args.get('model')
    agg = request.args.get('agg')
    

    if not query:
        return '<h2>Error: please enter your query </h2>'
    
    if not pipeline:
        pipeline = "a"

    if not model:
        model = 'bm25'

    if not agg:
        agg = 'max'

    print("use model: {}".format(model))
    body = pyterrier_search(query, id, pipeline, model, agg, colbert, splade, monoT5, duoT5, conn)
    
    if body == "invalid pipeline":
        return '<h2>Error: invalid pipeline</h2>'
    if body == "invalid agg":
       return '<h2>Error: aggregation function not exists, please type max, first, mean or kmax(replace k with a number such as 10max) </h2>'
    if body == "invalid model":
       return '<h2>Error: invalid model</h2>'

    end = datetime.datetime.now() 
    if model == 'colbert':
        app.logger.info('Flask colbert use {}'.format(end - start))
    elif model == 'bm25':
        app.logger.info('Flask bm25 use {}'.format(end - start))
    
    res = app.response_class(
        response=json.dumps(body),
        status=200,
        mimetype='application/json'
    )
    # app.logger.info("===========body============")
    #  resapp.logger.info(body)
    return res

if __name__ == '__main__':
    app.run(debug=True)


