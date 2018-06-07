# encoding:utf-8
"""
==========================
calculation of similarity
==========================

use charactor  'tag', 'director', 'country',
    'cast', 'language', 'year', 'grade_score'
To redis it is that result of calculation be saved

Created by:
    yonggang Huang
On:
    03-31-2017
Modified:
    05-30-2018
"""

import json
import cPickle as pickle
from numpy import load
from datetime import date
from main import Sim
from util import get_logger, get_redis_client
import traceback
import argparse
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

MODELS = ['movie', 'tv',
          'sports', 'entertainment', 'variety',
          'education', 'doc', 'cartoon']
WEIGHT = {'movie': {'language': 0.1, 'country': 0.5,
                    'tag': 0.9, 'cast': 0.3, 'director': 0.8,
                    'grade_score': 0.4, 'year': 0.4},
          'tv': {'language': 0.9, 'country': 0.9,
                 'tag': 0.5, 'cast': 1, 'director': 0.9,
                 'grade_score': 0.7, 'year': 0.7, },
          "cartoon": {"country": 0.5, "tag": 0.9, "year": 0.7, }, }

KEY_PATTERN = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:{}'

DEBUG = False

curDate = date.today().strftime("%Y-%m-%d")
logger = get_logger('./log/similar_{0}.log'.format(curDate))


def main(config, mode, data_file_path, ):

    def load_data(model, static=False):
        data = dict()
        logger.info('Load {} data'.format(model))
        data = dict()
        for feat in WEIGHT[model].keys():
            if static:
                data[feat] = (data_file_path + r'/' + model + "_" + feat + ".npy")
            else:
                data[feat] = load(data_file_path + r'/' + model + "_" + feat + ".npy")
        id_list = load(data_file_path + r'/' + model + "_" + "id_" + ".npy")
        logger.info('the features of {}: {}'.format(model, data.keys()))
        return data, id_list

    def train():
        for model in MODELS:
            if model not in ['tv', 'movie', "cartoon"]:
                continue
            data, ids = load_data(model, static=True)
            logger.info('Init sim handler')
            s = Sim(data, ids, static_=True)
            logger.info('Search weight from douban')
            logger.info('Load {} data from douban '.format(model))
            with open(data_file_path + r'/' + model + r'_douban.dat', 'rb') as f:
                dataset = pickle.load(f)
            logger.info('Searching weight ...')
            w, score = s.weight_search(dataset, patch_size=200, verbose=True)
            logger.info('Finished weight search, socre: {}, weight: {}'.format(score, w))

    def predict():

        logger.info('Use history weight: {}'.format(WEIGHT))
        logger.info("Initting connection to redis")
        con = get_redis_client(config)
        for model in MODELS:
            if model not in ["tv", "movie", "cartoon", ]:
                continue
            data, ids = load_data(model, static=True)
            logger.info('Init sim handler')
            s = Sim(data, ids, weight=WEIGHT[model], static_=True)
            count = 0
            try:
                for cover_id, result in s.process():
                    count += 1
                    if DEBUG:
                        logger.info('{}  {}'.format(cover_id, result))
                        continue
                    con.set(KEY_PATTERN.format(cover_id), json.dumps(result))
                    con.expire(KEY_PATTERN.format(cover_id), 2592000)
            except Exception, e:
                logger.error('catched error :{}, processed num: {}, model: {}'.format(e, count, model))
                traceback.print_exc()
                raise Exception('Error')
            logger.info('Num of {} result : {}'.format(model, count))
            print('Num of {} result : {}'.format(model, count))
        logger.info('Finished')

    if mode == "predict":
        predict()
    elif mode == "train":
        train()
    else:
        raise ValueError('Mode Error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculation similarity between items\n")
    parser.add_argument("-config", "--configure", type=str,
                        default="etc/config.ini", help="configure file")
    parser.add_argument("-cache", default="data", help="work on debug mode")
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    parser.add_argument("mode", type=str, choices=["predict", "train"],
                        help="work mode <train / predict>")
    args = parser.parse_args()
    if not os.path.isfile(args.configure):
        raise Exception("Invalid configure file")
    if not os.path.isdir(args.cache):
        raise Exception("Invalid cache directory")
    DEBUG = True if args.debug else False
    main(args.configure, args.mode, args.cache)
